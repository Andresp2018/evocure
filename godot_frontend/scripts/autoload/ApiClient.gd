extends Node
## Asynchronous HTTP REST + Real-Time WebSocket Client for EVOCURE Godot 4.7.
## Connects to the local isolated Python .venv backend (default 127.0.0.1:8000)
## and converts RDKit 2D SVG drawings directly into crisp Godot ImageTextures.

var base_http_url: String = "http://127.0.0.1:8000"
var base_ws_url: String = "ws://127.0.0.1:8000/ws"

var _ws: WebSocketPeer = WebSocketPeer.new()
var _ws_connected: bool = false
var _reconnect_timer: float = 0.0
var _poll_state_timer: float = 0.0
var _svg_cache: Dictionary = {}


func _ready() -> void:
	_parse_cli_overrides()
	_connect_websocket()
	# Initial bootstrap queries
	call_deferred("refresh_bootstrap_data")


func _parse_cli_overrides() -> void:
	var all_args: PackedStringArray = OS.get_cmdline_args()
	all_args.append_array(OS.get_cmdline_user_args())
	for arg in all_args:
		if arg.begins_with("--backend-url="):
			base_http_url = arg.trim_prefix("--backend-url=").rstrip("/")
			var host_part: String = base_http_url.replace("https://", "").replace("http://", "")
			var proto: String = "wss://" if base_http_url.begins_with("https://") else "ws://"
			base_ws_url = proto + host_part + "/ws"


func refresh_bootstrap_data() -> void:
	fetch_state()
	fetch_library()
	fetch_diseases()
	fetch_ai_providers()
	fetch_missions_glossary()


func _process(delta: float) -> void:
	_ws.poll()
	var ws_state: WebSocketPeer.State = _ws.get_ready_state()

	if ws_state == WebSocketPeer.STATE_OPEN:
		if not _ws_connected:
			_ws_connected = true
			AppState.backend_connected = true
		while _ws.get_available_packet_count() > 0:
			var raw: PackedByteArray = _ws.get_packet()
			var txt: String = raw.get_string_from_utf8()
			var parsed: Variant = JSON.parse_string(txt)
			if parsed is Dictionary:
				_handle_ws_message(parsed)
	elif ws_state == WebSocketPeer.STATE_CLOSED:
		AppState.backend_connected = false
		if _ws_connected:
			_ws_connected = false
		_reconnect_timer += delta
		if _reconnect_timer >= 2.5:
			_reconnect_timer = 0.0
			_connect_websocket()

	_poll_state_timer += delta
	var poll_interval: float = 1.8 if _ws_connected else 2.2
	if _poll_state_timer >= poll_interval:
		_poll_state_timer = 0.0
		if not AppState.is_playing or not _ws_connected:
			fetch_state()


func _connect_websocket() -> void:
	_ws = WebSocketPeer.new()
	_ws.inbound_buffer_size = 4 * 1024 * 1024
	_ws.outbound_buffer_size = 512 * 1024
	var err: int = _ws.connect_to_url(base_ws_url)
	if err != OK:
		_ws_connected = false


func _handle_ws_message(msg: Dictionary) -> void:
	var mtype: String = str(msg.get("type", ""))
	if mtype in ["init_state", "state_update", "world_reset", "compound_added", "preset_advanced", "disease_changed", "experiment_loaded", "checkpoint_restored", "universe_stepped", "tumor_compound_applied"]:
		var payload: Variant = msg.get("data", msg.get("payload", {}))
		if payload is Dictionary:
			AppState.ingest_server_state(payload)
	elif mtype == "tumor_stepped":
		if msg.get("tumor") is Dictionary:
			AppState.tumor_snapshot = msg["tumor"]
			AppState.state_updated.emit(AppState.latest_state)
	elif mtype == "compound_inspected":
		if msg.get("data") is Dictionary:
			AppState.inspected_compound = msg["data"]
			AppState.compound_inspected.emit(msg["data"])


func send_ws_json(payload: Dictionary) -> void:
	if _ws.get_ready_state() == WebSocketPeer.STATE_OPEN:
		_ws.send_text(JSON.stringify(payload))


func request_json(method: HTTPClient.Method, endpoint: String, body_dict: Variant = null, on_complete: Callable = Callable()) -> void:
	var req := HTTPRequest.new()
	req.timeout = 30.0
	add_child(req)

	req.request_completed.connect(func(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
		req.queue_free()
		var text: String = body.get_string_from_utf8()
		var parser := JSON.new()
		var parsed: Variant = parser.data if parser.parse(text) == OK else {"error": "Backend returned HTTP " + str(response_code)}
		if on_complete.is_valid():
			on_complete.call(response_code, parsed)
	)

	var url: String = base_http_url + endpoint
	var headers := PackedStringArray(["Content-Type: application/json", "Accept: application/json"])
	var payload_str: String = JSON.stringify(body_dict) if body_dict != null else ""
	var err: int = req.request(url, headers, method, payload_str)
	if err != OK:
		req.queue_free()
		if on_complete.is_valid():
			on_complete.call(0, {"error": "Request dispatch failed"})


func download_export_to_file(exp_id: String, fmt: String, save_path: String, on_complete: Callable = Callable()) -> void:
	var req := HTTPRequest.new()
	req.timeout = 45.0
	add_child(req)
	req.request_completed.connect(func(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
		req.queue_free()
		if response_code >= 200 and response_code < 300 and body.size() > 0:
			var f := FileAccess.open(save_path, FileAccess.WRITE)
			if f != null:
				f.store_buffer(body)
				f.close()
				AppState.notify_toast("Saved export to " + save_path, "success")
				if on_complete.is_valid():
					on_complete.call(true, save_path)
				return
		AppState.notify_toast("Export failed (HTTP " + str(response_code) + ")", "error")
		if on_complete.is_valid():
			on_complete.call(false, "")
	)
	req.request(base_http_url + "/api/v1/export/" + exp_id.uri_encode() + "/" + fmt.uri_encode())


# --- High-Level Domain Calls ---

func fetch_state() -> void:
	request_json(HTTPClient.METHOD_GET, "/api/v1/state", null, func(code: int, res: Variant) -> void:
		if code == 200 and res is Dictionary:
			AppState.ingest_server_state(res)
	)


func fetch_library() -> void:
	request_json(HTTPClient.METHOD_GET, "/api/v1/library", null, func(code: int, res: Variant) -> void:
		if code == 200 and res is Array:
			AppState.compound_library = res
			AppState.library_loaded.emit(res)
	)


func fetch_diseases() -> void:
	request_json(HTTPClient.METHOD_GET, "/api/v1/diseases", null, func(code: int, res: Variant) -> void:
		if code == 200 and res is Array:
			AppState.available_diseases = res
			AppState.diseases_loaded.emit(res)
	)


func fetch_missions_glossary() -> void:
	request_json(HTTPClient.METHOD_GET, "/api/v1/missions-glossary", null, func(code: int, res: Variant) -> void:
		if code == 200 and res is Dictionary:
			AppState.missions_glossary = res
	)


func fetch_ai_providers(extra_cb: Callable = Callable()) -> void:
	request_json(HTTPClient.METHOD_GET, "/api/v1/assistant/providers", null, func(code: int, res: Variant) -> void:
		if code == 200 and res is Dictionary:
			AppState.ai_providers_status = res
			AppState.ai_providers_updated.emit(res)
			if extra_cb.is_valid():
				extra_cb.call(res)
		else:
			# Fallback to /assistant/status if /providers not yet warm
			request_json(HTTPClient.METHOD_GET, "/api/v1/assistant/status", null, func(c2: int, r2: Variant) -> void:
				if c2 == 200 and r2 is Dictionary and extra_cb.is_valid():
					extra_cb.call(r2)
			)
	)


func configure_ai_provider(provider: String, model_name: String, api_key: String = "", base_url: String = "", persist_env: bool = false, cb: Callable = Callable()) -> void:
	var payload := {
		"provider": provider,
		"model": model_name,
		"persist_to_dotenv": persist_env,
	}
	if not api_key.strip_edges().is_empty():
		payload["api_key"] = api_key.strip_edges()
	if not base_url.strip_edges().is_empty():
		payload["base_url"] = base_url.strip_edges()

	request_json(HTTPClient.METHOD_POST, "/api/v1/assistant/providers/configure", payload, func(code: int, res: Variant) -> void:
		fetch_ai_providers()
		var inner: Variant = res.get("data", res) if res is Dictionary else res
		var ok_flag: bool = (code == 200) and (bool(res.get("success", true)) if res is Dictionary else false)
		if cb.is_valid():
			cb.call(ok_flag, inner)
	)


func test_ai_provider(provider: String, model_name: String = "", api_key: String = "", base_url: String = "", cb: Callable = Callable()) -> void:
	var payload := {"provider": provider}
	if not model_name.strip_edges().is_empty():
		payload["model"] = model_name.strip_edges()
	if not api_key.strip_edges().is_empty():
		payload["api_key"] = api_key.strip_edges()
	if not base_url.strip_edges().is_empty():
		payload["base_url"] = base_url.strip_edges()

	request_json(HTTPClient.METHOD_POST, "/api/v1/assistant/providers/test", payload, func(code: int, res: Variant) -> void:
		var inner: Variant = res.get("data", res) if res is Dictionary else res
		var ok_flag: bool = (code == 200) and (bool(res.get("success", true)) if res is Dictionary else false)
		if cb.is_valid():
			cb.call(ok_flag, inner)
	)


func set_playback(action: String, speed: int = 1, target: String = "universe") -> void:
	AppState.playback_speed = speed
	AppState.playback_target = target
	request_json(HTTPClient.METHOD_POST, "/api/v1/playback", {
		"action": action,
		"speed_multiplier": speed,
		"target": target,
	}, func(code: int, _res: Variant) -> void:
		if code == 200 and _res is Dictionary:
			var payload: Dictionary = _res.get("data", {})
			if payload.get("state") is Dictionary:
				AppState.ingest_server_state(payload["state"])
		elif code != 200:
			AppState.notify_toast("Playback failed. Check the local engine connection.", "error")
		if code != 200 and target == "both":
			request_json(HTTPClient.METHOD_POST, "/api/v1/playback", {
				"action": action,
				"speed_multiplier": speed,
				"target": "universe",
			})
	)


func advance_time_preset(preset: String, cb: Callable = Callable()) -> void:
	AppState.notify_toast("Simulating evolutionary jump (" + preset + ")...", "info")
	request_json(HTTPClient.METHOD_POST, "/api/v1/world/advance-preset", {"preset": preset}, func(code: int, res: Variant) -> void:
		if code == 200 and res is Dictionary:
			if res.get("data") is Dictionary:
				AppState.ingest_server_state(res["data"])
			AppState.notify_toast("Running " + preset.replace("_", " ") + " - pause or resume at any time.", "info")
		if cb.is_valid():
			cb.call(code == 200, res)
	)


func spawn_compound(query: String, count: int = 10, x: Variant = null, y: Variant = null, role: String = "autotroph", kind: String = "PROTOCELL") -> void:
	var body: Dictionary = {
		"query": query,
		"input_type": "auto",
		"count": count,
		"entity_kind": kind,
		"ecological_role": role,
	}
	if x != null and y != null:
		body["x"] = float(x)
		body["y"] = float(y)
	request_json(HTTPClient.METHOD_POST, "/api/v1/world/add-compound", body, func(code: int, res: Variant) -> void:
		if code == 200:
			AppState.notify_toast("Added " + str(count) + "x " + query + " to Universe", "success")
			fetch_state()
		else:
			AppState.notify_toast("Could not add compound: " + str(res), "error")
	)


func apply_environment_preset(preset_id: String) -> void:
	AppState.active_environment_preset = preset_id
	AppState.environment_preset_changed.emit(preset_id)
	request_json(HTTPClient.METHOD_POST, "/api/v1/environment/preset", {"preset": preset_id}, func(code: int, _res: Variant) -> void:
		if code == 200:
			fetch_state()
		else:
			# Fallback direct parameter patch
			var presets := {
				"pancreatic_tissue": {"temperature_k": 310.15, "ph": 7.35, "oxygen_norm": 0.75, "viscosity_pa_s": 0.0014},
				"bloodstream": {"temperature_k": 310.15, "ph": 7.40, "oxygen_norm": 1.25, "current_vx": 0.45},
				"tumor_microenvironment": {"temperature_k": 310.65, "ph": 6.65, "oxygen_norm": 0.32, "viscosity_pa_s": 0.0028},
				"oxidative_stress": {"temperature_k": 311.15, "ph": 7.10, "oxygen_norm": 1.65, "energy_source_flux": 2.2},
				"acidic_low_ph": {"temperature_k": 310.15, "ph": 6.15, "oxygen_norm": 0.65},
				"hypoxia_low_o2": {"temperature_k": 310.15, "ph": 6.80, "oxygen_norm": 0.14},
			}
			if presets.has(preset_id):
				request_json(HTTPClient.METHOD_POST, "/api/v1/environment", presets[preset_id], func(_c: int, _r: Variant) -> void:
					fetch_state()
				)
	)


func select_disease(disease_id: String) -> void:
	request_json(HTTPClient.METHOD_POST, "/api/v1/diseases/select", {"disease_id": disease_id}, func(code: int, res: Variant) -> void:
		if code == 200 and res is Dictionary and res.get("data") is Dictionary:
			var st: Variant = res["data"].get("state", {})
			if st is Dictionary and not st.is_empty():
				AppState.ingest_server_state(st)
			AppState.notify_toast("Switched active biomedical engine to " + disease_id, "info")
	)


func inspect_compound(query: String, enable_pbpk: bool = false, cb: Callable = Callable()) -> void:
	var url: String = "/api/v1/compound/" + query.uri_encode() + ("?enable_systemic_pbpk=true" if enable_pbpk else "")
	request_json(HTTPClient.METHOD_GET, url, null, func(code: int, res: Variant) -> void:
		if code == 200 and res is Dictionary:
			AppState.inspected_compound = res
			AppState.compound_inspected.emit(res)
			if cb.is_valid():
				cb.call(res)
	)


func fetch_entity_lineage(compound_id: String) -> void:
	request_json(HTTPClient.METHOD_GET, "/api/v1/entity-lineage/" + compound_id.uri_encode(), null, func(code: int, res: Variant) -> void:
		if code == 200 and res is Dictionary:
			var payload: Variant = res.get("data", res)
			if payload is Dictionary:
				AppState.selected_lineage_chain = payload.get("ancestors", payload.get("chain", []))
				AppState.lineage_loaded.emit(payload)
	)


func svg_to_texture(svg_text: String, raster_scale: float = 1.35, dark_adapt: bool = true) -> ImageTexture:
	if svg_text.strip_edges().is_empty():
		return null
	var cache_key: String = str(svg_text.hash()) + "_" + str(dark_adapt)
	if _svg_cache.has(cache_key):
		return _svg_cache[cache_key]

	var adapted: String = svg_text
	if dark_adapt:
		# Make RDKit white canvas subtle dark translucent slate and black strokes luminous ice-blue
		adapted = adapted.replace("fill:#FFFFFF;", "fill:#0B162C;")
		adapted = adapted.replace("fill='#FFFFFF'", "fill='#0B162C'")
		adapted = adapted.replace("stroke:#000000;", "stroke:#E2F1FF;")
		adapted = adapted.replace("fill:#000000;", "fill:#E2F1FF;")

	var img := Image.new()
	var err: int = img.load_svg_from_string(adapted, raster_scale)
	if err != OK or img.is_empty():
		return null
	var tex := ImageTexture.create_from_image(img)
	if _svg_cache.size() > 96:
		_svg_cache.clear()
	_svg_cache[cache_key] = tex
	return tex
