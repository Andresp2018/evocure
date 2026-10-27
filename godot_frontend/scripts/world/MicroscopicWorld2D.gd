class_name MicroscopicWorld2D
extends Control
## 2D Microscopic World Simulation Viewport for EVOCURE.
## Features SubViewport rendering, microenvironment background shader with biophysical presets,
## 60 FPS interpolated creature & cell entity pools, glowing antitumor energy beams,
## interactive Camera2D with zoom/pan, MinimapControl, and scientific HUD overlays.

const WORLD_WIDTH: float = 1280.0
const WORLD_HEIGHT: float = 720.0
const STROMA_NICHE_RECT := Rect2(880.0, 80.0, 360.0, 560.0)

# Nodes
var _subviewport_container: SubViewportContainer = null
var _subviewport: SubViewport = null
var _world_root: Node2D = null
var _backdrop_sprite: Sprite2D = null
var _backdrop_mat: ShaderMaterial = null
var _heatmap_overlay: Node2D = null
var _beams_overlay: Node2D = null
var _camera: Camera2D = null

# Pools
var _creature_nodes: Dictionary = {} # id -> CreatureRig2D
var _cell_nodes: Dictionary = {}     # id -> BioCellNode2D
var _preview_creatures: Array[CreatureRig2D] = []
var _preview_cells: Array[BioCellNode2D] = []

# HUD Elements
var _hud_container: Control = null
var _pill_label: Label = null
var _tumor_badge: PanelContainer = null
var _tumor_label: Label = null
var _minimap: MinimapControl = null
var _scale_bar: Control = null

# Camera interaction state
var _zoom_target: float = 1.0
var _is_panning: bool = false
var _pan_start_mouse: Vector2 = Vector2.ZERO
var _pan_start_cam_pos: Vector2 = Vector2.ZERO

# Shaders and State
var _current_heatmap_mode: String = "none"
var _beam_timer: float = 0.0
var _beam_connections: Array[Dictionary] = []
var _seen_reactions: Dictionary = {}
var _reaction_effects: Array[Dictionary] = []


func _sync_reaction_events(state: Dictionary, env_w: float, env_h: float) -> void:
	for event in state.get("recent_events", []):
		if not event is Dictionary or event.get("type", "") != "REACTION_OCCURRED":
			continue
		var key: String = JSON.stringify(event)
		if _seen_reactions.has(key):
			continue
		_seen_reactions[key] = true
		_reaction_effects.append({"position": Vector2(70.0 + float(event.get("x", 0)) / env_w * 770.0, 75.0 + float(event.get("y", 0)) / env_h * 580.0), "life": 1.0, "label": str(event.get("reaction_type", "reaction")).replace("_", " ")})
	if _seen_reactions.size() > 500:
		_seen_reactions.clear()


func _update_reaction_effects(delta: float) -> void:
	for effect in _reaction_effects:
		effect["life"] = float(effect["life"]) - delta * 0.6
	_reaction_effects = _reaction_effects.filter(func(e: Dictionary) -> bool: return float(e["life"]) > 0.0)
	_beams_overlay.queue_redraw()


func _ready() -> void:
	clip_contents = true
	_build_viewport_world()
	_build_floating_hud()
	_connect_app_signals()
	if not AppState.latest_state.is_empty():
		_on_state_updated(AppState.latest_state)


func _build_viewport_world() -> void:
	_subviewport_container = SubViewportContainer.new()
	_subviewport_container.name = "WorldContainer"
	_subviewport_container.stretch = true
	_subviewport_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ThemeBuilder.fill_rect(_subviewport_container)
	add_child(_subviewport_container)

	_subviewport = SubViewport.new()
	_subviewport.name = "SimulationViewport"
	_subviewport.size = Vector2i(int(WORLD_WIDTH), int(WORLD_HEIGHT))
	_subviewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	_subviewport.handle_input_locally = false
	_subviewport_container.add_child(_subviewport)

	_world_root = Node2D.new()
	_world_root.name = "World2D"
	_subviewport.add_child(_world_root)

	# 1. Microenvironment Backdrop Sprite with Organic Fluid Shader
	_backdrop_sprite = Sprite2D.new()
	_backdrop_sprite.name = "BackdropSprite"
	var bg_tex: Texture2D = load("res://assets/textures/microenvironment_backdrop.png") as Texture2D
	if bg_tex != null:
		_backdrop_sprite.texture = bg_tex
		# Backdrop texture is 1920x1080; scale to cover 1280x720
		_backdrop_sprite.scale = Vector2(WORLD_WIDTH / 1920.0, WORLD_HEIGHT / 1080.0)
		_backdrop_sprite.position = Vector2(WORLD_WIDTH * 0.5, WORLD_HEIGHT * 0.5)

	var bg_shader: Shader = load("res://assets/shaders/microenvironment_bg.gdshader") as Shader
	if bg_shader != null:
		_backdrop_mat = ShaderMaterial.new()
		_backdrop_mat.shader = bg_shader
		_backdrop_sprite.material = _backdrop_mat
	_world_root.add_child(_backdrop_sprite)

	# 2. PDE Heatmap Overlay
	_heatmap_overlay = Node2D.new()
	_heatmap_overlay.name = "HeatmapOverlay"
	_heatmap_overlay.draw.connect(_on_draw_heatmap_overlay)
	_world_root.add_child(_heatmap_overlay)

	# 3. Antitumor Energy Beams Overlay
	_beams_overlay = Node2D.new()
	_beams_overlay.name = "BeamsOverlay"
	_beams_overlay.draw.connect(_on_draw_beams_overlay)
	_world_root.add_child(_beams_overlay)

	# 4. Interactive Camera2D
	_camera = Camera2D.new()
	_camera.name = "MicroCamera2D"
	_camera.position = Vector2(WORLD_WIDTH * 0.5, WORLD_HEIGHT * 0.5)
	_camera.zoom = Vector2(1.0, 1.0)
	_world_root.add_child(_camera)


func _build_floating_hud() -> void:
	_hud_container = Control.new()
	_hud_container.name = "FloatingHUD"
	ThemeBuilder.fill_rect(_hud_container)
	_hud_container.mouse_filter = MOUSE_FILTER_IGNORE
	add_child(_hud_container)

	# Top-left telemetry glass pill
	var top_pill := PanelContainer.new()
	top_pill.name = "TopTelemetryPill"
	var sb_pill := StyleBoxFlat.new()
	sb_pill.bg_color = Color(0.027, 0.051, 0.106, 0.88)
	sb_pill.border_color = Color(0.133, 0.827, 0.933, 0.75)
	sb_pill.set_border_width_all(1)
	sb_pill.set_corner_radius_all(14)
	sb_pill.content_margin_left = 14
	sb_pill.content_margin_right = 14
	sb_pill.content_margin_top = 6
	sb_pill.content_margin_bottom = 6
	top_pill.add_theme_stylebox_override("panel", sb_pill)
	top_pill.position = Vector2(18, 16)
	_hud_container.add_child(top_pill)

	_pill_label = Label.new()
	_pill_label.name = "PillLabel"
	_pill_label.text = "Time: 0.0 yrs   |   Generation: 0   |   Entities: 0"
	_pill_label.add_theme_color_override("font_color", Color(0.918, 0.957, 1.0))
	_pill_label.add_theme_font_size_override("font_size", 12)
	top_pill.add_child(_pill_label)

	# Floating crimson tumor badge (near right stromal niche)
	_tumor_badge = PanelContainer.new()
	_tumor_badge.name = "TumorCalloutBadge"
	var sb_tumor := StyleBoxFlat.new()
	sb_tumor.bg_color = Color(0.25, 0.05, 0.10, 0.88)
	sb_tumor.border_color = Color(0.957, 0.247, 0.369, 0.95)
	sb_tumor.set_border_width_all(1)
	sb_tumor.set_corner_radius_all(10)
	sb_tumor.content_margin_left = 12
	sb_tumor.content_margin_right = 12
	sb_tumor.content_margin_top = 5
	sb_tumor.content_margin_bottom = 5
	_tumor_badge.add_theme_stylebox_override("panel", sb_tumor)
	_tumor_badge.anchor_left = 1.0
	_tumor_badge.anchor_right = 1.0
	_tumor_badge.offset_left = -270
	_tumor_badge.offset_right = -20
	_tumor_badge.offset_top = 16
	_tumor_badge.offset_bottom = 46
	_hud_container.add_child(_tumor_badge)

	_tumor_label = Label.new()
	_tumor_label.name = "TumorLabel"
	_tumor_label.text = "Pancreatic Tumor (PDAC)"
	_tumor_label.add_theme_color_override("font_color", Color(1.0, 0.45, 0.55))
	_tumor_label.add_theme_font_size_override("font_size", 12)
	_tumor_badge.add_child(_tumor_label)

	# Bottom-left MinimapControl
	_minimap = MinimapControl.new()
	_minimap.name = "HUDMinimap"
	_minimap.anchor_top = 1.0
	_minimap.anchor_bottom = 1.0
	_minimap.offset_left = 18
	_minimap.offset_right = 194
	_minimap.offset_top = -126
	_minimap.offset_bottom = -18
	_minimap.camera_jump_requested.connect(_on_minimap_camera_jump)
	_hud_container.add_child(_minimap)

	# Bottom-right zoom buttons & scale bar
	_build_bottom_right_controls()


func _build_bottom_right_controls() -> void:
	var zoom_panel := PanelContainer.new()
	zoom_panel.name = "ZoomPanel"
	var sb_z := StyleBoxFlat.new()
	sb_z.bg_color = Color(0.027, 0.051, 0.106, 0.85)
	sb_z.border_color = Color(0.165, 0.286, 0.467, 0.65)
	sb_z.set_border_width_all(1)
	sb_z.set_corner_radius_all(8)
	sb_z.content_margin_left = 6
	sb_z.content_margin_right = 6
	sb_z.content_margin_top = 4
	sb_z.content_margin_bottom = 4
	zoom_panel.add_theme_stylebox_override("panel", sb_z)

	zoom_panel.anchor_left = 1.0
	zoom_panel.anchor_top = 1.0
	zoom_panel.anchor_right = 1.0
	zoom_panel.anchor_bottom = 1.0
	zoom_panel.offset_left = -215
	zoom_panel.offset_right = -18
	zoom_panel.offset_top = -54
	zoom_panel.offset_bottom = -18
	_hud_container.add_child(zoom_panel)

	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 6)
	zoom_panel.add_child(hbox)

	var btn_in := Button.new()
	btn_in.text = "+"
	btn_in.custom_minimum_size = Vector2(28, 26)
	btn_in.tooltip_text = "Zoom In (Mouse Wheel Up)"
	btn_in.pressed.connect(zoom_in)
	hbox.add_child(btn_in)

	var btn_out := Button.new()
	btn_out.text = "-"
	btn_out.custom_minimum_size = Vector2(28, 26)
	btn_out.tooltip_text = "Zoom Out (Mouse Wheel Down)"
	btn_out.pressed.connect(zoom_out)
	hbox.add_child(btn_out)

	var btn_fit := Button.new()
	btn_fit.text = "[ ]"
	btn_fit.custom_minimum_size = Vector2(30, 26)
	btn_fit.tooltip_text = "Reset View & Camera"
	btn_fit.pressed.connect(reset_camera)
	hbox.add_child(btn_fit)

	# Scale bar Control
	_scale_bar = Control.new()
	_scale_bar.custom_minimum_size = Vector2(72, 26)
	_scale_bar.draw.connect(_on_draw_scale_bar)
	hbox.add_child(_scale_bar)


func _on_draw_scale_bar() -> void:
	if _scale_bar == null:
		return
	var w: float = 58.0
	var y: float = 16.0
	var x_start: float = 6.0
	var x_end: float = x_start + w
	var col := Color(0.918, 0.957, 1.0, 0.9)

	# Horizontal bar
	_scale_bar.draw_line(Vector2(x_start, y), Vector2(x_end, y), col, 1.8, true)
	# Left tick
	_scale_bar.draw_line(Vector2(x_start, y - 4), Vector2(x_start, y + 4), col, 1.8, true)
	# Right tick
	_scale_bar.draw_line(Vector2(x_end, y - 4), Vector2(x_end, y + 4), col, 1.8, true)

	var font: Font = ThemeDB.fallback_font
	if font != null:
		_scale_bar.draw_string(font, Vector2(x_start + 10.0, y - 5.0), "sim units", HORIZONTAL_ALIGNMENT_LEFT, -1, 10, col)


func _get_app_state() -> Node:
	if is_inside_tree() and get_tree().root.has_node("AppState"):
		return get_tree().root.get_node("AppState")
	return null


func _get_api_client() -> Node:
	if is_inside_tree() and get_tree().root.has_node("ApiClient"):
		return get_tree().root.get_node("ApiClient")
	return null


func _connect_app_signals() -> void:
	var st: Node = _get_app_state()
	if st != null:
		if st.has_signal("state_updated") and not st.state_updated.is_connected(_on_state_updated):
			st.state_updated.connect(_on_state_updated)
		if st.has_signal("environment_preset_changed") and not st.environment_preset_changed.is_connected(_on_environment_preset_changed):
			st.environment_preset_changed.connect(_on_environment_preset_changed)
		if st.has_signal("entity_selected") and not st.entity_selected.is_connected(_on_entity_selected_in_app):
			st.entity_selected.connect(_on_entity_selected_in_app)


func _on_environment_preset_changed(preset_id: String) -> void:
	if _backdrop_mat == null:
		return
	var tint := Color(1.0, 1.0, 1.0, 1.0)
	match preset_id:
		"pancreatic_tissue":
			tint = Color(1.0, 0.98, 0.96, 1.0)
		"bloodstream":
			tint = Color(1.15, 0.88, 0.88, 1.0)
		"tumor_microenvironment":
			tint = Color(1.12, 0.78, 0.85, 1.0)
		"oxidative_stress":
			tint = Color(1.12, 1.04, 0.76, 1.0)
		"acidic_low_ph":
			tint = Color(0.88, 1.08, 0.78, 1.0)
		"hypoxia_low_o2":
			tint = Color(0.72, 0.88, 1.18, 1.0)
		_:
			tint = Color(1.0, 1.0, 1.0, 1.0)
	_backdrop_mat.set_shader_parameter("env_tint", tint)


func _on_state_updated(state: Dictionary) -> void:
	var env_dict: Dictionary = state.get("environment", {}) if state.get("environment") is Dictionary else {}
	var env_w: float = maxf(100.0, float(env_dict.get("width", 1000.0)))
	var env_h: float = maxf(100.0, float(env_dict.get("height", 700.0)))

	# Render the complete small ecosystem; clearly report sampling above the cap.
	var raw_entities: Array = state.get("entities", []) if state.get("entities") is Array else []
	var selected_entities: Array = []
	var stride: int = maxi(1, int(ceil(float(maxi(1, raw_entities.size())) / 180.0)))
	for idx in range(0, raw_entities.size(), stride):
		if selected_entities.size() >= 180:
			break
		var cand: Variant = raw_entities[idx]
		if cand is Dictionary and not selected_entities.has(cand):
			selected_entities.append(cand)

	if not _preview_creatures.is_empty() and selected_entities.size() > 0:
		for rig in _preview_creatures:
			rig.queue_free()
		_preview_creatures.clear()

	var active_ids := {}
	for ent in selected_entities:
		var ed: Dictionary = (ent as Dictionary).duplicate()
		var eid: String = str(ed.get("entity_id", ed.get("id", ed.get("name", ""))))
		if eid.is_empty():
			continue
		active_ids[eid] = true
		ed["entity_id"] = eid
		ed["structure"] = state.get("structure_catalog", {}).get(ed.get("compound_id", ""), {})

		# Map simulation physics coordinates [0..env_w, 0..env_h] into fluid chamber [70..845, 75..655]
		var rx: float = float(ed.get("x", 400.0))
		var ry: float = float(ed.get("y", 350.0))
		var wx: float = 70.0 + clampf(rx / env_w, 0.02, 0.98) * 770.0
		var wy: float = 75.0 + clampf(ry / env_h, 0.03, 0.97) * 580.0
		var k_val: String = str(ed.get("entity_kind", ed.get("k", "")))
		ed["x"] = wx
		ed["y"] = wy

		if _creature_nodes.has(eid):
			var rig: CreatureRig2D = _creature_nodes[eid]
			rig.setup_entity(ed)
		else:
			var rig := CreatureRig2D.new()
			rig.name = "Creature_" + eid
			rig.entity_data = ed
			_world_root.add_child(rig)
			_creature_nodes[eid] = rig

	# Remove dead creature nodes
	var to_remove := []
	for eid in _creature_nodes.keys():
		if not active_ids.has(eid):
			to_remove.append(eid)
	for eid in to_remove:
		var rig: CreatureRig2D = _creature_nodes[eid]
		_creature_nodes.erase(eid)
		rig.queue_free()

	# Sync tumor & stromal cells into STROMA_NICHE_RECT [895..1225, 95..635]
	var tumor: Dictionary = state.get("tumor", {}) if state.get("tumor") is Dictionary else {}
	if tumor.has("cells") and tumor["cells"] is Array:
		var raw_cells: Array = tumor["cells"]
		var dom: Dictionary = tumor.get("domain", {}) if tumor.get("domain") is Dictionary else {}
		var dom_w: float = maxf(100.0, float(dom.get("width", 800.0)))
		var dom_h: float = maxf(100.0, float(dom.get("height", 600.0)))

		var alive_cells: Array = []
		for rc in raw_cells:
			if rc is Dictionary:
				var st_mode: String = str(rc.get("st", rc.get("death_mode", "alive"))).to_lower()
				if st_mode == "alive":
					alive_cells.append(rc)

		var cell_stride: int = maxi(1, int(ceil(float(maxi(1, alive_cells.size())) / 36.0)))
		var sampled_cells: Array = []
		for c_idx in range(0, alive_cells.size(), cell_stride):
			if sampled_cells.size() >= 36:
				break
			sampled_cells.append(alive_cells[c_idx])

		if not _preview_cells.is_empty() and sampled_cells.size() > 0:
			for cell in _preview_cells:
				cell.queue_free()
			_preview_cells.clear()

		var active_cell_ids := {}
		for c_i in range(sampled_cells.size()):
			var cdata: Dictionary = (sampled_cells[c_i] as Dictionary).duplicate()
			var cid: String = str(cdata.get("cell_id", cdata.get("id", "cell_" + str(c_i))))
			if cid.is_empty():
				continue
			active_cell_ids[cid] = true

			var raw_ct: String = str(cdata.get("cell_type", cdata.get("t", "tumor_pdac"))).to_lower()
			var mapped_ct: String = raw_ct
			if raw_ct.contains("basal") or raw_ct.contains("persister") or float(cdata.get("res", 0.0)) >= 0.45:
				mapped_ct = "tumor_resistant"
			elif raw_ct.contains("tumor") or raw_ct.contains("classical") or raw_ct.contains("cancer"):
				mapped_ct = "tumor_pdac"
			elif raw_ct.contains("caf") or raw_ct.contains("fibro") or raw_ct.contains("stroma"):
				mapped_ct = "stroma_caf"
			elif raw_ct.contains("immune") or raw_ct.contains("macro") or raw_ct.contains("t_cell"):
				mapped_ct = "immune"
			elif raw_ct.contains("healthy") or raw_ct.contains("epithelial"):
				mapped_ct = "healthy"

			var cx_raw: float = float(cdata.get("x", dom_w * 0.5))
			var cy_raw: float = float(cdata.get("y", dom_h * 0.5))
			cdata["cell_id"] = cid
			cdata["cell_type"] = mapped_ct
			cdata["clone_id"] = str(cdata.get("clone_id", cdata.get("cl", "CLONE")))
			cdata["viability"] = float(cdata.get("viability", cdata.get("v", 1.0)))
			cdata["is_resistant"] = (mapped_ct == "tumor_resistant")
			var disease: String = str(tumor.get("disease_id", "PDAC"))
			if disease == "AMR_BACTERIAL" and raw_ct in ["tumor_sensitive", "tumor_resistant"]:
				cdata["cell_type"] = "bacteria_amr"
			elif disease == "METABOLIC_MASLD" and raw_ct in ["tumor_sensitive", "tumor_resistant", "epithelial_healthy"]:
				cdata["cell_type"] = "hepatocyte_masld"
			elif disease == "NSCLC_KRAS" and raw_ct == "tumor_sensitive":
				cdata["cell_type"] = "tumor_nsclc"
			cdata["x"] = 895.0 + clampf(cx_raw / dom_w, 0.04, 0.96) * 325.0
			cdata["y"] = 95.0 + clampf(cy_raw / dom_h, 0.04, 0.96) * 535.0

			if _cell_nodes.has(cid):
				var node: BioCellNode2D = _cell_nodes[cid]
				node.setup_cell(cdata)
			else:
				var node := BioCellNode2D.new()
				node.name = "Cell_" + cid
				node.cell_data = cdata
				_world_root.add_child(node)
				_cell_nodes[cid] = node

		var dead_cells := []
		for cid in _cell_nodes.keys():
			if not active_cell_ids.has(cid):
				dead_cells.append(cid)
		for cid in dead_cells:
			var node: BioCellNode2D = _cell_nodes[cid]
			_cell_nodes.erase(cid)
			node.queue_free()

	_update_tumor_badge_text()
	_sync_reaction_events(state, env_w, env_h)
	_heatmap_overlay.queue_redraw()


func _update_tumor_badge_text() -> void:
	if _tumor_label == null:
		return
	var st: Node = _get_app_state()
	var dis_val: Variant = st.get("active_disease") if st != null else null
	var dis: String = str(dis_val).to_upper() if dis_val != null else "PDAC"
	_tumor_label.text = {"PDAC": "Pancreatic tumour (PDAC)", "NSCLC_KRAS": "Lung tumour (NSCLC)", "AMR_BACTERIAL": "Bacterial biofilm (AMR)", "METABOLIC_MASLD": "Liver tissue (MASLD)"}.get(dis, dis)


func _on_entity_selected_in_app(entity_data: Dictionary) -> void:
	var sel_id: String = str(entity_data.get("entity_id", entity_data.get("name", entity_data.get("cell_id", ""))))
	for eid in _creature_nodes.keys():
		var rig: CreatureRig2D = _creature_nodes[eid]
		rig.set_selected(eid == sel_id)
	for cid in _cell_nodes.keys():
		var cell: BioCellNode2D = _cell_nodes[cid]
		cell.set_selected(cid == sel_id)


func _process(delta: float) -> void:
	if _camera == null:
		return
	_beam_timer += delta

	# Smooth camera zoom
	if absf(_camera.zoom.x - _zoom_target) > 0.002:
		var cur_z: float = lerpf(_camera.zoom.x, _zoom_target, clampf(10.0 * delta, 0.0, 1.0))
		_camera.zoom = Vector2(cur_z, cur_z)

	# Update HUD pill
	var st: Node = _get_app_state()
	var clocks_val: Variant = st.get("clocks") if st != null else null
	var clocks: Dictionary = clocks_val if clocks_val is Dictionary else {}
	var yr: float = float(clocks.get("virtual_years", 0.0))
	var gen: int = int(clocks.get("evolutionary_generation", 0))
	var ent_count: int = _creature_nodes.size() if _creature_nodes.size() > 0 else _preview_creatures.size()
	if st != null and st.get("latest_state") is Dictionary:
		var ls: Dictionary = st.get("latest_state")
		ent_count = maxi(ent_count, int(ls.get("total_entity_count", ent_count)))
	if _pill_label != null:
		_pill_label.text = "Gen %d  |  %d entities (%d shown)  |  %d reactions" % [gen, ent_count, _creature_nodes.size(), int(AppState.latest_state.get("reactions_count", 0))]

	# Update Antitumor Energy Beams
	_update_reaction_effects(delta)

	# Update Minimap
	_update_minimap_data()


func _on_draw_beams_overlay() -> void:
	for effect in _reaction_effects:
		var life: float = float(effect["life"])
		var pos: Vector2 = effect["position"]
		_beams_overlay.draw_arc(pos, 12.0 + (1.0 - life) * 45.0, 0, TAU, 32, Color(0.3, 1.0, 0.8, life), 2.0, true)
		_beams_overlay.draw_string(ThemeDB.fallback_font, pos + Vector2(10, -20), str(effect["label"]), HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.6, 1.0, 0.9, life))


func _update_minimap_data() -> void:
	if _minimap == null or _camera == null:
		return

	var creatures_arr: Array = _creature_nodes.values() if _creature_nodes.size() > 0 else _preview_creatures
	var cells_arr: Array = _cell_nodes.values() if _cell_nodes.size() > 0 else _preview_cells

	# Calculate visible camera viewport in world coordinates
	var vp_size: Vector2 = Vector2(WORLD_WIDTH, WORLD_HEIGHT) / _camera.zoom
	var cam_rect := Rect2(_camera.position - vp_size * 0.5, vp_size)
	var world_bounds := Rect2(0, 0, WORLD_WIDTH, WORLD_HEIGHT)

	_minimap.update_minimap(creatures_arr, cells_arr, cam_rect, world_bounds)


func _on_minimap_camera_jump(world_pos: Vector2) -> void:
	if _camera != null:
		_camera.position = Vector2(
			clampf(world_pos.x, 200.0, WORLD_WIDTH - 200.0),
			clampf(world_pos.y, 150.0, WORLD_HEIGHT - 150.0)
		)


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb: InputEventMouseButton = event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			zoom_in()
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			zoom_out()
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_MIDDLE or mb.button_index == MOUSE_BUTTON_RIGHT:
			_is_panning = mb.pressed
			if mb.pressed:
				_pan_start_mouse = mb.position
				_pan_start_cam_pos = _camera.position
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_LEFT and mb.pressed:
			_handle_left_click(mb.position)
			accept_event()

	elif event is InputEventMouseMotion:
		var mm: InputEventMouseMotion = event as InputEventMouseMotion
		if _is_panning:
			var diff: Vector2 = (mm.position - _pan_start_mouse) / _camera.zoom
			_camera.position = _pan_start_cam_pos - diff
			_camera.position.x = clampf(_camera.position.x, 100.0, WORLD_WIDTH - 100.0)
			_camera.position.y = clampf(_camera.position.y, 100.0, WORLD_HEIGHT - 100.0)
			accept_event()


func _screen_to_world(screen_pos: Vector2) -> Vector2:
	var vp_size: Vector2 = size
	if vp_size.x <= 0 or vp_size.y <= 0:
		vp_size = Vector2(WORLD_WIDTH, WORLD_HEIGHT)
	var norm: Vector2 = screen_pos / vp_size
	var cam_half: Vector2 = (Vector2(WORLD_WIDTH, WORLD_HEIGHT) / _camera.zoom) * 0.5
	var cam_top_left: Vector2 = _camera.position - cam_half
	return cam_top_left + norm * (cam_half * 2.0)


func _handle_left_click(screen_pos: Vector2) -> void:
	var world_pos: Vector2 = _screen_to_world(screen_pos)
	var st: Node = _get_app_state()

	# 1. If click-to-spawn compound is armed
	var spawn_val: Variant = st.get("active_spawn_compound") if st != null else null
	var active_spawn: String = str(spawn_val) if spawn_val != null else ""
	if not active_spawn.strip_edges().is_empty():
		var qname: String = active_spawn
		var api: Node = _get_api_client()
		if api != null and api.has_method("spawn_compound"):
			api.call("spawn_compound", qname, 5, clampf((world_pos.x - 70.0) / 770.0, 0.0, 1.0) * float(AppState.environment.get("width", 1000)), clampf((world_pos.y - 75.0) / 580.0, 0.0, 1.0) * float(AppState.environment.get("height", 700)))
		if st != null:
			st.set("active_spawn_compound", "")
			if st.has_method("notify_toast"):
				st.call("notify_toast", "Spawned 5x " + qname + " at cursor coordinates", "success")
		return

	# 2. Hit-testing creatures
	var all_creatures := _creature_nodes.values()
	if all_creatures.is_empty():
		all_creatures = _preview_creatures

	for cre in all_creatures:
		if cre is CreatureRig2D:
			var rig: CreatureRig2D = cre as CreatureRig2D
			if rig.is_point_inside(world_pos):
				if st != null and st.has_method("select_entity"):
					st.call("select_entity", rig.entity_data)
				return

	# 3. Hit-testing cells
	var all_cells := _cell_nodes.values()
	if all_cells.is_empty():
		all_cells = _preview_cells

	for c in all_cells:
		if c is BioCellNode2D:
			var cell: BioCellNode2D = c as BioCellNode2D
			if cell.is_point_inside(world_pos):
				if st != null and st.has_method("select_entity"):
					st.call("select_entity", cell.cell_data)
				return


# --- Exact Public Methods ---

func zoom_in() -> void:
	_zoom_target = clampf(_zoom_target * 1.25, 0.55, 2.8)


func zoom_out() -> void:
	_zoom_target = clampf(_zoom_target / 1.25, 0.55, 2.8)


func reset_camera() -> void:
	_zoom_target = 1.0
	if _camera != null:
		_camera.position = Vector2(WORLD_WIDTH * 0.5, WORLD_HEIGHT * 0.5)


func set_heatmap_overlay(mode: String) -> void:
	_current_heatmap_mode = mode
	var st: Node = _get_app_state()
	if st != null:
		st.set("heatmap_overlay", mode)
	if _heatmap_overlay != null:
		_heatmap_overlay.queue_redraw()


func _on_draw_heatmap_overlay() -> void:
	if _current_heatmap_mode == "none" or _heatmap_overlay == null:
		return

	var state: Dictionary = AppState.latest_state
	var values: Array = []
	var area := Rect2(70, 75, 770, 580)
	var tint := Color(0.2, 0.9, 0.6)
	if _current_heatmap_mode == "substrate":
		values = state.get("substrate_heatmap", [])
	else:
		var key: String = {"oxygen": "oxygen", "ecm": "ecm_density", "drug": "drug_free"}.get(_current_heatmap_mode, "oxygen")
		values = state.get("tumor", {}).get("heatmaps", {}).get(key, [])
		area = Rect2(895, 95, 325, 535)
		tint = Color(0.4, 0.6, 1.0) if key == "oxygen" else Color(0.9, 0.5, 0.8)
	if values.is_empty() or not values[0] is Array or values[0].is_empty():
		return
	var peak: float = 0.001
	for row in values:
		for value in row:
			peak = maxf(peak, float(value))
	var cell := Vector2(area.size.x / values[0].size(), area.size.y / values.size())
	for y in range(values.size()):
		for x in range(values[y].size()):
			var color := Color(tint, clampf(float(values[y][x]) / peak, 0, 1) * 0.45)
			_heatmap_overlay.draw_rect(Rect2(area.position + Vector2(x, y) * cell, cell), color)
