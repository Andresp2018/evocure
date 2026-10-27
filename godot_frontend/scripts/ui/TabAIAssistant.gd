class_name TabAIAssistant
extends Control

signal open_ai_settings_requested
var _messages: VBoxContainer
var _chat_scroll: ScrollContainer
var _input: LineEdit
var _send: Button
var _status: Label
var _title: LineEdit
var _saves: OptionButton
var _checkpoints: VBoxContainer
var _file_dialog: FileDialog
var _format: String = "json"
var _saved_id: String = ""
var _busy: bool = false

func _ready() -> void:
	ThemeBuilder.fill_rect(self)
	var columns := HBoxContainer.new()
	ThemeBuilder.fill_rect(columns)
	columns.add_theme_constant_override("separation", 12)
	add_child(columns)
	var left := VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(left)
	var heading := Label.new()
	heading.text = "CHAT · local scientific assistant"
	heading.add_theme_font_size_override("font_size", 18)
	left.add_child(heading)
	var settings := Button.new()
	settings.text = "AI settings"
	settings.pressed.connect(func() -> void: open_ai_settings_requested.emit())
	left.add_child(settings)
	_button(left, "Start optional local Gemma model", func() -> void:
		ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/assistant/start-local-llm", {}, func(code: int, result: Variant) -> void:
			_append("Local model", str(result.get("message", "Request finished")) if result is Dictionary else "Model start returned HTTP %d" % code)
		)
	)
	var chips := HFlowContainer.new()
	left.add_child(chips)
	for prompt in ["What is happening?", "Which family is dominant?", "What does mutation mean?", "How can I compare worlds?"]:
		var button := Button.new()
		button.text = prompt
		button.pressed.connect(func() -> void: _input.text = prompt; _send_message())
		chips.add_child(button)
	_chat_scroll = ScrollContainer.new()
	_chat_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_chat_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	left.add_child(_chat_scroll)
	_messages = VBoxContainer.new()
	_messages.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_chat_scroll.add_child(_messages)
	var input_row := HBoxContainer.new()
	left.add_child(input_row)
	_input = LineEdit.new()
	_input.placeholder_text = "Ask about this world..."
	_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_input.text_submitted.connect(func(_text: String) -> void: _send_message())
	input_row.add_child(_input)
	_send = Button.new()
	_send.text = "Send"
	_send.pressed.connect(_send_message)
	input_row.add_child(_send)
	var note := Label.new()
	note.text = "Works offline with explanations from current simulation data. A local language model is optional. Chat does not change simulation parameters."
	note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	left.add_child(note)
	var right_scroll := ScrollContainer.new()
	right_scroll.custom_minimum_size.x = 340
	right_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	columns.add_child(right_scroll)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right_scroll.add_child(right)
	var reports := Label.new()
	reports.text = "SAVE · REOPEN · EXPORT"
	reports.add_theme_font_size_override("font_size", 18)
	right.add_child(reports)
	_title = LineEdit.new()
	_title.text = "Chemical universe"
	right.add_child(_title)
	_button(right, "Save current world", _save_world)
	_saves = OptionButton.new()
	_saves.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	right.add_child(_saves)
	_button(right, "Reopen selected world", _load_world)
	_button(right, "Refresh saved worlds", _refresh_saves)
	_button(right, "Create checkpoint", _create_checkpoint)
	_checkpoints = VBoxContainer.new()
	right.add_child(_checkpoints)
	right.add_child(HSeparator.new())
	for format in ["json", "csv", "sdf", "md", "pdf"]:
		_button(right, "Export current world · " + format.to_upper(), func() -> void: _choose_export(format))
	_button(right, "Open reports folder", func() -> void: OS.shell_open(ProjectSettings.globalize_path("user://reports")))
	_status = Label.new()
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status.text = "Choose an export format and destination."
	right.add_child(_status)
	_file_dialog = FileDialog.new()
	_file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_file_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_file_dialog.size = Vector2i(800, 540)
	_file_dialog.file_selected.connect(_export_to)
	add_child(_file_dialog)
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path("user://reports"))
	_append("Assistant", "Welcome. Ask what is happening to inspect the actual population, generation, and families. If no language model is running, the engine uses local rule-based explanations.")
	_refresh_saves()
	_refresh_checkpoints()

func _button(parent: Node, text: String, action: Callable) -> void:
	var button := Button.new()
	button.text = text
	button.pressed.connect(action)
	parent.add_child(button)

func _append(speaker: String, text: String) -> void:
	var card := PanelContainer.new()
	card.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 12, 10))
	_messages.add_child(card)
	var body := Label.new()
	body.text = speaker + "\n" + text
	body.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	body.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	card.add_child(body)
	call_deferred("_scroll_bottom")

func _scroll_bottom() -> void:
	_chat_scroll.scroll_vertical = int(_chat_scroll.get_v_scroll_bar().max_value)

func _send_message() -> void:
	var question: String = _input.text.strip_edges()
	if question.is_empty() or _busy:
		return
	_busy = true
	_send.disabled = true
	_send.text = "Thinking..."
	_input.clear()
	_append("You", question)
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/assistant/ask", {"question": question}, func(code: int, response: Variant) -> void:
		_busy = false
		_send.disabled = false
		_send.text = "Send"
		if code == 200 and response is Dictionary:
			var data: Dictionary = response.get("data", {})
			_append(str(data.get("provider", "Local assistant")), str(data.get("answer", "No answer returned.")))
			if data.get("proposed_action") is Dictionary:
				_review_action(data["proposed_action"])
		else:
			_append("Connection problem", "The local engine could not answer (HTTP %d). Start EVOCURE using start_evocure.bat, then try again." % code)
	)

func _save_world() -> void:
	_status.text = "Saving..."
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/experiments/save", {"title": _title.text}, func(code: int, response: Variant) -> void:
		if code == 200 and response is Dictionary and bool(response.get("success", false)):
			_saved_id = str(response.get("data", {}).get("experiment_id", ""))
			_status.text = "World saved: " + _saved_id
			_refresh_saves()
		else:
			_status.text = "Save failed (HTTP %d)." % code
	)

func _refresh_saves() -> void:
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/experiments", null, func(code: int, data: Variant) -> void:
		if code != 200 or not data is Array:
			return
		_saves.clear()
		for record in data:
			_saves.add_item(str(record.get("title", record.get("experiment_id", "World"))))
			_saves.set_item_metadata(_saves.item_count - 1, str(record.get("experiment_id", "")))
	)

func _load_world() -> void:
	if _saves.selected < 0:
		return
	var id: String = str(_saves.get_item_metadata(_saves.selected))
	ApiClient.set_playback("pause")
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/experiments/load/" + id.uri_encode(), {}, func(code: int, response: Variant) -> void:
		_status.text = "World reopened." if code == 200 else "Could not reopen this save: " + str(response)
		ApiClient.fetch_state()
	)

func _create_checkpoint() -> void:
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/checkpoints/create", {"label": _title.text}, func(code: int, _data: Variant) -> void:
		_status.text = "Checkpoint created." if code == 200 else "Checkpoint failed."
		_refresh_checkpoints()
	)

func _refresh_checkpoints() -> void:
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/checkpoints", null, func(code: int, data: Variant) -> void:
		if code != 200 or not data is Array:
			return
		for child in _checkpoints.get_children():
			child.queue_free()
		for record in data.slice(maxi(0, data.size() - 8)):
			var id: String = str(record.get("checkpoint_id", ""))
			_button(_checkpoints, "Restore · Gen %d · %s" % [int(record.get("generation", 0)), str(record.get("label", "Snapshot"))], func() -> void:
				ApiClient.set_playback("pause")
				ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/checkpoints/restore/" + id.uri_encode(), {}, func(c: int, _r: Variant) -> void:
					_status.text = "Checkpoint restored." if c == 200 else "Restore failed."
					ApiClient.fetch_state()
				)
			)
	)

func _choose_export(format: String) -> void:
	_format = format
	_file_dialog.filters = PackedStringArray(["*." + format + " ; " + format.to_upper()])
	_file_dialog.current_dir = ProjectSettings.globalize_path("user://reports")
	_file_dialog.current_file = "EVOCURE_Gen_%d.%s" % [int(AppState.clocks.get("evolutionary_generation", 0)), format]
	_file_dialog.popup_centered_ratio(0.7)

func _export_to(path: String) -> void:
	_status.text = "Saving current state and exporting..."
	var format: String = _format
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/experiments/save", {"title": _title.text}, func(code: int, response: Variant) -> void:
		if code != 200 or not response is Dictionary:
			_status.text = "Could not save the world for export."
			return
		var id: String = str(response.get("data", {}).get("experiment_id", ""))
		ApiClient.download_export_to_file(id, format, path, func(ok: bool, saved_path: String) -> void:
			_status.text = "Export saved: " + saved_path if ok else "Export failed. Please try again."
		)
	)

func _review_action(action: Dictionary) -> void:
	var button := Button.new()
	button.text = "Review proposed simulation change"
	_messages.add_child(button)
	button.pressed.connect(func() -> void:
		var dialog := ConfirmationDialog.new()
		dialog.title = "Approve this change?"
		dialog.dialog_text = JSON.stringify(action, "  ")
		add_child(dialog)
		dialog.confirmed.connect(func() -> void:
			ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/assistant/approve-action", {"action": action}, func(code: int, result: Variant) -> void:
				_append("Simulation", "Approved change applied." if code == 200 else "Change failed: " + str(result))
				ApiClient.fetch_state()
			)
			dialog.queue_free()
		)
		dialog.canceled.connect(func() -> void: dialog.queue_free())
		dialog.popup_centered(Vector2i(640, 360))
	)
