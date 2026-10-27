class_name TabFamilyAndEditor
extends Control
var _graph: GenealogyGraphControl
var _status: Label
var _parent: LineEdit
var _parent_smiles: String = "CC(=O)Oc1ccccc1C(=O)O"
var _edited: String = ""
var _original_image: TextureRect
var _edited_image: TextureRect
var _details: Label
var _operation: OptionButton
var _parameter: OptionButton
var _catalog: Dictionary = {}
var _save: FileDialog
var _dag: Dictionary = {}
var _family: OptionButton

func _ready() -> void:
	ThemeBuilder.fill_rect(self)
	var columns := HBoxContainer.new()
	ThemeBuilder.fill_rect(columns)
	add_child(columns)
	var left := VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(left)
	var toolbar := HBoxContainer.new()
	left.add_child(toolbar)
	_button(toolbar, "Refresh real genealogy", _fetch_genealogy_dag)
	_button(toolbar, "Selected molecule's lineage", _lineage)
	_button(toolbar, "Export genealogy JSON", func() -> void: _save.popup_centered_ratio(0.7))
	_family = OptionButton.new()
	_family.add_item("All chemical families")
	_family.item_selected.connect(func(_i: int) -> void: _draw_dag())
	left.add_child(_family)
	_status = _label(left, "Loading genealogy...", 12)
	_graph = GenealogyGraphControl.new()
	_graph.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_graph.node_clicked.connect(func(data: Dictionary) -> void:
		_parent.text = str(data.get("compound_id", data.get("id", "")))
		_load_parent()
	)
	left.add_child(_graph)
	var right_scroll := ScrollContainer.new()
	right_scroll.custom_minimum_size.x = 470
	right_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	columns.add_child(right_scroll)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right_scroll.add_child(right)
	_label(right, "MOLECULAR STRUCTURE EDITOR", 17)
	_parent = LineEdit.new()
	_parent.text = "Aspirin"
	right.add_child(_parent)
	_button(right, "Load compound / SMILES", _load_parent)
	_original_image = _image(right)
	_operation = OptionButton.new()
	for name in ["Add substituent", "Replace atom", "Append ring", "Bioisostere swap", "Validate / canonicalize"]:
		_operation.add_item(name)
	_operation.item_selected.connect(func(_i: int) -> void: _parameters())
	right.add_child(_operation)
	_parameter = OptionButton.new()
	right.add_child(_parameter)
	_button(right, "Apply valid structural edit", _on_apply_mutation_clicked)
	_edited_image = _image(right)
	_details = _label(right, "No edit has been applied. Invalid or inapplicable operations return an explanation.", 12)
	_button(right, "Deploy edited structure", func() -> void:
		if not _edited.is_empty():
			ApiClient.spawn_compound(_edited, 10, null, null, "neutral", "MOLECULE")
	)
	_button(right, "Assess edited structure", func() -> void:
		AppState.analysis_compound_query = _edited if not _edited.is_empty() else _parent_smiles
		AppState.set_workspace_tab(4)
	)
	_button(right, "View edited structure in tissue", func() -> void:
		AppState.tissue_compound_query = _edited if not _edited.is_empty() else _parent_smiles
		AppState.set_workspace_tab(2)
	)
	_save = FileDialog.new()
	_save.access = FileDialog.ACCESS_FILESYSTEM
	_save.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_save.filters = PackedStringArray(["*.json ; Genealogy JSON"])
	_save.current_file = "EVOCURE_genealogy.json"
	_save.file_selected.connect(func(path: String) -> void:
		var file := FileAccess.open(path, FileAccess.WRITE)
		if file != null:
			file.store_string(JSON.stringify(_dag, "  "))
			_status.text = "Genealogy saved to " + path
		else:
			_status.text = "Could not write genealogy file."
	)
	add_child(_save)
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/editor/operations", null, func(code: int, data: Variant) -> void:
		if code == 200 and data is Dictionary:
			_catalog = data
			_parameters()
	)
	visibility_changed.connect(func() -> void:
		if visible:
			_fetch_genealogy_dag()
			if not AppState.analysis_compound_query.is_empty():
				_parent.text = AppState.analysis_compound_query
				_load_parent()
	)
	_fetch_genealogy_dag()
	_load_parent()

func _label(parent: Node, text: String, font_size: int) -> Label:
	var label := Label.new()
	label.text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.add_theme_font_size_override("font_size", font_size)
	parent.add_child(label)
	return label

func _button(parent: Node, text: String, action: Callable) -> void:
	var button := Button.new()
	button.text = text
	button.pressed.connect(action)
	parent.add_child(button)

func _image(parent: Node) -> TextureRect:
	var image := TextureRect.new()
	image.custom_minimum_size = Vector2(440, 180)
	image.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	image.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	parent.add_child(image)
	return image

func _fetch_genealogy_dag() -> void:
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/genealogy", null, func(code: int, data: Variant) -> void:
		if code != 200 or not data is Dictionary:
			_status.text = "Genealogy unavailable (HTTP %d)." % code
			return
		_dag = data
		_family.clear()
		_family.add_item("All chemical families")
		for name in data.get("family_counts", {}):
			_family.add_item(str(name))
		_status.text = "%d recorded compounds · %d lineage edges · wheel to zoom; drag to pan" % [data.get("nodes", []).size(), data.get("edges", []).size()]
		_draw_dag()
	)

func _draw_dag() -> void:
	var nodes: Array = _dag.get("nodes", [])
	if _family.selected > 0:
		var family: String = _family.get_item_text(_family.selected)
		nodes = nodes.filter(func(n: Dictionary) -> bool: return str(n.get("family_id", "")) == family)
	_graph.set_genealogy_data(nodes, _dag.get("edges", []))

func _lineage() -> void:
	var query: String = str(AppState.selected_entity.get("compound_id", ""))
	if query.is_empty():
		_status.text = "Select a particle in the universe first."
		return
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/entity-lineage/" + query.uri_encode(), null, func(code: int, data: Variant) -> void:
		if code == 200 and data is Dictionary:
			var nodes: Array = data.get("data", data).get("ancestors", [])
			var edges: Array = []
			for node in nodes:
				for parent in node.get("parent_ids", []):
					edges.append({"source": parent, "target": node.get("compound_id", "")})
			_graph.set_genealogy_data(nodes, edges)
			_status.text = "Recorded lineage: %d compounds" % nodes.size()
	)

func _load_parent() -> void:
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/discovery/drug-comparison?query=" + _parent.text.uri_encode(), null, func(code: int, data: Variant) -> void:
		if code == 200 and data is Dictionary:
			var record: Dictionary = data.get("candidate", {})
			_parent_smiles = str(record.get("canonical_smiles", ""))
			_original_image.texture = ApiClient.svg_to_texture(str(record.get("svg_2d", "")))
		else:
			_details.text = "Could not load parent: " + str(data)
	)

func _parameters() -> void:
	_parameter.clear()
	var key: String = ["substituents", "atom_replacements", "ring_fusions_and_appends", "bioisostere_swaps", ""][_operation.selected]
	for item in _catalog.get(key, []):
		_parameter.add_item(str(item.get("description", item.get("id", ""))))
		_parameter.set_item_metadata(_parameter.item_count - 1, str(item.get("id", "")))
	_parameter.disabled = key.is_empty()

func _on_apply_mutation_clicked() -> void:
	var operation: String = ["add_group", "replace_atom", "fuse_or_append_ring", "bioisostere_swap", "clean_and_canonicalize"][_operation.selected]
	var param: String = str(_parameter.get_item_metadata(_parameter.selected)) if _parameter.selected >= 0 else ""
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/editor/apply", {"smiles": _parent_smiles, "operation": operation, "param": param, "substituent": param}, func(code: int, data: Variant) -> void:
		var result: Dictionary = data.get("data", {}) if data is Dictionary else {}
		if code != 200 or not bool(result.get("valid", false)):
			_details.text = "Edit not applied: " + str(result.get("error", data))
			return
		_edited = str(result.get("edited_smiles", ""))
		var diff: Dictionary = result.get("diff", {})
		_edited_image.texture = ApiClient.svg_to_texture(str(diff.get("svg_b_highlighted", result.get("svg_2d", ""))))
		_details.text = "Edited: %s\nDelta MW %.3f Da | LogP %.3f | TPSA %.2f | QED %.4f\nRDKit structure validation passed; biological activity remains unknown." % [_edited, float(diff.get("delta_mw", 0)), float(diff.get("delta_logp", 0)), float(diff.get("delta_tpsa", 0)), float(diff.get("delta_qed", 0))]
	)
