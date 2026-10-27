class_name TabEvolutionLab
extends Control

var _summary: Label
var _status: Label
var _progress: ProgressBar
var _list: VBoxContainer
var _source: OptionButton
var _filter: LineEdit
var _page: Label
var _prompt: TextEdit
var _plan_label: Label
var _approve: Button
var _plan: Dictionary = {}
var _selected: Dictionary = {}
var _candidate_label: Label
var _candidate_image: TextureRect
var _matches: OptionButton
var _reference_image: TextureRect
var _comparison: Label
var _match_data: Array = []
var _offset: int = 0
var _total: int = 0
var _timer: float = 0
var _refreshing: bool = false
var _last_job: String = "idle"
var _experiment: String = ""

func _ready() -> void:
	ThemeBuilder.fill_rect(self)
	var root := VBoxContainer.new()
	ThemeBuilder.fill_rect(root)
	root.add_theme_constant_override("separation", 8)
	add_child(root)
	_summary = _label(root, "Connecting to the real compound registry...", 16)
	var columns := HBoxContainer.new()
	columns.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(columns)
	var left := VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(left)
	var toolbar := HBoxContainer.new()
	left.add_child(toolbar)
	_source = OptionButton.new()
	for name in ["Universe discoveries", "Directed shortlist", "All directed structures", "All registered compounds"]:
		_source.add_item(name)
	_source.item_selected.connect(func(_i: int) -> void: _offset = 0; _refresh())
	toolbar.add_child(_source)
	_filter = LineEdit.new()
	_filter.placeholder_text = "Name, ID, formula or SMILES"
	_filter.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_filter.text_submitted.connect(func(_t: String) -> void: _offset = 0; _refresh())
	toolbar.add_child(_filter)
	_button(toolbar, "Filter / refresh", func() -> void: _offset = 0; _refresh())
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	left.add_child(scroll)
	_list = VBoxContainer.new()
	_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_list)
	var pages := HBoxContainer.new()
	left.add_child(pages)
	_button(pages, "Previous", func() -> void: _offset = maxi(0, _offset - 60); _refresh())
	_page = _label(pages, "", 12)
	_page.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_button(pages, "Next", func() -> void: if _offset + 60 < _total: _offset += 60; _refresh())
	_label(left, "GUIDE EVOLUTION WITH A RESEARCH OBJECTIVE", 14)
	_prompt = TextEdit.new()
	_prompt.custom_minimum_size.y = 65
	_prompt.text = "Find 5 experimental candidates for pancreatic cancer, prioritizing low predicted toxicity and drug-likeness."
	_prompt.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	left.add_child(_prompt)
	var controls := HBoxContainer.new()
	left.add_child(controls)
	_button(controls, "1. Ask AI to plan", _create_plan)
	_approve = _button(controls, "2. Run this guided search", _run_plan)
	_approve.disabled = true
	_button(controls, "Run balanced search", _run_balanced)
	_button(controls, "Stop search", func() -> void: ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/discovery/cancel", {}))
	_plan_label = _label(left, "Review the plan before running. Candidates are research hypotheses, not cures.", 12)
	_progress = ProgressBar.new()
	_progress.max_value = 100
	left.add_child(_progress)
	_status = _label(left, "No directed search has been run. Universe discoveries appear above automatically.", 12)
	var right_scroll := ScrollContainer.new()
	right_scroll.custom_minimum_size.x = 440
	right_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	columns.add_child(right_scroll)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right_scroll.add_child(right)
	_label(right, "SELECTED COMPOUND", 16)
	_candidate_image = _image(right)
	_candidate_label = _label(right, "Select a compound from the live list.", 12)
	_button(right, "Deploy 10 molecules into universe", _deploy)
	_button(right, "Expose tissue to selected compound", _expose)
	right.add_child(HSeparator.new())
	_label(right, "COMPARE WITH EXISTING DRUGS", 16)
	_matches = OptionButton.new()
	_matches.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_matches.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	_matches.item_selected.connect(func(i: int) -> void: _show_match(i))
	right.add_child(_matches)
	_reference_image = _image(right)
	_comparison = _label(right, "Selecting any compound automatically searches the local drug library. Exact identity and structural similarity are reported separately.", 12)
	_button(right, "View nearest drug in tissue", _expose_reference)
	_label(right, "Local structural comparisons do not prove novelty, safety or therapeutic activity. All tissue predictions remain exploratory.", 11)
	visibility_changed.connect(func() -> void: if visible: _refresh())
	_refresh()

func _label(parent: Node, text: String, font_size: int) -> Label:
	var label := Label.new()
	label.text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.add_theme_font_size_override("font_size", font_size)
	parent.add_child(label)
	return label

func _button(parent: Node, text: String, callback: Callable) -> Button:
	var button := Button.new()
	button.text = text
	button.pressed.connect(callback)
	parent.add_child(button)
	return button

func _image(parent: Node) -> TextureRect:
	var image := TextureRect.new()
	image.custom_minimum_size = Vector2(410, 145)
	image.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	image.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	parent.add_child(image)
	return image

func _process(delta: float) -> void:
	_timer += delta
	if visible and _timer > 3.0:
		_timer = 0
		_refresh()

func _refresh() -> void:
	if _refreshing:
		return
	_refreshing = true
	var source: String = ["universe", "shortlist", "directed", "all"][_source.selected]
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/discovery/compounds?source=%s&offset=%d&search=%s" % [source, _offset, _filter.text.uri_encode()], null, func(code: int, response: Variant) -> void:
		_refreshing = false
		if code != 200 or not response is Dictionary:
			_status.text = "Discovery registry unavailable (HTTP %d). Start the updated local engine." % code
			return
		if _experiment != str(response.get("experiment_id", "")):
			_experiment = str(response.get("experiment_id", ""))
			_selected = {}
			_plan = {}
			_approve.disabled = true
		_summary.text = ("Gemma online · " if bool(response.get("local_ai_running", false)) else "Rule-based planner · ") + "LIVE · %d compounds registered · %d generated · %d particles · generation %d · %d reactions" % [int(response.get("registered_compounds", 0)), int(response.get("generated_compounds", 0)), int(response.get("population", 0)), int(response.get("generation", 0)), int(response.get("reactions", 0))]
		_total = int(response.get("total", 0))
		_page.text = "%d-%d of %d" % [mini(_offset + 1, _total), mini(_offset + 60, _total), _total]
		for child in _list.get_children():
			child.queue_free()
		var rows: Array = response.get("compounds", [])
		if rows.is_empty():
			_label(_list, "No compounds in this view yet. Run the universe or start a directed search. No sample discoveries are shown.", 13)
		for item in rows:
			var button := Button.new()
			var known: String = str(item.get("exact_reference_name", "")) if item.get("exact_reference_name") != null else ""
			button.text = "%s · %s\nGen %d | %d particles | %s%s" % [str(item.get("compound_id", "")), str(item.get("name", "")), int(item.get("origin_generation", 0)), int(item.get("population", 0)), str(item.get("transformation_type", "")), " | Known: " + known if not known.is_empty() else ""]
			button.alignment = HORIZONTAL_ALIGNMENT_LEFT
			button.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
			button.tooltip_text = str(item.get("canonical_smiles", ""))
			button.pressed.connect(func() -> void: _select(item))
			_list.add_child(button)
		if _selected.is_empty() and not rows.is_empty():
			_select(rows[0])
		var job: Dictionary = response.get("job", {})
		var job_status: String = str(job.get("status", "idle"))
		if job_status == "running":
			var target: int = int(job.get("target_generations", 1))
			_progress.value = 100.0 * (maxi(0, int(job.get("generation", 1)) - 1) * 15 + int(job.get("stage", 0))) / maxf(1, target * 15)
			_status.text = "Generation %d/%d · stage %d: %s" % [int(job.get("generation", 0)), target, int(job.get("stage", 0)), str(job.get("stage_name", "Preparing"))]
		elif job_status != "idle":
			_status.text = str(job.get("message", job_status))
			_progress.value = 100 if job_status == "complete" else _progress.value
		if job_status == "complete" and _last_job == "running":
			_source.select(1)
			_offset = 0
			var shortlist: Array = response.get("shortlist", [])
			if not shortlist.is_empty():
				_select(shortlist[0])
		_last_job = job_status
	)

func _select(compound: Dictionary) -> void:
	_selected = compound
	var scoring: String = "Biological activity and toxicity have not been evaluated for this universe compound."
	if compound.get("risk_index") != null:
		scoring = "Exploratory model · risk index %s | pIC50 %s | in domain: %s" % [str(compound.get("risk_index")), str(compound.get("pic50")), str(compound.get("in_applicability_domain", false))]
	_candidate_label.text = "%s\n%s\nGeneration %d | %s\n%s\n%s" % [str(compound.get("name", "Compound")), str(compound.get("formula", "")), int(compound.get("origin_generation", 0)), str(compound.get("transformation_type", "directed_optimization")), str(compound.get("canonical_smiles", "")), scoring]
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/discovery/drug-comparison?query=" + str(compound.get("compound_id", "")).uri_encode(), null, func(code: int, data: Variant) -> void:
		if code != 200 or not data is Dictionary:
			_comparison.text = "Comparison unavailable: " + str(data)
			return
		if str(data.get("candidate", {}).get("compound_id", "")) != str(_selected.get("compound_id", "")):
			return
		_candidate_image.texture = ApiClient.svg_to_texture(str(data.get("candidate", {}).get("svg_2d", "")))
		_match_data = data.get("matches", [])
		_matches.clear()
		for match_item in _match_data:
			_matches.add_item("%s · %.1f%% similarity%s" % [str(match_item["name"]), float(match_item["similarity"]) * 100, " · EXACT" if bool(match_item["exact_match"]) else ""])
		_show_match(0)
	)

func _show_match(index: int) -> void:
	if index < 0 or index >= _match_data.size():
		return
	var match_item: Dictionary = _match_data[index]
	_reference_image.texture = ApiClient.svg_to_texture(str(match_item.get("svg_2d", "")))
	var delta: Dictionary = match_item.get("candidate_minus_reference", {})
	_comparison.text = "%s\n%s\nTanimoto %.4f (Morgan radius 2)\nCandidate minus reference: MW %.2f Da | LogP %.2f | TPSA %.2f | QED %.3f\nReference: %s\nSimilarity does not establish the same biological activity." % [str(match_item.get("name", "")), str(match_item.get("identity", "")), float(match_item.get("similarity", 0)), float(delta.get("mol_weight", 0)), float(delta.get("logp", 0)), float(delta.get("tpsa", 0)), float(delta.get("qed", 0)), str(match_item.get("provenance", "Local library"))]

func _create_plan() -> void:
	_approve.disabled = true
	_plan_label.text = "Asking the local planner..."
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/discovery/plan", {"prompt": _prompt.text}, func(code: int, data: Variant) -> void:
		if code != 200 or not data is Dictionary:
			_plan_label.text = "Planning failed: " + str(data)
			return
		_plan = data
		_plan_label.text = "Provider: %s | %d experimental candidates | %s\nPriorities: %s | seeds: %s | %d generations\n%s" % [str(data.get("provider", "")), int(data.get("candidate_count", 5)), str(data.get("disease_id", "PDAC")), str(data.get("priorities", [])), str(data.get("seed_queries", [])), int(data.get("generations", 2)), str(data.get("limitation", ""))]
		_approve.disabled = false
	)

func _run_plan() -> void:
	if not _plan.is_empty():
		_start({"plan_id": _plan["plan_id"]})

func _run_balanced() -> void:
	_start({"generations": 2, "population_size": 12, "candidate_count": 5})

func _start(settings: Dictionary) -> void:
	_status.text = "Starting background search..."
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/discovery/start", settings, func(code: int, data: Variant) -> void:
		_status.text = "Search running. Progress updates every three seconds." if code == 200 else "Search failed: " + str(data)
		if code == 200:
			_last_job = "running"
		_refresh()
	)

func _deploy() -> void:
	if not _selected.is_empty():
		ApiClient.spawn_compound(str(_selected["compound_id"]), 10, null, null, "neutral", "MOLECULE")

func _expose() -> void:
	if not _selected.is_empty():
		AppState.tissue_compound_query = str(_selected["compound_id"])
		AppState.set_workspace_tab(2)

func _expose_reference() -> void:
	if _matches.selected >= 0 and _matches.selected < _match_data.size():
		AppState.tissue_compound_query = str(_match_data[_matches.selected]["name"])
		AppState.set_workspace_tab(2)
