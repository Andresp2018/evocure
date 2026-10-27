class_name TabCancerBattle
extends Control

var _view: TissueView
var _compound: LineEdit
var _dose: SpinBox
var _route: OptionButton
var _overlay: OptionButton
var _metrics: Label
var _events: Label
var _selected: Label
var _status: Label
var _disease: String = ""
var _busy: bool = false
var _advanced: VBoxContainer
var _second_compound: LineEdit

func _ready() -> void:
	ThemeBuilder.fill_rect(self)
	var root := HBoxContainer.new()
	ThemeBuilder.fill_rect(root)
	root.add_theme_constant_override("separation", 12)
	add_child(root)
	var left := VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_child(left)
	var controls := HFlowContainer.new()
	left.add_child(controls)
	_button(controls, "Play tissue", func() -> void: ApiClient.set_playback("play", 1, "tumor"))
	_button(controls, "Play chemistry + tissue", func() -> void: ApiClient.set_playback("play", 1, "both"))
	_button(controls, "Pause", func() -> void: ApiClient.set_playback("pause", 1, "tumor"))
	_button(controls, "Step 15 biological minutes", func() -> void: ApiClient.set_playback("step", 1, "tumor"))
	_overlay = OptionButton.new()
	for name in ["Oxygen / hypoxia", "Free drug concentration", "ECM / biofilm / fibrosis", "Nutrients", "Cells only"]:
		_overlay.add_item(name)
	_overlay.item_selected.connect(func(i: int) -> void: _view.overlay = ["oxygen", "drug_free", "ecm_density", "nutrient", "none"][i])
	controls.add_child(_overlay)
	_button(controls, "Reset zoom", func() -> void: _view.zoom = 1)
	_view = TissueView.new()
	_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	left.add_child(_view)
	_view.cell_selected.connect(_inspect_cell)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size.x = 370
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	root.add_child(scroll)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(right)
	_label(right, "LIVE TISSUE EXPERIMENT", 17)
	_metrics = _label(right, "Waiting for tissue state...", 14)
	_label(right, "Compound name, ID or SMILES", 12)
	_compound = LineEdit.new()
	_compound.text = AppState.tissue_compound_query
	right.add_child(_compound)
	_label(right, "Applied concentration (uM; model input)", 12)
	_dose = SpinBox.new()
	_dose.min_value = 0
	_dose.max_value = 1000
	_dose.step = 0.5
	_dose.value = 5
	right.add_child(_dose)
	_route = OptionButton.new()
	for name in ["Vascular delivery", "Uniform tissue exposure"]:
		_route.add_item(name)
	right.add_child(_route)
	_button(right, "Apply exposure & watch tissue", _apply)
	_button(right, "Compare with untreated control (24h)", _compare)
	var reset := ConfirmationDialog.new()
	reset.dialog_text = "Restore this disease's starting tissue and remove its exposure? The chemical universe is retained."
	add_child(reset)
	_button(right, "Reset this tissue", func() -> void: reset.popup_centered(Vector2i(570, 150)))
	reset.confirmed.connect(func() -> void:
		ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/tumor/reset", {}, func(code: int, data: Variant) -> void:
			_status.text = "Tissue reset." if code == 200 else str(data)
			ApiClient.fetch_state()
		)
	)
	_status = _label(right, "Apply an exposure to simulate diffusion, uptake and cell response. Cells change only when the model state changes.", 12)
	_advanced = VBoxContainer.new()
	right.add_child(_advanced)
	_label(_advanced, "PDAC RESEARCH EXPERIMENTS", 14)
	_second_compound = LineEdit.new()
	_second_compound.text = "Paclitaxel"
	_second_compound.placeholder_text = "Second drug (same concentration)"
	_advanced.add_child(_second_compound)
	_button(_advanced, "Compare control / A / B / A+B (6h)", _combination)
	_button(_advanced, "Run 2 chemical-tumour coevolution cycles", _coevolve)
	_label(_advanced, "Independent drug hazards are assumed; synergy is not established. Coevolution advances this tissue and mutates real chemical structures.", 11)
	right.add_child(HSeparator.new())
	_selected = _label(right, "Click a cell to inspect type, clone, oxygen, uptake and state.", 12)
	right.add_child(HSeparator.new())
	_label(right, "RECENT MODEL EVENTS", 14)
	_events = _label(right, "No events yet. Advance biological time.", 12)
	_label(right, "Exploratory 2D models. Shapes and membrane motion are illustrative. They do not reproduce patient tissue or prove efficacy. Unknown disease-compound responses are rejected rather than assigned a default effect.", 11)
	AppState.state_updated.connect(_update)
	visibility_changed.connect(func() -> void:
		if visible:
			_compound.text = AppState.tissue_compound_query
			ApiClient.fetch_state()
	)
	_update(AppState.latest_state)

func _button(parent: Node, text: String, action: Callable) -> void:
	var button := Button.new()
	button.text = text
	button.pressed.connect(action)
	parent.add_child(button)

func _label(parent: Node, text: String, font_size: int) -> Label:
	var label := Label.new()
	label.text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.add_theme_font_size_override("font_size", font_size)
	parent.add_child(label)
	return label

func _update(state: Dictionary) -> void:
	var tumor: Dictionary = state.get("tumor", {})
	if tumor.is_empty():
		return
	var disease: String = str(tumor.get("disease_id", "PDAC"))
	if disease != _disease:
		_disease = disease
		_advanced.visible = disease == "PDAC"
		AppState.tissue_compound_query = {"PDAC": "Gemcitabine", "NSCLC_KRAS": "Sotorasib", "AMR_BACTERIAL": "Meropenem", "METABOLIC_MASLD": "Resmetirom"}.get(disease, "Gemcitabine")
		_compound.text = AppState.tissue_compound_query
	_view.update_snapshot(tumor)
	var summary: Dictionary = tumor.get("summary_metrics", {})
	var counts: Dictionary = tumor.get("cell_counts_by_type", {})
	var deaths: Dictionary = tumor.get("cumulative_deaths", {})
	var path_name: String = "Tumour cells"
	if disease == "AMR_BACTERIAL":
		path_name = "Bacteria"
	elif disease == "METABOLIC_MASLD":
		path_name = "Pathological liver cells"
	var compound: Variant = tumor.get("active_compound")
	var compound_name: String = str(compound.get("name", "None")) if compound is Dictionary else "None"
	_metrics.text = "%s\nBiological time: %.2f hours\n%s alive: %d\nHealthy epithelial cells: %d\nResistant fraction: %.1f%%\nApoptosis: %d | Necrosis: %d\nMean intracellular drug: %.4f uM\nExposure: %s" % [str(tumor.get("display_name", disease)), float(tumor.get("biological_hours", 0)), path_name, int(summary.get("viable_tumor_cells", 0)), int(counts.get("EPITHELIAL_HEALTHY", 0)), float(summary.get("resistant_fraction", 0)) * 100, int(deaths.get("apoptosis", 0)), int(deaths.get("necrosis", 0)), float(summary.get("mean_intracellular_drug_um", 0)), compound_name]
	if disease == "AMR_BACTERIAL":
		_metrics.text = _metrics.text.replace("Apoptosis:", "Drug-associated deaths:").replace("Necrosis:", "Stress deaths:")
	elif disease == "METABOLIC_MASLD":
		_metrics.text = _metrics.text.replace("Healthy epithelial cells:", "Healthier hepatocytes:").replace("Resistant fraction:", "Activated stellate fraction:")
	var lines: Array[String] = []
	for event in state.get("tissue_events", []):
		lines.append("%.2fh: %d divisions, %d apoptosis, %d necrosis%s" % [float(event.get("biological_hours", 0)), int(event.get("division", 0)), int(event.get("apoptosis", 0)), int(event.get("necrosis", 0)), ", steatosis cleared: %d" % int(event.get("steatosis_cleared", 0)) if event.has("steatosis_cleared") else ""])
	_events.text = "\n".join(lines) if not lines.is_empty() else "No recorded tissue events. Advance biological time."
	if disease == "AMR_BACTERIAL":
		_events.text = _events.text.replace("apoptosis", "drug-associated deaths").replace("necrosis", "stress deaths")
	for cell in tumor.get("cells", []):
		if str(cell.get("id", "")) == _view.selected_id:
			_inspect_cell(cell)
			break

func _inspect_cell(cell: Dictionary) -> void:
	var type: String = str(cell.get("t", ""))
	if _disease == "AMR_BACTERIAL":
		type = {"TUMOR_SENSITIVE": "Bacterium", "TUMOR_RESISTANT": "Resistant bacterium", "EPITHELIAL_HEALTHY": "Host mucosal cell"}.get(type, type)
	elif _disease == "METABOLIC_MASLD":
		type = {"TUMOR_SENSITIVE": "Steatotic hepatocyte", "TUMOR_RESISTANT": "Activated stellate cell", "EPITHELIAL_HEALTHY": "Healthier hepatocyte", "CAF": "Fibrogenic stellate cell"}.get(type, type)
	_selected.text = "%s\nType: %s\nClone: %s\nState: %s | generation %d\nOxygen %.3f | local drug %.4f uM\nIntracellular drug %.4f uM\nDivision progress %.1f%% | resistance %.3f\nMetabolic state: %s" % [str(cell.get("id", "")), type, str(cell.get("cl", "")), str(cell.get("st", "")), int(cell.get("generation", 0)), float(cell.get("o2", 0)), float(cell.get("local_drug", 0)), float(cell.get("drg", 0)), float(cell.get("cycle", 0)) * 100, float(cell.get("res", 0)), str(cell.get("metabolic", "unknown"))]

func _apply() -> void:
	_status.text = "Applying model exposure..."
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/tumor/introduce-compound", {"compound_query": _compound.text, "applied_concentration_um": _dose.value, "injection_region": ["vascular", "uniform"][_route.selected]}, func(code: int, data: Variant) -> void:
		if code == 200 and data is Dictionary:
			_status.text = str(data.get("data", {}).get("model_info", {}).get("note", "Exploratory model exposure applied."))
			_overlay.select(1)
			_view.overlay = "drug_free"
			ApiClient.set_playback("play", 1, "tumor")
		else:
			_status.text = "Exposure was not applied: " + str(data)
	)

func _compare() -> void:
	if _busy:
		return
	_busy = true
	_status.text = "Running matched-seed control and exposure for 24 model hours..."
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/tumor/controlled-comparison", {"compound_query": _compound.text, "applied_concentration_um": _dose.value, "duration_hours": 24}, func(code: int, data: Variant) -> void:
		_busy = false
		if code == 200 and data is Dictionary:
			_status.text = "Matched-seed model comparison (not clinical evidence):\n" + JSON.stringify(data.get("data", {}).get("deltas", {}), "  ")
		else:
			_status.text = "Comparison failed: " + str(data)
	)

func _combination() -> void:
	if _busy:
		return
	_busy = true
	ApiClient.set_playback("pause", 1, "tumor")
	_status.text = "Running four matched-seed model arms..."
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/tumor/combination", {"compound_a": _compound.text, "compound_b": _second_compound.text, "dose_a_um": _dose.value, "dose_b_um": _dose.value, "hours": 6, "run_4_arm_trial": true}, func(code: int, data: Variant) -> void:
		_busy = false
		_status.text = "Independent-hazard model comparison:\n" + JSON.stringify(data.get("data", {}).get("trial_4_arm", {}).get("deltas", {}), "  ") if code == 200 and data is Dictionary else "Combination failed: " + str(data)
		ApiClient.fetch_state()
	)

func _coevolve() -> void:
	if _busy:
		return
	_busy = true
	ApiClient.set_playback("pause", 1, "tumor")
	_status.text = "Running two 6-hour PDAC coevolution cycles..."
	ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/tumor/coevolution", {"n_cycles": 2, "hours_per_cycle": 6, "applied_concentration_um": _dose.value}, func(code: int, data: Variant) -> void:
		_busy = false
		if code == 200 and data is Dictionary:
			var result: Dictionary = data.get("data", {})
			var trajectory: Dictionary = result.get("trajectory", {})
			_status.text = "Coevolution complete.\n%d cycles · %.1f model hours\nViable tumour trajectory: %s\nResistant fraction trajectory: %s\nLead: %s" % [int(result.get("n_cycles", 0)), float(result.get("total_hours", 0)), str(trajectory.get("viable_tumor_cells", [])), str(trajectory.get("resistant_fraction", [])), str(result.get("lead_compound_final", {}).get("name", "Unknown"))]
		else:
			_status.text = "Coevolution failed: " + str(data)
		ApiClient.fetch_state()
	)
