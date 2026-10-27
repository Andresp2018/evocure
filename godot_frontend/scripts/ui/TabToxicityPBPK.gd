class_name TabToxicityPBPK
extends Control
var _query: LineEdit
var _pbpk: CheckBox
var _status: Label
var _risks: VBoxContainer
var _summary: Label
var _chart: RealtimeLineChart
var _busy: bool = false

func _ready() -> void:
	ThemeBuilder.fill_rect(self)
	var root := VBoxContainer.new()
	ThemeBuilder.fill_rect(root)
	add_child(root)
	var controls := HBoxContainer.new()
	root.add_child(controls)
	_query = LineEdit.new()
	_query.text = "Gemcitabine"
	_query.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	controls.add_child(_query)
	_pbpk = CheckBox.new()
	_pbpk.text = "Include exploratory systemic PBPK"
	controls.add_child(_pbpk)
	var run := Button.new()
	run.text = "Run assessment"
	run.pressed.connect(_run)
	controls.add_child(run)
	_status = Label.new()
	_status.text = "No assessment yet. Risk scores are computational screening outputs, not probabilities of human harm."
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	root.add_child(_status)
	var panels := HBoxContainer.new()
	panels.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(panels)
	var scroll := ScrollContainer.new()
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	panels.add_child(scroll)
	_risks = VBoxContainer.new()
	_risks.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_risks)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panels.add_child(right)
	_chart = RealtimeLineChart.new()
	_chart.custom_minimum_size.y = 300
	_chart.configure("PBPK concentration (uM)", false, true)
	_chart.x_axis_title = "Time (hours)"
	right.add_child(_chart)
	var summary_scroll := ScrollContainer.new()
	summary_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right.add_child(summary_scroll)
	_summary = Label.new()
	_summary.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_summary.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_summary.text = "PBPK and systemic biomarkers have not been calculated."
	summary_scroll.add_child(_summary)
	visibility_changed.connect(func() -> void:
		if visible and not AppState.analysis_compound_query.is_empty():
			_query.text = AppState.analysis_compound_query
	)

func _run() -> void:
	if _busy:
		return
	_busy = true
	_status.text = "Computing structure-based screening and selected models..."
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/compound/" + _query.text.uri_encode() + ("?enable_systemic_pbpk=true" if _pbpk.button_pressed else ""), null, func(code: int, result: Variant) -> void:
		_busy = false
		if code != 200 or not result is Dictionary:
			_status.text = "Assessment failed: " + str(result)
			return
		_status.text = "Calculated for %s · applicability: %s\n%s" % [str(result.get("name", _query.text)), JSON.stringify(result.get("applicability_domain", {})), str(result.get("epistemic_disclaimer", "Exploratory model"))]
		for child in _risks.get_children():
			child.queue_free()
		var toxicity: Dictionary = result.get("toxicology", {})
		for key in toxicity:
			var value: Variant = toxicity[key]
			if not value is Dictionary or not value.has("value"):
				continue
			var label := Label.new()
			label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			label.text = "%s: %s %s\nCategory: %s | uncertainty [%s, %s]\nSource: %s" % [str(value.get("name", key)), str(value.get("value", "Unknown")), str(value.get("unit", "")), str(value.get("epistemic_category", "Unknown")), str(value.get("uncertainty_low", "unknown")), str(value.get("uncertainty_high", "unknown")), str(value.get("source", "Unknown"))]
			_risks.add_child(label)
			if value["value"] is float or value["value"] is int:
				var bar := ProgressBar.new()
				bar.max_value = 1 if float(value["value"]) <= 1 else 100
				bar.value = float(value["value"])
				bar.show_percentage = false
				_risks.add_child(bar)
		var index := Label.new()
		index.text = "Aggregate computational risk index: " + str(toxicity.get("internal_computational_risk_index_0_100", "Unknown")) + " / 100"
		_risks.add_child(index)
		var pbpk: Variant = result.get("systemic_pbpk")
		if pbpk is Dictionary:
			var concentrations: Dictionary = pbpk.get("concentrations", {})
			var colors := {"blood": Color.CYAN, "liver": Color.CORAL, "kidneys": Color.GREEN, "bone_marrow": Color.VIOLET, "pancreas": Color.YELLOW, "tumor": Color.MAGENTA, "peripheral": Color.SKY_BLUE}
			_chart.set_multi_series(pbpk.get("time_points_hours", []), concentrations, colors)
			_summary.text = "PBPK model input: %s mg, %s\nHalf-life: %s hours\nPeak concentrations (uM): %s\nModel-derived biological indicators:\n%s" % [str(pbpk.get("dose_mg", "Unknown")), str(pbpk.get("administration_route", "Unknown")), str(pbpk.get("elimination_half_life_hours", {}).get("value", "Unknown")), JSON.stringify(pbpk.get("cmax", {})), JSON.stringify(result.get("biological_indicators", {}), "  ")]
		else:
			_chart.set_multi_series([], {}, {})
			_summary.text = "Systemic PBPK disabled. Systemic biomarkers are not calculated.\nTissue/model indicators:\n" + JSON.stringify(result.get("biological_indicators", {}), "  ")
	)
