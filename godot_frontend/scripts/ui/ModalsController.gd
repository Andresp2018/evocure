class_name ModalsController
extends Control
## Central Controller for Overlays, Modals, and Notifications in EVOCURE.
## Manages AI Provider Settings, 45-Compound Library/SDF Import,
## Guided Missions, Scientific Glossary, and non-blocking Toasts.

signal modal_closed(modal_name: String)

var _overlay_backdrop: ColorRect
var _modal_center: CenterContainer
var _ai_modal: PanelContainer
var _library_modal: PanelContainer
var _missions_modal: PanelContainer
var _glossary_modal: PanelContainer

# AI Provider Modal fields
var _ai_provider_opt: OptionButton
var _ai_model_opt: OptionButton
var _ai_custom_model: LineEdit
var _ai_key_input: LineEdit
var _ai_base_url_input: LineEdit
var _ai_persist_checkbox: CheckBox
var _ai_status_label: Label
var _bloom_toggle: CheckBox
var _colorblind_opt: OptionButton

# Library Modal fields
var _lib_search_input: LineEdit
var _lib_category_opt: OptionButton
var _lib_grid_container: VBoxContainer
var _custom_smiles_input: LineEdit
var _custom_role_opt: OptionButton
var _sdf_path_input: LineEdit

# Glossary & Tutorial fields
var _glossary_search: LineEdit
var _glossary_list: VBoxContainer
var _tutorial_card: PanelContainer
var _tutorial_step_label: Label
var _tutorial_title_label: Label
var _tutorial_desc_label: Label
var _tutorial_index: int = 0

# Toast banner
var _toast_box: PanelContainer
var _toast_label: Label
var _toast_timer: Timer

const TUTORIAL_STEPS: Array[Dictionary] = [
	{
		"step": "Step 1/10",
		"title": "Welcome to EVOCURE: Bioluminescent Universe",
		"desc": "EVOCURE simulates prebiotic molecular ecosystems and tumor-immune coevolution under biophysical conservation laws. Observe protocells, catalysts, and drug molecules in real time."
	},
	{
		"step": "Step 2/10",
		"title": "Dual Simulation Modes: LAB vs PLAY",
		"desc": "LAB mode lets you observe prebiotic emergence freely without tumor pressure. PLAY mode introduces aggressive cancer/biofilm populations with targeted research objectives."
	},
	{
		"step": "Step 3/10",
		"title": "Adding Molecular Species",
		"desc": "Use the Left Sidebar in the Chemical Universe to deploy basic building blocks (Glucose, Glycine, Lipids) or antitumor compounds (Gemcitabine). Open the 45-Compound Library for custom structures."
	},
	{
		"step": "Step 4/10",
		"title": "Biophysical Microenvironment Presets",
		"desc": "Select from 6 clinical tissue environments: Pancreatic Stroma (dense desmoplasia), Bloodstream (shear flow), Tumor Core (hypoxic, acidic pH 6.6), and Oxidative Stress."
	},
	{
		"step": "Step 5/10",
		"title": "Evolution Lab: NSGA-II Discovery",
		"desc": "Switch to Evolution Lab to execute 15-stage Pareto optimization balancing Antitumor Potency, Minimal Toxicity, and Synthetic Accessibility."
	},
	{
		"step": "Step 6/10",
		"title": "Cancer Battle: Combination Synergy",
		"desc": "Evaluate 2-drug cocktails on an 8x8 Bliss/Loewe synergy grid. Test drug penetration across the ECM and track tumor subclone resistance emergence."
	},
	{
		"step": "Step 7/10",
		"title": "Family Tree & Live RDKit Editor",
		"desc": "Inspect full evolutionary genealogies. In the 2D Editor, apply bioisosteric transformations, view RDKit structural diffs, and spawn modified molecules back into the universe."
	},
	{
		"step": "Step 8/10",
		"title": "Toxicity & 7-Compartment Systemic PBPK",
		"desc": "Predict 10 QSAR organ toxicities (hERG, DILI, Ames) with Tanimoto applicability domain boundaries. Activate 7-compartment PBPK to model systemic clearance."
	},
	{
		"step": "Step 9/10",
		"title": "Multi-Provider AI Assistant with Scientific Guardrails",
		"desc": "Leverage offline Local Gemma 4 GGUF, Google Gemini 2.5 Flash, or OpenAI. All quantitative chemistry is computed by deterministic backends; the AI provides grounded hypotheses."
	},
	{
		"step": "Step 10/10",
		"title": "Reproducible Scientific Reports",
		"desc": "Export native binary PDF reports, Markdown summaries, CSV kinetics, and SDF chemical libraries with one click from the AI Assistant tab."
	}
]

const GLOSSARY_ITEMS: Array[Dictionary] = [
	{
		"term": "NSGA-II Multi-Objective Optimization",
		"category": "Algorithm",
		"definition": "Non-dominated Sorting Genetic Algorithm II. Optimizes competing chemical objectives (potency vs toxicity vs synthesizability) to find an optimal Pareto frontier."
	},
	{
		"term": "Bliss Independence vs Loewe Additivity",
		"category": "Synergy Metric",
		"definition": "Bliss independence models independent non-interacting biological mechanisms. Loewe additivity tests whether drugs behave as dilutions of each other. Combination Index (CI) < 1.0 indicates synergy."
	},
	{
		"term": "7-Compartment PBPK Model",
		"category": "Pharmacology",
		"definition": "Physiologically Based Pharmacokinetic simulation tracking drug concentrations across Blood, Liver, Kidney, Tumor Interstitium, Heart, Brain, and Adipose."
	},
	{
		"term": "Tanimoto Applicability Domain",
		"category": "Cheminformatics",
		"definition": "Calculates Morgan fingerprint similarity (0.0 to 1.0) against training databases. Scores < 0.40 flag novel chemical space where ML uncertainty is elevated."
	},
	{
		"term": "Epistemic Category vs AI Hypothesis",
		"category": "Transparency",
		"definition": "Clear separation between mathematically validated simulation data (PBPK ODEs, QSAR models) and natural language AI hypotheses."
	},
	{
		"term": "Reaction-Diffusion Spatial PDEs",
		"category": "Biophysics",
		"definition": "Partial differential equations modeling spatial gradients of Oxygen, Glucose, and Chemotherapeutics across heterogeneous tumor tissue."
	},
	{
		"term": "Hill Cooperative Binding Kinetics",
		"category": "Biochemistry",
		"definition": "Nonlinear enzymatic reaction kinetics modeling allosteric activation and substrate saturation in evolved catalyst molecules."
	}
]


func _ready() -> void:
	ThemeBuilder.fill_rect(self)
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	_build_ui()
	_connect_signals()


func _build_ui() -> void:
	# Full-screen semi-transparent backdrop for modals
	_overlay_backdrop = ColorRect.new()
	ThemeBuilder.fill_rect(_overlay_backdrop)
	_overlay_backdrop.color = Color(0.015, 0.03, 0.06, 0.78)
	_overlay_backdrop.mouse_filter = Control.MOUSE_FILTER_STOP
	_overlay_backdrop.visible = false
	_overlay_backdrop.gui_input.connect(func(event: InputEvent) -> void:
		if event is InputEventMouseButton and event.pressed:
			close_all_modals()
	)
	add_child(_overlay_backdrop)

	_modal_center = CenterContainer.new()
	ThemeBuilder.fill_rect(_modal_center)
	_modal_center.mouse_filter = Control.MOUSE_FILTER_PASS
	_overlay_backdrop.add_child(_modal_center)

	_build_ai_modal()
	_build_library_modal()
	_build_missions_modal()
	_build_glossary_modal()
	_build_toast_banner()


# ==============================================================================
# 1. AI PROVIDERS & ENGINE SETTINGS MODAL
# ==============================================================================
func _build_ai_modal() -> void:
	_ai_modal = PanelContainer.new()
	_ai_modal.visible = false
	_ai_modal.custom_minimum_size = Vector2(680, 580)
	_ai_modal.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_CYAN, 12, 2, 24, 20, 16, Color(0.13, 0.82, 0.93, 0.35)))
	_modal_center.add_child(_ai_modal)

	var root_vbox := VBoxContainer.new()
	root_vbox.add_theme_constant_override("separation", 14)
	_ai_modal.add_child(root_vbox)

	# Header
	var header_hbox := HBoxContainer.new()
	var title_lbl := Label.new()
	title_lbl.text = "AI PROVIDER & SCIENTIFIC ENGINE SETTINGS"
	title_lbl.add_theme_font_size_override("font_size", 16)
	title_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
	title_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header_hbox.add_child(title_lbl)

	var close_btn := Button.new()
	close_btn.text = "✕"
	close_btn.custom_minimum_size = Vector2(30, 30)
	close_btn.pressed.connect(close_all_modals)
	header_hbox.add_child(close_btn)
	root_vbox.add_child(header_hbox)

	# Epistemic Guardrail Notice Banner
	var guardrail_box := PanelContainer.new()
	guardrail_box.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(Color(0.08, 0.15, 0.28, 0.8), ThemeBuilder.BORDER_EMERALD, 8, 1, 12, 8))
	var guardrail_lbl := Label.new()
	guardrail_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	guardrail_lbl.text = "Scientific Guardrail: Whether using Local Gemma 4, Google Gemini, or OpenAI, all biological efficacy and toxicity metrics are computed strictly by EVOCURE's validated mathematical/QSAR models, never invented by an LLM."
	guardrail_lbl.add_theme_font_size_override("font_size", 11)
	guardrail_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD)
	guardrail_box.add_child(guardrail_lbl)
	root_vbox.add_child(guardrail_box)

	# Provider Selection
	var form_grid := GridContainer.new()
	form_grid.columns = 2
	form_grid.add_theme_constant_override("h_separation", 14)
	form_grid.add_theme_constant_override("v_separation", 10)
	form_grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root_vbox.add_child(form_grid)

	# Row 1: Active Provider
	var prov_lbl := Label.new()
	prov_lbl.text = "Active AI Provider:"
	form_grid.add_child(prov_lbl)

	_ai_provider_opt = OptionButton.new()
	_ai_provider_opt.add_item("Auto Local (Gemma 4 / Offline Default)", 0)
	_ai_provider_opt.set_item_metadata(0, "auto_local")
	_ai_provider_opt.add_item("Local Gemma 4 (GGUF CUDA/CPU)", 1)
	_ai_provider_opt.set_item_metadata(1, "local_gemma")
	_ai_provider_opt.add_item("Google Gemini API", 2)
	_ai_provider_opt.set_item_metadata(2, "gemini")
	_ai_provider_opt.add_item("OpenAI API (GPT / Compatible)", 3)
	_ai_provider_opt.set_item_metadata(3, "openai")
	_ai_provider_opt.add_item("Deterministic Grounded Scientific Engine", 4)
	_ai_provider_opt.set_item_metadata(4, "deterministic")
	_ai_provider_opt.item_selected.connect(_on_ai_provider_selected)
	form_grid.add_child(_ai_provider_opt)

	# Row 2: Model Preset
	var model_lbl := Label.new()
	model_lbl.text = "Model Preset:"
	form_grid.add_child(model_lbl)

	_ai_model_opt = OptionButton.new()
	_ai_model_opt.item_selected.connect(func(idx: int) -> void:
		_ai_custom_model.text = _ai_model_opt.get_item_text(idx)
	)
	form_grid.add_child(_ai_model_opt)

	# Row 3: Custom Model ID
	var custom_lbl := Label.new()
	custom_lbl.text = "Model Identifier:"
	form_grid.add_child(custom_lbl)

	_ai_custom_model = LineEdit.new()
	_ai_custom_model.placeholder_text = "e.g. gemini-2.5-flash or gpt-4o"
	form_grid.add_child(_ai_custom_model)

	# Row 4: Masked API Key
	var key_lbl := Label.new()
	key_lbl.text = "API Key (Secret):"
	form_grid.add_child(key_lbl)

	_ai_key_input = LineEdit.new()
	_ai_key_input.secret = true
	_ai_key_input.placeholder_text = "Enter API Key... (Encrypted in transit)"
	form_grid.add_child(_ai_key_input)

	# Row 5: Custom Base URL
	var url_lbl := Label.new()
	url_lbl.text = "Base URL (Optional):"
	form_grid.add_child(url_lbl)

	_ai_base_url_input = LineEdit.new()
	_ai_base_url_input.placeholder_text = "Default API URL (or custom proxy)"
	form_grid.add_child(_ai_base_url_input)

	# Row 6: Persist
	var persist_lbl := Label.new()
	persist_lbl.text = "Environment:"
	form_grid.add_child(persist_lbl)

	_ai_persist_checkbox = CheckBox.new()
	_ai_persist_checkbox.text = "Save configuration to local .env"
	_ai_persist_checkbox.button_pressed = true
	form_grid.add_child(_ai_persist_checkbox)

	# Visual & Accessibility settings
	var vis_sep := HSeparator.new()
	root_vbox.add_child(vis_sep)

	var vis_grid := GridContainer.new()
	vis_grid.columns = 2
	vis_grid.add_theme_constant_override("h_separation", 14)
	vis_grid.add_theme_constant_override("v_separation", 8)
	root_vbox.add_child(vis_grid)

	var bloom_lbl := Label.new()
	bloom_lbl.text = "2D Bioluminescent Bloom:"
	vis_grid.add_child(bloom_lbl)

	_bloom_toggle = CheckBox.new()
	_bloom_toggle.text = "Enable glow effects"
	_bloom_toggle.button_pressed = AppState.bloom_enabled
	_bloom_toggle.toggled.connect(func(toggled_on: bool) -> void:
		AppState.bloom_enabled = toggled_on
		AppState.notify_toast("Bloom " + ("enabled" if toggled_on else "disabled"), "info")
	)
	vis_grid.add_child(_bloom_toggle)

	var pal_lbl := Label.new()
	pal_lbl.text = "Color Palette:"
	vis_grid.add_child(pal_lbl)

	_colorblind_opt = OptionButton.new()
	_colorblind_opt.add_item("Bioluminescent Cyan/Emerald (Default)", 0)
	_colorblind_opt.add_item("Okabe-Ito Colorblind Safe", 1)
	_colorblind_opt.add_item("High-Contrast Neon", 2)
	_colorblind_opt.item_selected.connect(func(idx: int) -> void:
		var pals := ["bioluminescent", "okabe_ito", "high_contrast"]
		AppState.colorblind_palette = pals[idx]
		AppState.notify_toast("Palette changed to " + pals[idx], "info")
	)
	vis_grid.add_child(_colorblind_opt)

	# Live Status / Latency Banner
	_ai_status_label = Label.new()
	_ai_status_label.text = "Ready to connect or configure AI provider."
	_ai_status_label.add_theme_font_size_override("font_size", 12)
	_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	root_vbox.add_child(_ai_status_label)

	# Action Buttons
	var btn_hbox := HBoxContainer.new()
	btn_hbox.add_theme_constant_override("separation", 10)
	btn_hbox.alignment = BoxContainer.ALIGNMENT_END
	root_vbox.add_child(btn_hbox)

	var local_server_btn := Button.new()
	local_server_btn.text = "⚡ Start Local Gemma Server"
	local_server_btn.pressed.connect(_on_start_local_gemma_clicked)
	btn_hbox.add_child(local_server_btn)

	var test_btn := Button.new()
	test_btn.text = "🔍 Test Connection"
	test_btn.pressed.connect(_on_test_ai_clicked)
	btn_hbox.add_child(test_btn)

	var save_btn := Button.new()
	save_btn.text = "✓ Save & Activate"
	ThemeBuilder.style_accent_button(save_btn, ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_CYAN, Color(0.13, 0.82, 0.93, 0.4))
	save_btn.pressed.connect(_on_save_ai_clicked)
	btn_hbox.add_child(save_btn)

	# Initialize provider dropdown state
	_populate_model_presets_for("auto_local")


func _on_ai_provider_selected(idx: int) -> void:
	var prov_id: String = _ai_provider_opt.get_item_metadata(idx)
	_populate_model_presets_for(prov_id)


func _populate_model_presets_for(prov_id: String) -> void:
	_ai_model_opt.clear()
	_ai_custom_model.editable = prov_id not in ["auto_local", "local_gemma", "deterministic"]
	match prov_id:
		"gemini":
			_ai_model_opt.add_item("gemini-2.5-flash (example model ID)")
			_ai_model_opt.add_item("gemini-2.5-pro (example model ID)")
			_ai_model_opt.add_item("gemini-2.0-flash")
			_ai_custom_model.text = "gemini-2.5-flash"
			_ai_base_url_input.text = ""
			_ai_base_url_input.placeholder_text = "https://generativelanguage.googleapis.com/v1beta"
		"openai":
			_ai_model_opt.add_item("gpt-4o-mini (example model ID)")
			_ai_model_opt.add_item("gpt-4o (example model ID)")
			_ai_model_opt.add_item("o3-mini (example model ID)")
			_ai_custom_model.text = "gpt-4o-mini"
			_ai_base_url_input.text = "https://api.openai.com/v1"
		"local_gemma":
			_ai_model_opt.add_item("Gemma-4-E2B-Heretic-Q4_K_M (Default Offline)")
			_ai_custom_model.text = "Gemma-4-E2B-Heretic-Q4_K_M"
			_ai_base_url_input.text = "http://127.0.0.1:18089"
		"deterministic":
			_ai_model_opt.add_item("Deterministic Grounded Scientific Engine")
			_ai_custom_model.text = "offline-grounded-v2"
			_ai_base_url_input.text = ""
		_: # auto_local
			_ai_model_opt.add_item("Auto Local (Gemma 4 / Offline Engine)")
			_ai_custom_model.text = "gemma-4-E2B-it-Q4_K_M"
			_ai_base_url_input.text = ""
	var configured: Dictionary = AppState.ai_providers_status.get("providers", {}).get(prov_id, {})
	_ai_custom_model.text = str(configured.get("current_model", _ai_custom_model.text))
	_ai_base_url_input.text = str(configured.get("base_url", _ai_base_url_input.text))


func _on_test_ai_clicked() -> void:
	var idx: int = _ai_provider_opt.selected
	var prov_id: String = _ai_provider_opt.get_item_metadata(idx)
	var model: String = _ai_custom_model.text.strip_edges()
	var key: String = _ai_key_input.text.strip_edges()
	var url: String = _ai_base_url_input.text.strip_edges()

	_ai_status_label.text = "Testing connection to " + prov_id + " (" + model + ")..."
	_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_AMBER)

	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").test_ai_provider(prov_id, model, key, url, func(ok: bool, res: Variant) -> void:
			if ok and res is Dictionary:
				var msg: String = str(res.get("message", "Connection verified successfully."))
				_ai_status_label.text = "✓ OK: " + msg + (" (%s ms)" % str(res["latency_ms"]) if res.has("latency_ms") else "")
				_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD)
			else:
				var err_msg: String = str(res.get("error", "Failed to reach provider")) if res is Dictionary else "Failed"
				_ai_status_label.text = "✗ Connection Error: " + err_msg
				_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.BORDER_CRIMSON)
		)


func _on_start_local_gemma_clicked() -> void:
	_ai_status_label.text = "Spawning Local Gemma GGUF background server..."
	_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_AMBER)
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").request_json(HTTPClient.METHOD_POST, "/api/v1/assistant/start-local-llm", {}, func(code: int, res: Variant) -> void:
			if code == 200 and res is Dictionary and bool(res.get("success", false)):
				_ai_status_label.text = "✓ Local Gemma 4 server is ready."
				_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD)
				AppState.notify_toast("Local Gemma 4 Server started", "success")
			else:
				_ai_status_label.text = "Local Gemma server status: " + str(res)
				_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_AMBER)
		)


func _on_save_ai_clicked() -> void:
	var idx: int = _ai_provider_opt.selected
	var prov_id: String = _ai_provider_opt.get_item_metadata(idx)
	var model: String = _ai_custom_model.text.strip_edges()
	var key: String = _ai_key_input.text.strip_edges()
	var url: String = _ai_base_url_input.text.strip_edges()
	var persist: bool = _ai_persist_checkbox.button_pressed

	_ai_status_label.text = "Saving and activating provider..."
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").configure_ai_provider(prov_id, model, key, url, persist, func(ok: bool, res: Variant) -> void:
			if ok:
				AppState.ai_providers_status["active_provider"] = prov_id
				AppState.ai_providers_status["active_model"] = model
				AppState.notify_toast("Active AI Provider updated to: " + prov_id, "success")
				close_all_modals()
			else:
				_ai_status_label.text = "Error saving provider: " + str(res)
				_ai_status_label.add_theme_color_override("font_color", ThemeBuilder.BORDER_CRIMSON)
		)


# ==============================================================================
# 2. 45-COMPOUND LIBRARY, CUSTOM SMILES & SDF IMPORT MODAL
# ==============================================================================
func _build_library_modal() -> void:
	_library_modal = PanelContainer.new()
	_library_modal.visible = false
	_library_modal.custom_minimum_size = Vector2(820, 640)
	_library_modal.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_CYAN, 12, 2, 20, 18, 16, Color(0.13, 0.82, 0.93, 0.35)))
	_modal_center.add_child(_library_modal)

	var root_vbox := VBoxContainer.new()
	root_vbox.add_theme_constant_override("separation", 12)
	_library_modal.add_child(root_vbox)

	# Header
	var header_hbox := HBoxContainer.new()
	var title_lbl := Label.new()
	title_lbl.text = "45-COMPOUND SCIENTIFIC LIBRARY & SDF IMPORTER"
	title_lbl.add_theme_font_size_override("font_size", 16)
	title_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
	title_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header_hbox.add_child(title_lbl)

	var close_btn := Button.new()
	close_btn.text = "✕"
	close_btn.custom_minimum_size = Vector2(30, 30)
	close_btn.pressed.connect(close_all_modals)
	header_hbox.add_child(close_btn)
	root_vbox.add_child(header_hbox)

	# Filter bar
	var filter_hbox := HBoxContainer.new()
	filter_hbox.add_theme_constant_override("separation", 10)
	root_vbox.add_child(filter_hbox)

	_lib_search_input = LineEdit.new()
	_lib_search_input.placeholder_text = "Search by Compound Name, SMILES, or Formula..."
	_lib_search_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_lib_search_input.text_changed.connect(func(_t: String) -> void: _refresh_library_cards())
	filter_hbox.add_child(_lib_search_input)

	_lib_category_opt = OptionButton.new()
	_lib_category_opt.add_item("All Categories (45)", 0)
	_lib_category_opt.add_item("Prebiotic Autotrophs", 1)
	_lib_category_opt.add_item("Catalytic & Coenzymes", 2)
	_lib_category_opt.add_item("Chemotherapeutics & Antitumor", 3)
	_lib_category_opt.add_item("Lipids & Membrane Formers", 4)
	_lib_category_opt.item_selected.connect(func(_i: int) -> void: _refresh_library_cards())
	filter_hbox.add_child(_lib_category_opt)

	# Scrollable Cards View
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(0, 320)
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root_vbox.add_child(scroll)

	_lib_grid_container = VBoxContainer.new()
	_lib_grid_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_lib_grid_container.add_theme_constant_override("separation", 6)
	scroll.add_child(_lib_grid_container)

	# Custom SMILES & SDF Importer Box
	var custom_panel := PanelContainer.new()
	custom_panel.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 12, 10))
	root_vbox.add_child(custom_panel)

	var custom_vbox := VBoxContainer.new()
	custom_vbox.add_theme_constant_override("separation", 8)
	custom_panel.add_child(custom_vbox)

	var custom_title := Label.new()
	custom_title.text = "Custom SMILES / SDF Structure Input"
	custom_title.add_theme_color_override("font_color", ThemeBuilder.TEXT_CYAN)
	custom_vbox.add_child(custom_title)

	var smiles_hbox := HBoxContainer.new()
	smiles_hbox.add_theme_constant_override("separation", 8)
	custom_vbox.add_child(smiles_hbox)

	_custom_smiles_input = LineEdit.new()
	_custom_smiles_input.placeholder_text = "Enter SMILES string (e.g. C1=CC=C(C=C1)O)..."
	_custom_smiles_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	smiles_hbox.add_child(_custom_smiles_input)

	_custom_role_opt = OptionButton.new()
	_custom_role_opt.add_item("autotroph", 0)
	_custom_role_opt.add_item("catalyst", 1)
	_custom_role_opt.add_item("drug", 2)
	_custom_role_opt.add_item("structural", 3)
	smiles_hbox.add_child(_custom_role_opt)

	var spawn_custom_btn := Button.new()
	spawn_custom_btn.text = "Spawn in Universe"
	spawn_custom_btn.pressed.connect(_on_spawn_custom_smiles_clicked)
	smiles_hbox.add_child(spawn_custom_btn)

	var sdf_hbox := HBoxContainer.new()
	sdf_hbox.add_theme_constant_override("separation", 8)
	custom_vbox.add_child(sdf_hbox)

	_sdf_path_input = LineEdit.new()
	_sdf_path_input.placeholder_text = "Path to .sdf file or paste SDF block..."
	_sdf_path_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	sdf_hbox.add_child(_sdf_path_input)

	var import_sdf_btn := Button.new()
	import_sdf_btn.text = "Import SDF File"
	import_sdf_btn.pressed.connect(_on_import_sdf_clicked)
	sdf_hbox.add_child(import_sdf_btn)


func _refresh_library_cards() -> void:
	for c in _lib_grid_container.get_children():
		c.queue_free()

	var query: String = _lib_search_input.text.strip_edges().to_lower()
	var cat_idx: int = _lib_category_opt.selected
	var list: Array = AppState.compound_library

	if list.is_empty():
		AppState.notify_toast("Compound library unavailable. Check the engine connection.", "error")

	for item in list:
		if not (item is Dictionary):
			continue
		var name_str: String = str(item.get("name", item.get("id", "Compound")))
		var smiles: String = str(item.get("smiles", ""))
		var role: String = str(item.get("ecological_role", item.get("category", "autotroph"))).to_lower()
		var mw: float = float(item.get("mw", 180.15))

		# Filters
		if not query.is_empty():
			if not name_str.to_lower().contains(query) and not smiles.to_lower().contains(query):
				continue

		if cat_idx == 1 and not (role.contains("auto") or role.contains("prebiotic")):
			continue
		elif cat_idx == 2 and not (role.contains("cat") or role.contains("enzyme")):
			continue
		elif cat_idx == 3 and not (role.contains("drug") or role.contains("antitumor") or role.contains("chemo")):
			continue
		elif cat_idx == 4 and not (role.contains("lipid") or role.contains("struct") or role.contains("membrane")):
			continue

		var row := PanelContainer.new()
		row.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 6, 1, 10, 6))
		var row_hbox := HBoxContainer.new()
		row_hbox.add_theme_constant_override("separation", 10)
		row.add_child(row_hbox)

		var name_lbl := Label.new()
		name_lbl.text = name_str
		name_lbl.custom_minimum_size = Vector2(160, 0)
		name_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_CYAN)
		row_hbox.add_child(name_lbl)

		var role_lbl := Label.new()
		role_lbl.text = "[" + role.capitalize() + "]"
		role_lbl.custom_minimum_size = Vector2(100, 0)
		role_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
		row_hbox.add_child(role_lbl)

		var smiles_lbl := Label.new()
		smiles_lbl.text = smiles
		smiles_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		smiles_lbl.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
		row_hbox.add_child(smiles_lbl)

		var mw_lbl := Label.new()
		mw_lbl.text = str(snappedf(mw, 0.01)) + " Da"
		mw_lbl.custom_minimum_size = Vector2(75, 0)
		row_hbox.add_child(mw_lbl)

		var inspect_btn := Button.new()
		inspect_btn.text = "Inspect"
		inspect_btn.pressed.connect(func() -> void:
			if has_node("/root/ApiClient"):
				get_node("/root/ApiClient").inspect_compound(name_str)
			close_all_modals()
		)
		row_hbox.add_child(inspect_btn)

		var spawn_btn := Button.new()
		spawn_btn.text = "+10 Universe"
		spawn_btn.pressed.connect(func() -> void:
			if has_node("/root/ApiClient"):
				get_node("/root/ApiClient").spawn_compound(name_str, 10, null, null, role)
		)
		row_hbox.add_child(spawn_btn)

		_lib_grid_container.add_child(row)


func _get_fallback_compounds() -> Array:
	return [
		{"name": "D-Glucose", "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O", "mw": 180.16, "ecological_role": "autotroph"},
		{"name": "Glycine", "smiles": "C(C(=O)O)N", "mw": 75.07, "ecological_role": "autotroph"},
		{"name": "L-Alanine", "smiles": "CC(C(=O)O)N", "mw": 89.09, "ecological_role": "autotroph"},
		{"name": "Palmitic Acid", "smiles": "CCCCCCCCCCCCCCCC(=O)O", "mw": 256.42, "ecological_role": "structural"},
		{"name": "Gemcitabine", "smiles": "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F", "mw": 263.20, "ecological_role": "drug"},
		{"name": "Cisplatin", "smiles": "N.N.[Cl-].[Cl-].[Pt+2]", "mw": 300.05, "ecological_role": "drug"},
		{"name": "Paclitaxel", "smiles": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C", "mw": 853.91, "ecological_role": "drug"},
		{"name": "Imidazole Catalyst", "smiles": "C1=CN=CN1", "mw": 68.08, "ecological_role": "catalyst"},
		{"name": "Phosphoenolpyruvate", "smiles": "C=C(C(=O)O)OP(=O)(O)O", "mw": 168.04, "ecological_role": "catalyst"},
		{"name": "ATP Triphosphate", "smiles": "C1=NC2=C(C(=N1)N)N=CN2C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O", "mw": 507.18, "ecological_role": "catalyst"},
		{"name": "Glutathione", "smiles": "C(CC(=O)NC(CS)C(=O)NCC(=O)O)C(C(=O)O)N", "mw": 307.32, "ecological_role": "catalyst"},
		{"name": "Erlotinib", "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC", "mw": 393.44, "ecological_role": "drug"},
		{"name": "Sotorasib", "smiles": "CC(C)C1=C(C(=C(C(=N1)C2=C(C=CC(=C2F)Cl)O)F)N3CCN(CC3)C(=O)C=C)C4=C(C=CC=N4)F", "mw": 560.98, "ecological_role": "drug"},
		{"name": "Nicotinamide Adenine Dinucleotide", "smiles": "C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C4N=CN=C5N)O)O)O)O)C(=O)N", "mw": 663.43, "ecological_role": "catalyst"},
		{"name": "Ribose-5-Phosphate", "smiles": "C(C1C(C(C(O1)O)O)O)OP(=O)(O)O", "mw": 230.11, "ecological_role": "autotroph"},
	]


func _on_spawn_custom_smiles_clicked() -> void:
	var smi: String = _custom_smiles_input.text.strip_edges()
	if smi.is_empty():
		AppState.notify_toast("Please enter a valid SMILES string", "warning")
		return
	var role: String = _custom_role_opt.get_item_text(_custom_role_opt.selected)
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").spawn_compound(smi, 10, null, null, role, "MOLECULE")
	_custom_smiles_input.text = ""


func _on_import_sdf_clicked() -> void:
	var path: String = _sdf_path_input.text.strip_edges()
	if path.is_empty():
		AppState.notify_toast("Specify SDF path or paste SDF block", "warning")
		return
	var sdf_content: String = path
	if FileAccess.file_exists(path):
		sdf_content = FileAccess.get_file_as_string(path)
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").request_json(HTTPClient.METHOD_POST, "/api/v1/world/import-sdf", {
			"sdf_text": sdf_content,
			"spawn_in_world": true
		}, func(code: int, res: Variant) -> void:
			if code == 200:
				AppState.notify_toast("Successfully imported SDF compounds into Universe", "success")
				get_node("/root/ApiClient").fetch_library()
				get_node("/root/ApiClient").fetch_state()
			else:
				AppState.notify_toast("SDF import failed: " + str(res), "error")
		)


# ==============================================================================
# 3. GUIDED MISSIONS & ACHIEVEMENTS MODAL
# ==============================================================================
func _build_missions_modal() -> void:
	_missions_modal = PanelContainer.new()
	_missions_modal.visible = false
	_missions_modal.custom_minimum_size = Vector2(760, 560)
	_missions_modal.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_EMERALD, 12, 2, 24, 20, 16, Color(0.20, 0.82, 0.60, 0.35)))
	_modal_center.add_child(_missions_modal)

	var root_vbox := VBoxContainer.new()
	root_vbox.add_theme_constant_override("separation", 14)
	_missions_modal.add_child(root_vbox)

	var header_hbox := HBoxContainer.new()
	var title_lbl := Label.new()
	title_lbl.text = "RESEARCH MISSIONS & SCIENTIFIC ACHIEVEMENTS"
	title_lbl.add_theme_font_size_override("font_size", 16)
	title_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD)
	title_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header_hbox.add_child(title_lbl)

	var close_btn := Button.new()
	close_btn.text = "✕"
	close_btn.custom_minimum_size = Vector2(30, 30)
	close_btn.pressed.connect(close_all_modals)
	header_hbox.add_child(close_btn)
	root_vbox.add_child(header_hbox)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root_vbox.add_child(scroll)

	var list_vbox := VBoxContainer.new()
	list_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	list_vbox.add_theme_constant_override("separation", 8)
	scroll.add_child(list_vbox)

	var missions := [
		{"id": "M1", "title": "Prebiotic Spark: Synthesize First Autotroph Protocell", "desc": "Deploy basic amino acids and glucose in prebiotic water, then advance 100 years.", "status": "Completed", "reward": "+100 EcoPoints"},
		{"id": "M2", "title": "Dense Stroma Challenge: PDAC Tumor Penetration", "desc": "Penetrate dense pancreatic desmoplasia with an evolved catalyst or Gemcitabine analogue.", "status": "In Progress", "reward": "+250 EcoPoints"},
		{"id": "M3", "title": "Synergy Seeker: 2-Drug Combination CI < 0.70", "desc": "Run an 8x8 combination assay yielding strong Bliss/Loewe synergy without systemic liver toxicity.", "status": "In Progress", "reward": "+300 EcoPoints"},
		{"id": "M4", "title": "Nonlinear Kinetics: Evolve Catalyst k_cat > 15 s⁻¹", "desc": "Mutate phosphate/imidazole enzymes in the 2D Editor to accelerate substrate turnover.", "status": "Available", "reward": "+200 EcoPoints"},
		{"id": "M5", "title": "Frontier Navigator: Run 15-Stage NSGA-II Discovery", "desc": "Complete an evolutionary multi-objective screening run yielding EVOCURE-001 candidate.", "status": "Available", "reward": "+400 EcoPoints"},
		{"id": "M6", "title": "Safety Guardian: Low Organ Toxicity in 7-Compartment PBPK", "desc": "Optimize candidate molecule so hERG, DILI, and Ames toxicity remain under 0.20.", "status": "Available", "reward": "+350 EcoPoints"},
		{"id": "M7", "title": "Bioisosteric Mastery: Fluorinate Aromatic Ring", "desc": "Use the RDKit Molecular Structure Editor to perform bioisosteric substitution with positive ΔQED.", "status": "Available", "reward": "+150 EcoPoints"},
		{"id": "M8", "title": "Scientific Publishing: Generate Native PDF Report", "desc": "Export a comprehensive peer-review ready PDF dossier from the AI Assistant workspace.", "status": "Available", "reward": "+500 EcoPoints"}
	]

	for m in missions:
		var card := PanelContainer.new()
		var is_comp: bool = (m["status"] == "Completed")
		var is_prog: bool = (m["status"] == "In Progress")
		var border_col: Color = ThemeBuilder.BORDER_EMERALD if is_comp else (ThemeBuilder.BORDER_CYAN if is_prog else ThemeBuilder.BORDER_SUBTLE)
		card.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, border_col, 8, 1, 14, 10))

		var chbox := HBoxContainer.new()
		chbox.add_theme_constant_override("separation", 12)
		card.add_child(chbox)

		var icon_lbl := Label.new()
		icon_lbl.text = "✓" if is_comp else ("⏳" if is_prog else "🔒")
		icon_lbl.add_theme_font_size_override("font_size", 16)
		icon_lbl.add_theme_color_override("font_color", border_col)
		chbox.add_child(icon_lbl)

		var text_vbox := VBoxContainer.new()
		text_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		chbox.add_child(text_vbox)

		var m_title := Label.new()
		m_title.text = m["title"]
		m_title.add_theme_color_override("font_color", ThemeBuilder.TEXT_PRIMARY)
		text_vbox.add_child(m_title)

		var m_desc := Label.new()
		m_desc.text = m["desc"]
		m_desc.add_theme_font_size_override("font_size", 11)
		m_desc.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
		text_vbox.add_child(m_desc)

		var reward_lbl := Label.new()
		reward_lbl.text = m["reward"]
		reward_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_AMBER)
		chbox.add_child(reward_lbl)

		list_vbox.add_child(card)


# ==============================================================================
# 4. SCIENTIFIC GLOSSARY & 10-STEP TUTORIAL MODAL
# ==============================================================================
func _build_glossary_modal() -> void:
	_glossary_modal = PanelContainer.new()
	_glossary_modal.visible = false
	_glossary_modal.custom_minimum_size = Vector2(800, 600)
	_glossary_modal.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_VIOLET, 12, 2, 22, 18, 16, Color(0.66, 0.33, 0.97, 0.35)))
	_modal_center.add_child(_glossary_modal)

	var root_vbox := VBoxContainer.new()
	root_vbox.add_theme_constant_override("separation", 12)
	_glossary_modal.add_child(root_vbox)

	var header_hbox := HBoxContainer.new()
	var title_lbl := Label.new()
	title_lbl.text = "SCIENTIFIC GLOSSARY & 10-STEP TUTORIAL"
	title_lbl.add_theme_font_size_override("font_size", 16)
	title_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_VIOLET)
	title_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header_hbox.add_child(title_lbl)

	var close_btn := Button.new()
	close_btn.text = "✕"
	close_btn.custom_minimum_size = Vector2(30, 30)
	close_btn.pressed.connect(close_all_modals)
	header_hbox.add_child(close_btn)
	root_vbox.add_child(header_hbox)

	# Top Section: 10-Step Interactive Tutorial Carousel
	_tutorial_card = PanelContainer.new()
	_tutorial_card.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(Color(0.08, 0.12, 0.22, 0.9), ThemeBuilder.BORDER_CYAN, 10, 1, 16, 12))
	root_vbox.add_child(_tutorial_card)

	var tut_vbox := VBoxContainer.new()
	tut_vbox.add_theme_constant_override("separation", 6)
	_tutorial_card.add_child(tut_vbox)

	var tut_nav := HBoxContainer.new()
	_tutorial_step_label = Label.new()
	_tutorial_step_label.text = "Step 1/10"
	_tutorial_step_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_CYAN)
	tut_nav.add_child(_tutorial_step_label)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	tut_nav.add_child(spacer)

	var prev_btn := Button.new()
	prev_btn.text = "◀ Previous"
	prev_btn.pressed.connect(_on_tutorial_prev)
	tut_nav.add_child(prev_btn)

	var next_btn := Button.new()
	next_btn.text = "Next ▶"
	next_btn.pressed.connect(_on_tutorial_next)
	tut_nav.add_child(next_btn)
	tut_vbox.add_child(tut_nav)

	_tutorial_title_label = Label.new()
	_tutorial_title_label.add_theme_font_size_override("font_size", 14)
	_tutorial_title_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_PRIMARY)
	tut_vbox.add_child(_tutorial_title_label)

	_tutorial_desc_label = Label.new()
	_tutorial_desc_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_tutorial_desc_label.add_theme_font_size_override("font_size", 12)
	_tutorial_desc_label.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	tut_vbox.add_child(_tutorial_desc_label)

	_update_tutorial_view()

	# Bottom Section: Searchable Glossary
	var gloss_hdr := Label.new()
	gloss_hdr.text = "Scientific & Mathematical Nomenclature:"
	gloss_hdr.add_theme_color_override("font_color", ThemeBuilder.TEXT_AMBER)
	root_vbox.add_child(gloss_hdr)

	_glossary_search = LineEdit.new()
	_glossary_search.placeholder_text = "Search nomenclature definitions..."
	_glossary_search.text_changed.connect(func(_t: String) -> void: _refresh_glossary())
	root_vbox.add_child(_glossary_search)

	var gloss_scroll := ScrollContainer.new()
	gloss_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root_vbox.add_child(gloss_scroll)

	_glossary_list = VBoxContainer.new()
	_glossary_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_glossary_list.add_theme_constant_override("separation", 8)
	gloss_scroll.add_child(_glossary_list)

	_refresh_glossary()


func _update_tutorial_view() -> void:
	if _tutorial_index < 0 or _tutorial_index >= TUTORIAL_STEPS.size():
		return
	var item: Dictionary = TUTORIAL_STEPS[_tutorial_index]
	_tutorial_step_label.text = str(item.get("step", ""))
	_tutorial_title_label.text = str(item.get("title", ""))
	_tutorial_desc_label.text = str(item.get("desc", ""))


func _on_tutorial_prev() -> void:
	if _tutorial_index > 0:
		_tutorial_index -= 1
		_update_tutorial_view()


func _on_tutorial_next() -> void:
	if _tutorial_index < TUTORIAL_STEPS.size() - 1:
		_tutorial_index += 1
		_update_tutorial_view()


func _refresh_glossary() -> void:
	for c in _glossary_list.get_children():
		c.queue_free()

	var q: String = _glossary_search.text.strip_edges().to_lower()
	for item in GLOSSARY_ITEMS:
		var term: String = str(item["term"])
		var cat: String = str(item["category"])
		var defn: String = str(item["definition"])

		if not q.is_empty() and not term.to_lower().contains(q) and not defn.to_lower().contains(q):
			continue

		var p := PanelContainer.new()
		p.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 6, 1, 10, 8))
		var vb := VBoxContainer.new()
		vb.add_theme_constant_override("separation", 3)
		p.add_child(vb)

		var t_lbl := Label.new()
		t_lbl.text = term + " [" + cat + "]"
		t_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
		vb.add_child(t_lbl)

		var d_lbl := Label.new()
		d_lbl.text = defn
		d_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		d_lbl.add_theme_font_size_override("font_size", 11)
		d_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
		vb.add_child(d_lbl)

		_glossary_list.add_child(p)


# ==============================================================================
# 5. NON-BLOCKING TOAST BANNER
# ==============================================================================
func _build_toast_banner() -> void:
	_toast_box = PanelContainer.new()
	_toast_box.visible = false
	_toast_box.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_toast_box.custom_minimum_size = Vector2(340, 50)
	_toast_box.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_toast_box.position = Vector2(-360, -80)
	_toast_box.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	_toast_box.grow_vertical = Control.GROW_DIRECTION_BEGIN
	add_child(_toast_box)

	_toast_label = Label.new()
	_toast_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_toast_label.add_theme_font_size_override("font_size", 12)
	_toast_box.add_child(_toast_label)

	_toast_timer = Timer.new()
	_toast_timer.one_shot = true
	_toast_timer.wait_time = 3.5
	_toast_timer.timeout.connect(func() -> void:
		_toast_box.visible = false
	)
	add_child(_toast_timer)


func show_toast(msg: String, severity: String = "info") -> void:
	var border_col: Color = ThemeBuilder.BORDER_CYAN
	match severity:
		"success":
			border_col = ThemeBuilder.BORDER_EMERALD
		"warning":
			border_col = ThemeBuilder.TEXT_AMBER
		"error":
			border_col = ThemeBuilder.BORDER_CRIMSON

	_toast_box.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, border_col, 8, 2, 14, 10, 8, border_col))
	_toast_label.text = msg
	_toast_label.add_theme_color_override("font_color", border_col)
	_toast_box.visible = true
	_toast_timer.start(3.5)


func _connect_signals() -> void:
	AppState.toast_requested.connect(show_toast)


# ==============================================================================
# PUBLIC MODAL OPEN / CLOSE APIS
# ==============================================================================
func open_ai_modal() -> void:
	close_all_modals()
	_bloom_toggle.set_pressed_no_signal(AppState.bloom_enabled)
	_colorblind_opt.select(maxi(0, ["bioluminescent", "okabe_ito", "high_contrast"].find(AppState.colorblind_palette)))
	_overlay_backdrop.visible = true
	_ai_modal.visible = true
	# Sync provider selection
	var active_prov: String = str(AppState.ai_providers_status.get("active_provider", "auto_local"))
	select_provider_for_preview(active_prov)


func select_provider_for_preview(prov_id: String) -> void:
	for i in range(_ai_provider_opt.item_count):
		if _ai_provider_opt.get_item_metadata(i) == prov_id:
			_ai_provider_opt.selected = i
			_populate_model_presets_for(prov_id)
			break


func open_library_modal() -> void:
	close_all_modals()
	_overlay_backdrop.visible = true
	_library_modal.visible = true
	_refresh_library_cards()


func open_missions_modal() -> void:
	close_all_modals()
	_overlay_backdrop.visible = true
	_missions_modal.visible = true


func open_glossary_modal() -> void:
	close_all_modals()
	_overlay_backdrop.visible = true
	_glossary_modal.visible = true
	_update_tutorial_view()


func close_all_modals() -> void:
	_overlay_backdrop.visible = false
	_ai_modal.visible = false
	_library_modal.visible = false
	_missions_modal.visible = false
	_glossary_modal.visible = false
	modal_closed.emit("all")
