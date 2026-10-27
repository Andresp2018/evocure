class_name MainDashboard
extends Control
## Central Bioluminescent Scientific Dashboard for EVOCURE Godot 4.7.
## Hosts the Top Navigation Bar, 6 Workspace Tabs (Chemical Universe, Evolution Lab,
## Cancer Battle, Family Tree & Editor, Toxicity & PBPK, AI Assistant & Reports),
## the Left/Right Sidebars, Microscopic World 2D viewport, Bottom Telemetry Charts,
## and ModalsController.

# Workspaces
var _tabs_container: Control
var _tab_universe: Control
var _tab_evolution_lab: TabEvolutionLab
var _tab_cancer_battle: TabCancerBattle
var _tab_family_editor: TabFamilyAndEditor
var _tab_toxicity_pbpk: TabToxicityPBPK
var _tab_ai_assistant: TabAIAssistant
var _tab_buttons: Array[Button] = []

# Modals Controller
var _modals_ctrl: ModalsController

# Top Navigation Bar elements
var _disease_opt: OptionButton
var _ai_badge_btn: Button
var _missions_btn: Button
var _settings_btn: Button
var _glossary_btn: Button
var _conn_status_lbl: Label

# Chemical Universe (Tab 0) - Left Sidebar
var _env_preset_buttons: Dictionary = {}
var _mode_lab_btn: Button
var _mode_play_btn: Button

# Chemical Universe (Tab 0) - Center Area
var _world_host: Control
var _world_view: Control
var _zoom_in_btn: Button
var _zoom_out_btn: Button
var _zoom_reset_btn: Button
var _overlay_opt: OptionButton

# Chemical Universe (Tab 0) - Right Sidebar (Entity Details)
var _portrait_host: Control
var _portrait_instance: Control
var _entity_name_lbl: Label
var _entity_family_lbl: Label
var _entity_gen_lbl: Label
var _entity_type_lbl: Label
var _entity_status_dot: Label

var _subtab_attr_btn: Button
var _subtab_lineage_btn: Button
var _subtab_compound_btn: Button
var _attr_view: VBoxContainer
var _lineage_view: VBoxContainer
var _compound_view: VBoxContainer

var _bar_toxicity: ProgressBar
var _val_toxicity: Label
var _bar_antitumor: ProgressBar
var _val_antitumor: Label
var _bar_energy: ProgressBar
var _val_energy: Label
var _bar_mutation: ProgressBar
var _val_mutation: Label
var _epistemic_tag_lbl: Label

var _lineage_parent_lbl: Label
var _lineage_gen_lbl: Label
var _lineage_smiles_lbl: Label
var _jump_family_tree_btn: Button

var _compound_texture: TextureRect
var _compound_formula_lbl: Label
var _compound_mw_lbl: Label
var _compound_logp_lbl: Label
var _compound_qed_lbl: Label
var _edit_struct_btn: Button
var _dose_tumor_btn: Button

var _strip_portraits: Array[Control] = []

# Chemical Universe (Tab 0) - Bottom Dock
var _chart_pop_host: Control
var _chart_pop_inst: Control
var _chart_tox_host: Control
var _chart_tox_inst: Control
var _chart_anti_host: Control
var _chart_anti_inst: Control

# Bottom Timeline & Fast Forward Bar
var _play_pause_btn: Button
var _speed_1x_btn: Button
var _speed_10x_btn: Button
var _speed_100x_btn: Button
var _sim_time_lbl: Label
var _sim_progress: ProgressBar
var _gen_stepper_lbl: Label
var _adv_100_btn: Button
var _adv_1000_btn: Button

var _chart_update_timer: float = 0.0
var _tour_out_dir: String = ""
var _tour_elapsed: float = 0.0
var _tour_stage: int = -1


func _ready() -> void:
	ThemeBuilder.fill_rect(self)
	theme = ThemeBuilder.build_main_theme()

	# Base full-screen background
	var bg := ColorRect.new()
	ThemeBuilder.fill_rect(bg)
	bg.color = ThemeBuilder.BG_ABYSS
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(bg)

	_build_ui_layout()
	_instantiate_modals()
	_connect_app_signals()
	_switch_tab(0)
	_check_cli_screenshot_tour()
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--desktop-audit="):
			var audit: Node = load("res://scripts/testing/DesktopAudit.gd").new()
			add_child(audit)
			audit.call_deferred("run", self, arg.trim_prefix("--desktop-audit="))


func _build_ui_layout() -> void:
	var main_vbox := VBoxContainer.new()
	ThemeBuilder.fill_rect(main_vbox)
	main_vbox.add_theme_constant_override("separation", 0)
	add_child(main_vbox)

	_build_top_nav_bar(main_vbox)
	var reset_bar := HBoxContainer.new()
	main_vbox.add_child(reset_bar)
	var scope_note := Label.new()
	scope_note.text = "Local exploratory models · Chemical generations and tissue biological time are separate"
	scope_note.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	reset_bar.add_child(scope_note)
	var compute_label := Label.new()
	compute_label.text = "Checking GPU..."
	compute_label.add_theme_font_size_override("font_size", 11)
	reset_bar.add_child(compute_label)
	ApiClient.request_json(HTTPClient.METHOD_GET, "/api/v1/compute/status", {}, func(code: int, data: Variant) -> void:
		if code == 200 and data is Dictionary:
			compute_label.text = str(data.get("gpu_name", "CPU")) + " · " + str(data.get("numerical_backend", ""))
			compute_label.tooltip_text = "GPU: Vulkan rendering, tissue substrate fields, local AI when loaded.\nCPU: RDKit chemistry, cellular agents, QSAR, PBPK and experiment storage."
		else:
			compute_label.text = "GPU status unavailable"
	)
	var reset_btn := Button.new()
	reset_btn.text = "Reset defaults"
	reset_bar.add_child(reset_btn)
	var confirm_reset := ConfirmationDialog.new()
	confirm_reset.title = "Reset to defaults"
	confirm_reset.dialog_text = "Restore the starting chemical world, tissue, environment and playback settings? Unsaved changes will be replaced. Saved experiments are kept."
	add_child(confirm_reset)
	reset_btn.pressed.connect(func() -> void: confirm_reset.popup_centered(Vector2i(600, 160)))
	confirm_reset.confirmed.connect(func() -> void:
		ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/world/reset-defaults", {}, func(code: int, data: Variant) -> void:
			if code == 200 and data is Dictionary:
				AppState.selected_entity = {}
				AppState.tissue_compound_query = "Gemcitabine"
				AppState.analysis_compound_query = ""
				AppState.bloom_enabled = true
				AppState.colorblind_palette = "bioluminescent"
				AppState.heatmap_overlay = "none"
				AppState.active_spawn_compound = ""
				if _world_view.has_method("reset_camera"):
					_world_view.call("reset_camera")
				if _world_view.has_method("set_heatmap_overlay"):
					_world_view.call("set_heatmap_overlay", "none")
				_tab_cancer_battle._view.zoom = 1
				_tab_cancer_battle._view.overlay = "oxygen"
				_tab_cancer_battle._overlay.select(0)
				AppState.active_environment_preset = "pancreatic_tissue"
				AppState.ingest_server_state(data.get("data", {}))
				_switch_tab(0)
				AppState.notify_toast("Starting world restored. Press Play to begin.", "success")
			else:
				AppState.notify_toast("Reset failed: " + str(data), "error")
		)
	)

	# Main container hosting the 6 workspaces
	_tabs_container = Control.new()
	_tabs_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_tabs_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	main_vbox.add_child(_tabs_container)

	_build_tab_0_chemical_universe()
	_build_other_tabs()


# ==============================================================================
# TOP NAVIGATION BAR (52px HEIGHT)
# ==============================================================================
func _build_top_nav_bar(parent: VBoxContainer) -> void:
	var nav_panel := PanelContainer.new()
	nav_panel.custom_minimum_size = Vector2(0, 52)
	nav_panel.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_SUBTLE, 0, 1, 16, 8))
	parent.add_child(nav_panel)

	var nav_scroll := ScrollContainer.new()
	nav_scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	nav_scroll.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	nav_panel.add_child(nav_scroll)
	var nav_hbox := HBoxContainer.new()
	nav_hbox.add_theme_constant_override("separation", 10)
	nav_scroll.add_child(nav_hbox)

	# Brand Badge
	var brand_box := HBoxContainer.new()
	brand_box.add_theme_constant_override("separation", 6)
	nav_hbox.add_child(brand_box)

	var brand_lbl := Label.new()
	brand_lbl.text = "EVOCURE"
	brand_lbl.add_theme_font_size_override("font_size", 18)
	brand_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
	brand_box.add_child(brand_lbl)

	var sub_lbl := Label.new()
	sub_lbl.text = "LAB v2.0"
	sub_lbl.add_theme_font_size_override("font_size", 10)
	sub_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	brand_box.add_child(sub_lbl)

	var sep1 := VSeparator.new()
	nav_hbox.add_child(sep1)

	# 6 Primary Workspace Tabs
	var tab_names := [
		"Chemical Universe",
		"Evolution Lab",
		"Disease & Tissue",
		"Family Tree & Editor",
		"Toxicity & PBPK",
		"Chat & Export"
	]

	for i in range(tab_names.size()):
		var t_btn := Button.new()
		t_btn.text = tab_names[i]
		t_btn.focus_mode = Control.FOCUS_NONE
		t_btn.pressed.connect(func() -> void: _switch_tab(i))
		nav_hbox.add_child(t_btn)
		_tab_buttons.append(t_btn)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	nav_hbox.add_child(spacer)

	# Connection status pill
	_conn_status_lbl = Label.new()
	_conn_status_lbl.text = "Connecting to engine..."
	_conn_status_lbl.add_theme_font_size_override("font_size", 11)
	_conn_status_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD)
	nav_hbox.add_child(_conn_status_lbl)

	# Disease Selector OptionButton
	var dis_lbl := Label.new()
	dis_lbl.text = "Disease:"
	dis_lbl.add_theme_font_size_override("font_size", 11)
	dis_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	nav_hbox.add_child(dis_lbl)

	_disease_opt = OptionButton.new()
	_disease_opt.add_item("PDAC (Pancreatic)", 0)
	_disease_opt.set_item_metadata(0, "PDAC")
	_disease_opt.add_item("NSCLC_KRAS (Lung)", 1)
	_disease_opt.set_item_metadata(1, "NSCLC_KRAS")
	_disease_opt.add_item("AMR_BACTERIAL (Biofilm)", 2)
	_disease_opt.set_item_metadata(2, "AMR_BACTERIAL")
	_disease_opt.add_item("METABOLIC_MASLD (Liver)", 3)
	_disease_opt.set_item_metadata(3, "METABOLIC_MASLD")
	_disease_opt.item_selected.connect(_on_disease_selected)
	nav_hbox.add_child(_disease_opt)

	# Active AI Provider Badge Button
	_ai_badge_btn = Button.new()
	_ai_badge_btn.text = "AI Settings"
	_ai_badge_btn.add_theme_font_size_override("font_size", 11)
	ThemeBuilder.style_accent_button(_ai_badge_btn, ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_CYAN, Color(0.13, 0.82, 0.93, 0.3))
	_ai_badge_btn.pressed.connect(func() -> void:
		if _modals_ctrl != null:
			_modals_ctrl.open_ai_modal()
	)
	nav_hbox.add_child(_ai_badge_btn)

	# Missions button
	_missions_btn = Button.new()
	_missions_btn.text = "Missions"
	_missions_btn.pressed.connect(func() -> void:
		if _modals_ctrl != null:
			_modals_ctrl.open_missions_modal()
	)
	nav_hbox.add_child(_missions_btn)

	# Settings button
	_settings_btn = Button.new()
	_settings_btn.text = "⚙"
	_settings_btn.pressed.connect(func() -> void:
		if _modals_ctrl != null:
			_modals_ctrl.open_ai_modal()
	)
	nav_hbox.add_child(_settings_btn)

	# Help / Glossary button
	_glossary_btn = Button.new()
	_glossary_btn.text = "?"
	_glossary_btn.pressed.connect(func() -> void:
		if _modals_ctrl != null:
			_modals_ctrl.open_glossary_modal()
	)
	nav_hbox.add_child(_glossary_btn)


# ==============================================================================
# TAB 0: CHEMICAL UNIVERSE (FUTURISTIC LABORATORY HUD)
# ==============================================================================
func _build_tab_0_chemical_universe() -> void:
	_tab_universe = Control.new()
	ThemeBuilder.fill_rect(_tab_universe)
	_tabs_container.add_child(_tab_universe)

	var u_vbox := VBoxContainer.new()
	ThemeBuilder.fill_rect(u_vbox)
	u_vbox.add_theme_constant_override("separation", 4)
	_tab_universe.add_child(u_vbox)

	# Middle Area: Left Sidebar (240px) | Center World View | Right Sidebar (285px)
	var middle_hbox := HBoxContainer.new()
	middle_hbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	middle_hbox.add_theme_constant_override("separation", 6)
	u_vbox.add_child(middle_hbox)

	_build_left_sidebar(middle_hbox)
	_build_center_world_area(middle_hbox)
	_build_right_sidebar(middle_hbox)

	# Bottom Dock (215px height): 3 Realtime Charts (155px) + Timeline Bar (52px)
	_build_bottom_dock(u_vbox)


# --- 1. LEFT SIDEBAR (240px) ---
func _build_left_sidebar(parent: HBoxContainer) -> void:
	var left_panel := PanelContainer.new()
	left_panel.custom_minimum_size = Vector2(240, 0)
	left_panel.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 10, 8))
	parent.add_child(left_panel)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	left_panel.add_child(scroll)

	var left_vbox := VBoxContainer.new()
	left_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	left_vbox.add_theme_constant_override("separation", 8)
	scroll.add_child(left_vbox)

	# Section 1: Add Compound
	var add_lbl := Label.new()
	add_lbl.text = "ADD COMPOUND"
	add_lbl.add_theme_font_size_override("font_size", 12)
	add_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
	left_vbox.add_child(add_lbl)

	var compounds := [
		{"name": "Glucose", "role": "autotroph", "kind": "PROTOCELL", "icon": "res://assets/sprites/creature_glucose.png"},
		{"name": "Glycine", "role": "autotroph", "kind": "PROTOCELL", "icon": "res://assets/sprites/creature_amino.png"},
		{"name": "Palmitic Acid", "role": "structural", "kind": "PROTOCELL", "icon": "res://assets/sprites/creature_lipid.png"},
		{"name": "Adenine", "role": "cooperator", "kind": "PROTOCELL", "icon": "res://assets/sprites/creature_benzene.png"},
		{"name": "Imidazole", "role": "catalyst", "kind": "CATALYST", "icon": "res://assets/sprites/creature_phosphate.png"},
	]

	for c in compounds:
		var c_btn := Button.new()
		c_btn.text = " " + c["name"]
		c_btn.custom_minimum_size = Vector2(0, 34)
		c_btn.alignment = HorizontalAlignment.HORIZONTAL_ALIGNMENT_LEFT
		if ResourceLoader.exists(c["icon"]):
			var tex: Texture2D = load(c["icon"])
			c_btn.icon = tex
			c_btn.expand_icon = true
			c_btn.add_theme_constant_override("icon_max_width", 26)
		c_btn.pressed.connect(func() -> void:
			if has_node("/root/ApiClient"):
				get_node("/root/ApiClient").spawn_compound(c["name"], 10, null, null, c["role"], c["kind"])
		)
		left_vbox.add_child(c_btn)

	var lib_btn := Button.new()
	lib_btn.text = "+ Custom / Full Library (45)"
	ThemeBuilder.style_accent_button(lib_btn, ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_CYAN, Color(0.13, 0.82, 0.93, 0.3))
	lib_btn.pressed.connect(func() -> void:
		if _modals_ctrl != null:
			_modals_ctrl.open_library_modal()
	)
	left_vbox.add_child(lib_btn)
	var stir_btn := Button.new()
	stir_btn.text = "Stir / mix populations"
	stir_btn.tooltip_text = "Apply mechanical agitation to the actual particles."
	stir_btn.pressed.connect(func() -> void:
		ApiClient.request_json(HTTPClient.METHOD_POST, "/api/v1/world/stir", {}, func(code: int, _res: Variant) -> void:
			if code == 200:
				ApiClient.set_playback("play", AppState.playback_speed)
		)
	)
	left_vbox.add_child(stir_btn)

	var sep1 := HSeparator.new()
	left_vbox.add_child(sep1)

	# Section 2: Environment Presets
	var env_lbl := Label.new()
	env_lbl.text = "ENVIRONMENT PRESET"
	env_lbl.add_theme_font_size_override("font_size", 12)
	env_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_EMERALD)
	left_vbox.add_child(env_lbl)

	var presets := [
		{"id": "pancreatic_tissue", "name": "Pancreatic Tissue"},
		{"id": "bloodstream", "name": "Bloodstream"},
		{"id": "tumor_microenvironment", "name": "Tumor Microenvironment"},
		{"id": "oxidative_stress", "name": "Oxidative Stress"},
		{"id": "acidic_low_ph", "name": "Acidic (Low pH)"},
		{"id": "hypoxia_low_o2", "name": "Hypoxia (Low O2)"}
	]

	for p in presets:
		var p_btn := Button.new()
		p_btn.text = p["name"]
		p_btn.alignment = HorizontalAlignment.HORIZONTAL_ALIGNMENT_LEFT
		p_btn.pressed.connect(func() -> void:
			_on_env_preset_clicked(p["id"])
		)
		left_vbox.add_child(p_btn)
		_env_preset_buttons[p["id"]] = p_btn

	_update_env_preset_highlight()

	var sep2 := HSeparator.new()
	left_vbox.add_child(sep2)

	# Section 3: Simulation Mode
	var mode_lbl := Label.new()
	mode_lbl.text = "SIMULATION MODE"
	mode_lbl.add_theme_font_size_override("font_size", 12)
	mode_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_AMBER)
	left_vbox.add_child(mode_lbl)

	_mode_lab_btn = Button.new()
	_mode_lab_btn.text = "Lab view\nMolecular structures"
	_mode_lab_btn.pressed.connect(func() -> void: _set_interface_mode("LAB"))
	left_vbox.add_child(_mode_lab_btn)

	_mode_play_btn = Button.new()
	_mode_play_btn.text = "Play view\nIllustrated ecosystem"
	_mode_play_btn.pressed.connect(func() -> void: _set_interface_mode("PLAY"))
	left_vbox.add_child(_mode_play_btn)

	_update_interface_mode_styles()


# --- 2. CENTER AREA: MicroscopicWorld2D ---
func _build_center_world_area(parent: HBoxContainer) -> void:
	var center_panel := PanelContainer.new()
	center_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	center_panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	center_panel.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_SUBTLE, 8, 1, 0, 0))
	parent.add_child(center_panel)

	_world_host = Control.new()
	ThemeBuilder.fill_rect(_world_host)
	_world_host.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_world_host.size_flags_vertical = Control.SIZE_EXPAND_FILL
	center_panel.add_child(_world_host)

	# Runtime load of MicroscopicWorld2D
	_world_view = _safe_instantiate_control("res://scripts/world/MicroscopicWorld2D.gd")
	ThemeBuilder.fill_rect(_world_view)
	_world_host.add_child(_world_view)

	# Overlay Selector at Top-Right of Center View (positioned cleanly to the left of the Tumor Badge)
	var overlay_bar := HBoxContainer.new()
	overlay_bar.anchor_left = 1.0
	overlay_bar.anchor_top = 0.0
	overlay_bar.anchor_right = 1.0
	overlay_bar.anchor_bottom = 0.0
	overlay_bar.offset_left = -435.0
	overlay_bar.offset_top = 15.0
	overlay_bar.offset_right = -280.0
	overlay_bar.offset_bottom = 45.0
	overlay_bar.alignment = BoxContainer.ALIGNMENT_END
	overlay_bar.add_theme_constant_override("separation", 6)
	_world_host.add_child(overlay_bar)

	_overlay_opt = OptionButton.new()
	_overlay_opt.add_item("Overlay: None", 0)
	_overlay_opt.add_item("Substrate", 1)
	_overlay_opt.add_item("Oxygen", 2)
	_overlay_opt.add_item("ECM / Biofilm", 3)
	_overlay_opt.add_item("Free Drug", 4)
	_overlay_opt.item_selected.connect(_on_overlay_selected)
	overlay_bar.add_child(_overlay_opt)


# --- 3. RIGHT SIDEBAR (306px) - ENTITY DETAILS ---
func _build_right_sidebar(parent: HBoxContainer) -> void:
	var right_panel := PanelContainer.new()
	right_panel.custom_minimum_size = Vector2(306, 0)
	right_panel.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 10, 8))
	parent.add_child(right_panel)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	right_panel.add_child(scroll)

	var right_vbox := VBoxContainer.new()
	right_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right_vbox.add_theme_constant_override("separation", 8)
	scroll.add_child(right_vbox)

	var title_lbl := Label.new()
	title_lbl.text = "ENTITY DETAILS"
	title_lbl.add_theme_font_size_override("font_size", 12)
	title_lbl.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
	right_vbox.add_child(title_lbl)

	# Top Entity Card: Portrait (96x96) on left + Info on right
	var top_card := PanelContainer.new()
	top_card.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_SUBTLE, 6, 1, 6, 6))
	right_vbox.add_child(top_card)

	var top_hbox := HBoxContainer.new()
	top_hbox.add_theme_constant_override("separation", 8)
	top_card.add_child(top_hbox)

	_portrait_host = Control.new()
	_portrait_host.custom_minimum_size = Vector2(96, 96)
	top_hbox.add_child(_portrait_host)

	# Runtime load CreaturePortraitControl
	_portrait_instance = _safe_instantiate_control("res://scripts/world/CreaturePortraitControl.gd")
	ThemeBuilder.fill_rect(_portrait_instance)
	_portrait_host.add_child(_portrait_instance)

	var info_vbox := VBoxContainer.new()
	info_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	info_vbox.add_theme_constant_override("separation", 2)
	top_hbox.add_child(info_vbox)

	_entity_name_lbl = Label.new()
	_entity_name_lbl.text = "C-4173 (Cytidine Lead)"
	_entity_name_lbl.add_theme_font_size_override("font_size", 12)
	_entity_name_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_CYAN)
	info_vbox.add_child(_entity_name_lbl)

	_entity_family_lbl = Label.new()
	_entity_family_lbl.text = "Family: Antitumor"
	_entity_family_lbl.add_theme_font_size_override("font_size", 10)
	_entity_family_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	info_vbox.add_child(_entity_family_lbl)

	_entity_gen_lbl = Label.new()
	_entity_gen_lbl.text = "Generation: awaiting state"
	_entity_gen_lbl.add_theme_font_size_override("font_size", 10)
	_entity_gen_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	info_vbox.add_child(_entity_gen_lbl)

	_entity_type_lbl = Label.new()
	_entity_type_lbl.text = "Type: awaiting state"
	_entity_type_lbl.add_theme_font_size_override("font_size", 10)
	info_vbox.add_child(_entity_type_lbl)

	_entity_status_dot = Label.new()
	_entity_status_dot.text = "● Status: Active"
	_entity_status_dot.add_theme_font_size_override("font_size", 10)
	_entity_status_dot.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD)
	info_vbox.add_child(_entity_status_dot)

	# 3 Sub-tab buttons
	var subtab_hbox := HBoxContainer.new()
	subtab_hbox.add_theme_constant_override("separation", 4)
	right_vbox.add_child(subtab_hbox)

	_subtab_attr_btn = Button.new()
	_subtab_attr_btn.text = "Attributes"
	_subtab_attr_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_subtab_attr_btn.pressed.connect(func() -> void: _set_entity_subtab(0))
	subtab_hbox.add_child(_subtab_attr_btn)

	_subtab_lineage_btn = Button.new()
	_subtab_lineage_btn.text = "Lineage"
	_subtab_lineage_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_subtab_lineage_btn.pressed.connect(func() -> void: _set_entity_subtab(1))
	subtab_hbox.add_child(_subtab_lineage_btn)

	_subtab_compound_btn = Button.new()
	_subtab_compound_btn.text = "Compound"
	_subtab_compound_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_subtab_compound_btn.pressed.connect(func() -> void: _set_entity_subtab(2))
	subtab_hbox.add_child(_subtab_compound_btn)

	# View 1: Attributes
	_attr_view = VBoxContainer.new()
	_attr_view.add_theme_constant_override("separation", 6)
	right_vbox.add_child(_attr_view)

	_bar_toxicity = _create_stat_bar(_attr_view, "Toxicity", ThemeBuilder.BORDER_CRIMSON, 0.18)
	_bar_antitumor = _create_stat_bar(_attr_view, "Antitumor Potential", ThemeBuilder.BORDER_EMERALD, 0.88)
	_bar_energy = _create_stat_bar(_attr_view, "Energy Reserve", ThemeBuilder.BORDER_CYAN, 0.65)
	_bar_mutation = _create_stat_bar(_attr_view, "Mutation Rate", ThemeBuilder.BORDER_VIOLET, 0.25)

	_epistemic_tag_lbl = Label.new()
	_epistemic_tag_lbl.text = "Epistemic: Mechanistic simulation result"
	_epistemic_tag_lbl.add_theme_font_size_override("font_size", 10)
	_epistemic_tag_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_AMBER)
	_attr_view.add_child(_epistemic_tag_lbl)

	# View 2: Lineage
	_lineage_view = VBoxContainer.new()
	_lineage_view.visible = false
	_lineage_view.add_theme_constant_override("separation", 4)
	right_vbox.add_child(_lineage_view)

	_lineage_parent_lbl = Label.new()
	_lineage_parent_lbl.text = "Parent: founder / see family tree"
	_lineage_parent_lbl.add_theme_font_size_override("font_size", 11)
	_lineage_view.add_child(_lineage_parent_lbl)

	_lineage_gen_lbl = Label.new()
	_lineage_gen_lbl.text = "Origin: select a molecule"
	_lineage_gen_lbl.add_theme_font_size_override("font_size", 11)
	_lineage_gen_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	_lineage_view.add_child(_lineage_gen_lbl)

	_lineage_smiles_lbl = Label.new()
	_lineage_smiles_lbl.text = "Scaffold: C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)O"
	_lineage_smiles_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_lineage_smiles_lbl.add_theme_font_size_override("font_size", 10)
	_lineage_smiles_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	_lineage_view.add_child(_lineage_smiles_lbl)

	_jump_family_tree_btn = Button.new()
	_jump_family_tree_btn.text = "🔍 View in Family Tree DAG"
	_jump_family_tree_btn.pressed.connect(func() -> void: _switch_tab(3))
	_lineage_view.add_child(_jump_family_tree_btn)

	# View 3: Compound
	_compound_view = VBoxContainer.new()
	_compound_view.visible = false
	_compound_view.add_theme_constant_override("separation", 4)
	right_vbox.add_child(_compound_view)

	_compound_texture = TextureRect.new()
	_compound_texture.custom_minimum_size = Vector2(260, 90)
	_compound_texture.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_compound_texture.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_compound_view.add_child(_compound_texture)

	_compound_formula_lbl = Label.new()
	_compound_formula_lbl.text = "Formula: C9H11F2N3O4"
	_compound_formula_lbl.add_theme_font_size_override("font_size", 11)
	_compound_view.add_child(_compound_formula_lbl)

	_compound_mw_lbl = Label.new()
	_compound_mw_lbl.text = "MW: 263.20 Da"
	_compound_mw_lbl.add_theme_font_size_override("font_size", 11)
	_compound_view.add_child(_compound_mw_lbl)

	_compound_logp_lbl = Label.new()
	_compound_logp_lbl.text = "LogP: -1.40 | QED: 0.68"
	_compound_logp_lbl.add_theme_font_size_override("font_size", 11)
	_compound_logp_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD)
	_compound_view.add_child(_compound_logp_lbl)

	var comp_btns_hb := HBoxContainer.new()
	comp_btns_hb.add_theme_constant_override("separation", 6)
	_compound_view.add_child(comp_btns_hb)

	_edit_struct_btn = Button.new()
	_edit_struct_btn.text = "Edit Structure"
	_edit_struct_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_edit_struct_btn.pressed.connect(func() -> void:
		AppState.analysis_compound_query = str(AppState.selected_entity.get("compound_id", ""))
		_switch_tab(3)
	)
	comp_btns_hb.add_child(_edit_struct_btn)

	_dose_tumor_btn = Button.new()
	_dose_tumor_btn.text = "View in tissue"
	_dose_tumor_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_dose_tumor_btn.pressed.connect(func() -> void:
		AppState.tissue_compound_query = str(AppState.selected_entity.get("compound_id", "Gemcitabine"))
		_switch_tab(2)
	)
	comp_btns_hb.add_child(_dose_tumor_btn)

	var sep3 := HSeparator.new()
	right_vbox.add_child(sep3)

	# Bottom Lineage Strip (G0 -> G1 -> G2 -> G3 -> G4)
	var strip_lbl := Label.new()
	strip_lbl.text = "Generational Evolution Strip:"
	strip_lbl.add_theme_font_size_override("font_size", 11)
	strip_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_MUTED)
	right_vbox.add_child(strip_lbl)

	var strip_hbox := HBoxContainer.new()
	strip_hbox.add_theme_constant_override("separation", 6)
	right_vbox.add_child(strip_hbox)

	strip_lbl.text = "Select a molecule to inspect its real lineage."

	_set_entity_subtab(0)


func _create_stat_bar(parent: VBoxContainer, title: String, col: Color, val: float) -> ProgressBar:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	parent.add_child(row)

	var l := Label.new()
	l.text = title
	l.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	l.add_theme_font_size_override("font_size", 11)
	row.add_child(l)

	var vl := Label.new()
	vl.text = "%0.2f" % val
	vl.add_theme_font_size_override("font_size", 11)
	vl.add_theme_color_override("font_color", col)
	row.add_child(vl)

	var pb := ProgressBar.new()
	pb.custom_minimum_size = Vector2(0, 8)
	pb.max_value = 1.0
	pb.value = val
	pb.show_percentage = false
	pb.set_meta("value_label", vl)
	var fill := ThemeBuilder.make_glass_box(col, col, 4, 0, 0, 0)
	pb.add_theme_stylebox_override("fill", fill)
	parent.add_child(pb)

	return pb


func _set_entity_subtab(idx: int) -> void:
	_attr_view.visible = (idx == 0)
	_lineage_view.visible = (idx == 1)
	_compound_view.visible = (idx == 2)
	_subtab_attr_btn.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN if idx == 0 else ThemeBuilder.TEXT_MUTED)
	_subtab_lineage_btn.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN if idx == 1 else ThemeBuilder.TEXT_MUTED)
	_subtab_compound_btn.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN if idx == 2 else ThemeBuilder.TEXT_MUTED)


# --- 4. BOTTOM DOCK (215px) ---
func _build_bottom_dock(parent: VBoxContainer) -> void:
	var dock_panel := PanelContainer.new()
	dock_panel.custom_minimum_size = Vector2(0, 215)
	dock_panel.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_PANEL, ThemeBuilder.BORDER_SUBTLE, 8, 1, 10, 6))
	parent.add_child(dock_panel)

	var dock_vbox := VBoxContainer.new()
	dock_vbox.add_theme_constant_override("separation", 4)
	dock_panel.add_child(dock_vbox)

	# Row 1 (155px height): 3 Realtime Line Charts side by side
	var charts_hbox := HBoxContainer.new()
	charts_hbox.custom_minimum_size = Vector2(0, 155)
	charts_hbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	charts_hbox.add_theme_constant_override("separation", 8)
	dock_vbox.add_child(charts_hbox)

	# Chart 1: Family Population
	_chart_pop_host = Control.new()
	_chart_pop_host.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	charts_hbox.add_child(_chart_pop_host)
	_chart_pop_inst = _safe_instantiate_control("res://scripts/charts/RealtimeLineChart.gd")
	ThemeBuilder.fill_rect(_chart_pop_inst)
	_chart_pop_host.add_child(_chart_pop_inst)
	if _chart_pop_inst.has_method("configure"):
		_chart_pop_inst.call("configure", "Ecological role populations", true, true)

	# Chart 2: Mean Toxicity
	_chart_tox_host = Control.new()
	_chart_tox_host.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	charts_hbox.add_child(_chart_tox_host)
	_chart_tox_inst = _safe_instantiate_control("res://scripts/charts/RealtimeLineChart.gd")
	ThemeBuilder.fill_rect(_chart_tox_inst)
	_chart_tox_host.add_child(_chart_tox_inst)
	if _chart_tox_inst.has_method("configure"):
		_chart_tox_inst.call("configure", "Mean energy reserve (0-1)", false, false)

	# Chart 3: Mean Antitumor Potential
	_chart_anti_host = Control.new()
	_chart_anti_host.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	charts_hbox.add_child(_chart_anti_host)
	_chart_anti_inst = _safe_instantiate_control("res://scripts/charts/RealtimeLineChart.gd")
	ThemeBuilder.fill_rect(_chart_anti_inst)
	_chart_anti_host.add_child(_chart_anti_inst)
	if _chart_anti_inst.has_method("configure"):
		_chart_anti_inst.call("configure", "Species diversity (unique / population)", false, false)

	# Row 2 (52px height): Timeline & Fast-Forward Bar
	var timeline_panel := PanelContainer.new()
	timeline_panel.custom_minimum_size = Vector2(0, 52)
	timeline_panel.add_theme_stylebox_override("panel", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 6, 1, 10, 6))
	dock_vbox.add_child(timeline_panel)

	var tl_hbox := HBoxContainer.new()
	tl_hbox.add_theme_constant_override("separation", 10)
	timeline_panel.add_child(tl_hbox)

	_play_pause_btn = Button.new()
	_play_pause_btn.text = "▶ Play"
	_play_pause_btn.pressed.connect(_on_play_pause_clicked)
	tl_hbox.add_child(_play_pause_btn)
	var step_btn := Button.new()
	step_btn.text = "Step"
	step_btn.pressed.connect(func() -> void: ApiClient.set_playback("step", AppState.playback_speed))
	tl_hbox.add_child(step_btn)

	_speed_1x_btn = Button.new()
	_speed_1x_btn.text = "1x"
	_speed_1x_btn.pressed.connect(func() -> void: _set_speed(1))
	tl_hbox.add_child(_speed_1x_btn)

	_speed_10x_btn = Button.new()
	_speed_10x_btn.text = "10x"
	_speed_10x_btn.pressed.connect(func() -> void: _set_speed(10))
	tl_hbox.add_child(_speed_10x_btn)

	_speed_100x_btn = Button.new()
	_speed_100x_btn.text = "Max"
	_speed_100x_btn.pressed.connect(func() -> void: _set_speed(100))
	tl_hbox.add_child(_speed_100x_btn)

	_sim_time_lbl = Label.new()
	_sim_time_lbl.text = "Time: 0 virtual years"
	_sim_time_lbl.add_theme_font_size_override("font_size", 11)
	tl_hbox.add_child(_sim_time_lbl)

	_sim_progress = ProgressBar.new()
	_sim_progress.custom_minimum_size = Vector2(160, 10)
	_sim_progress.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sim_progress.max_value = 100.0
	_sim_progress.value = 42.5
	tl_hbox.add_child(_sim_progress)

	_gen_stepper_lbl = Label.new()
	_gen_stepper_lbl.text = "Gen: 0"
	_gen_stepper_lbl.add_theme_font_size_override("font_size", 11)
	_gen_stepper_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_CYAN)
	tl_hbox.add_child(_gen_stepper_lbl)

	_adv_100_btn = Button.new()
	_adv_100_btn.text = "Run 100 gen"
	ThemeBuilder.style_accent_button(_adv_100_btn, ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_CYAN, Color(0.13, 0.82, 0.93, 0.35))
	_adv_100_btn.pressed.connect(func() -> void:
		if has_node("/root/ApiClient"):
			get_node("/root/ApiClient").advance_time_preset("100_gen")
	)
	tl_hbox.add_child(_adv_100_btn)

	_adv_1000_btn = Button.new()
	_adv_1000_btn.text = "Run 1,000 gen"
	ThemeBuilder.style_accent_button(_adv_1000_btn, ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_VIOLET, Color(0.66, 0.33, 0.97, 0.35))
	_adv_1000_btn.pressed.connect(func() -> void:
		if has_node("/root/ApiClient"):
			get_node("/root/ApiClient").advance_time_preset("1000_gen")
	)
	tl_hbox.add_child(_adv_1000_btn)


# ==============================================================================
# OTHER WORKSPACES (TABS 1 TO 5)
# ==============================================================================
func _build_other_tabs() -> void:
	_tab_evolution_lab = TabEvolutionLab.new()
	ThemeBuilder.fill_rect(_tab_evolution_lab)
	_tab_evolution_lab.visible = false
	_tabs_container.add_child(_tab_evolution_lab)

	_tab_cancer_battle = TabCancerBattle.new()
	ThemeBuilder.fill_rect(_tab_cancer_battle)
	_tab_cancer_battle.visible = false
	_tabs_container.add_child(_tab_cancer_battle)

	_tab_family_editor = TabFamilyAndEditor.new()
	ThemeBuilder.fill_rect(_tab_family_editor)
	_tab_family_editor.visible = false
	_tabs_container.add_child(_tab_family_editor)

	_tab_toxicity_pbpk = TabToxicityPBPK.new()
	ThemeBuilder.fill_rect(_tab_toxicity_pbpk)
	_tab_toxicity_pbpk.visible = false
	_tabs_container.add_child(_tab_toxicity_pbpk)

	_tab_ai_assistant = TabAIAssistant.new()
	ThemeBuilder.fill_rect(_tab_ai_assistant)
	_tab_ai_assistant.visible = false
	_tab_ai_assistant.open_ai_settings_requested.connect(func() -> void:
		if _modals_ctrl != null:
			_modals_ctrl.open_ai_modal()
	)
	_tabs_container.add_child(_tab_ai_assistant)


func _instantiate_modals() -> void:
	_modals_ctrl = ModalsController.new()
	ThemeBuilder.fill_rect(_modals_ctrl)
	add_child(_modals_ctrl)


# ==============================================================================
# TAB SWITCHING & REACTIVE STATE BINDINGS
# ==============================================================================
func _switch_tab(tab_idx: int) -> void:
	if AppState.active_workspace_tab != tab_idx:
		AppState.set_workspace_tab(tab_idx)
		return

	_tab_universe.visible = (tab_idx == 0)
	_tab_evolution_lab.visible = (tab_idx == 1)
	_tab_cancer_battle.visible = (tab_idx == 2)
	_tab_family_editor.visible = (tab_idx == 3)
	_tab_toxicity_pbpk.visible = (tab_idx == 4)
	_tab_ai_assistant.visible = (tab_idx == 5)

	for i in range(_tab_buttons.size()):
		var b: Button = _tab_buttons[i]
		if i == tab_idx:
			b.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_CYAN, 8, 2, 10, 7, 8, Color(0.13, 0.82, 0.93, 0.4)))
			b.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
		else:
			b.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 10, 7))
			b.add_theme_color_override("font_color", ThemeBuilder.TEXT_PRIMARY)


func _connect_app_signals() -> void:
	AppState.workspace_changed.connect(_switch_tab)
	AppState.state_updated.connect(_on_server_state_updated)
	AppState.entity_selected.connect(_on_entity_selected)
	AppState.compound_inspected.connect(func(data: Dictionary) -> void:
		var compound: Dictionary = data.get("compound", data)
		var svg: String = str(compound.get("svg_2d", data.get("svg_2d", "")))
		if not svg.is_empty():
			_compound_texture.texture = ApiClient.svg_to_texture(svg)
	)
	AppState.ai_providers_updated.connect(_on_ai_providers_updated)
	AppState.environment_preset_changed.connect(func(_p: String) -> void: _update_env_preset_highlight())
	AppState.lineage_loaded.connect(func(lineage: Dictionary) -> void:
		var chain: Array = lineage.get("ancestors", lineage.get("chain", []))
		_lineage_parent_lbl.text = "Recorded ancestors: %d" % chain.size()
		_lineage_gen_lbl.text = "Origin generation: %d" % int(AppState.selected_entity.get("generation", 0))
	)


func _process(delta: float) -> void:
	_chart_update_timer += delta
	if _chart_update_timer >= 0.5:
		_chart_update_timer = 0.0
		_refresh_bottom_charts()

	if _tour_stage >= 0:
		_step_screenshot_tour(delta)


func _on_server_state_updated(_state: Dictionary) -> void:
	var yrs: float = float(AppState.clocks.get("virtual_years", 0.0))
	var gen: int = int(AppState.clocks.get("evolutionary_generation", 0))

	_sim_time_lbl.text = "Gen %d | %.1f virtual yrs" % [gen, yrs]
	_play_pause_btn.text = "Pause" if AppState.is_playing else "Play"
	var job: Dictionary = _state.get("advance_job", {})
	if bool(job.get("active", false)):
		_sim_progress.value = 100.0 * (int(AppState.clocks.get("evolutionary_step", 0)) - int(job.get("start_step", 0))) / maxf(1, int(job.get("target_step", 1)) - int(job.get("start_step", 0)))
	else:
		_sim_progress.value = 0
	_gen_stepper_lbl.text = "Gen: %d" % gen

	_conn_status_lbl.text = "🟢 Engine: Active" if AppState.backend_connected else "🔴 Reconnecting..."
	_conn_status_lbl.add_theme_color_override("font_color", ThemeBuilder.TEXT_EMERALD if AppState.backend_connected else ThemeBuilder.BORDER_CRIMSON)

	# Update disease selection if synchronized from backend
	for i in range(_disease_opt.item_count):
		if _disease_opt.get_item_metadata(i) == AppState.active_disease:
			_disease_opt.selected = i
			break

	# Refresh current entity card if selected
	if not AppState.selected_entity.is_empty():
		_on_entity_selected(AppState.selected_entity)


func _on_entity_selected(ent: Dictionary) -> void:
	var name_str: String = str(ent.get("name", ent.get("compound_id", "Select a compound")))
	var fam: String = str(ent.get("ecological_role", ent.get("family", "Antitumor")))
	var gen: int = int(ent.get("generation", 0))
	var kind: String = str(ent.get("entity_kind", "THERAPEUTIC"))

	_entity_name_lbl.text = name_str
	_entity_family_lbl.text = "Family: " + fam.capitalize()
	_entity_gen_lbl.text = "Generation: G" + str(gen)
	_entity_type_lbl.text = "Type: " + kind

	if _portrait_instance != null and _portrait_instance.has_method("set_entity_data"):
		_portrait_instance.call("set_entity_data", ent)

	var tox: float = float(ent.get("toxicity_norm", 0.18))
	var anti: float = float(ent.get("antitumor_norm", 0.88))
	var energy: float = float(ent.get("energy_norm", 0.65))
	var mut: float = float(ent.get("mutation_rate", 0.25))

	_epistemic_tag_lbl.text = "Artificial life | activity/toxicity not calibrated"
	_bar_toxicity.value = tox
	_bar_antitumor.value = anti
	_bar_energy.value = energy
	_bar_mutation.value = mut
	for pb_pair in [[_bar_toxicity, tox], [_bar_antitumor, anti], [_bar_energy, energy], [_bar_mutation, mut]]:
		var pb_item: ProgressBar = pb_pair[0]
		if pb_item != null and pb_item.has_meta("value_label"):
			var vlbl: Label = pb_item.get_meta("value_label") as Label
			if vlbl != null:
				vlbl.text = "Unknown" if pb_item in [_bar_toxicity, _bar_antitumor] and not bool(ent.get("bioactivity_available", false)) else "%0.2f" % float(pb_pair[1])

	var smiles: String = str(ent.get("smiles", "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F"))
	_lineage_smiles_lbl.text = "Scaffold: " + smiles
	_compound_formula_lbl.text = "Formula: " + str(ent.get("formula", "C9H11F2N3O4"))
	_compound_mw_lbl.text = "MW: %0.2f Da" % float(ent.get("mol_weight", 263.20))
	_compound_logp_lbl.text = "LogP: %0.2f | QED: %0.2f" % [float(ent.get("logp", -1.40)), float(ent.get("qed", 0.68))]

	# Render RDKit SVG texture if available
	var svg: String = str(ent.get("svg_2d", ""))
	if not svg.is_empty() and has_node("/root/ApiClient"):
		var tex: Texture2D = get_node("/root/ApiClient").svg_to_texture(svg, 1.2, true)
		if tex != null:
			_compound_texture.texture = tex


func _refresh_bottom_charts() -> void:
	if AppState.chart_years.is_empty():
		return

	var family_colors: Dictionary = {
		"Catalysts": ThemeBuilder.BORDER_CYAN,
		"Transporters": ThemeBuilder.BORDER_EMERALD,
		"Degraders": ThemeBuilder.BORDER_CRIMSON,
		"Antitumor": ThemeBuilder.BORDER_VIOLET,
		"Structural": ThemeBuilder.TEXT_AMBER,
		"Others": ThemeBuilder.TEXT_MUTED
	}

	if _chart_pop_inst != null and _chart_pop_inst.has_method("set_multi_series"):
		_chart_pop_inst.call("set_multi_series", AppState.chart_years, AppState.chart_family_series, family_colors, AppState.latest_family_counts)

	if _chart_tox_inst != null and _chart_tox_inst.has_method("set_band_series"):
		_chart_tox_inst.call("set_band_series", AppState.chart_years, AppState.chart_mean_toxicity, AppState.chart_std_toxicity, ThemeBuilder.BORDER_CRIMSON)

	if _chart_anti_inst != null and _chart_anti_inst.has_method("set_band_series"):
		_chart_anti_inst.call("set_band_series", AppState.chart_years, AppState.chart_mean_antitumor, AppState.chart_std_antitumor, ThemeBuilder.BORDER_EMERALD)


func _on_ai_providers_updated(dict: Dictionary) -> void:
	var prov: String = str(dict.get("active_provider", "auto_local"))
	var model: String = str(dict.get("active_model", "gemma-4-E2B-it-Q4_K_M"))
	_ai_badge_btn.text = "AI: " + prov.capitalize()


func _on_disease_selected(idx: int) -> void:
	var d_id: String = _disease_opt.get_item_metadata(idx)
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").select_disease(d_id)


func _on_env_preset_clicked(preset_id: String) -> void:
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").apply_environment_preset(preset_id)
	_update_env_preset_highlight()


func _update_env_preset_highlight() -> void:
	var active_p: String = AppState.active_environment_preset
	for p_id in _env_preset_buttons.keys():
		var btn: Button = _env_preset_buttons[p_id]
		if p_id == active_p:
			btn.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_CYAN, 8, 2, 10, 6, 6, Color(0.13, 0.82, 0.93, 0.35)))
			btn.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN)
		else:
			btn.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 10, 6))
			btn.add_theme_color_override("font_color", ThemeBuilder.TEXT_PRIMARY)


func _set_interface_mode(mode: String) -> void:
	AppState.interface_mode = mode
	_update_interface_mode_styles()
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").request_json(HTTPClient.METHOD_POST, "/api/v1/mode/interface", {"mode": mode}, func(_c: int, _r: Variant) -> void:
			AppState.notify_toast("Interface mode switched to " + mode, "info")
		)


func _check_cli_screenshot_tour() -> void:
	var all_args: PackedStringArray = OS.get_cmdline_args()
	all_args.append_array(OS.get_cmdline_user_args())
	var out_dir: String = ""
	for arg in all_args:
		if arg.begins_with("--screenshot-tour="):
			out_dir = arg.trim_prefix("--screenshot-tour=").rstrip("/\\")
			break
	if out_dir.is_empty():
		return
	DirAccess.make_dir_recursive_absolute(out_dir)
	_tour_out_dir = out_dir
	_tour_elapsed = 0.0
	_tour_stage = 0


func _step_screenshot_tour(delta: float) -> void:
	_tour_elapsed += delta
	if _tour_stage == 0 and _tour_elapsed >= 2.8:
		if not AppState.entities.is_empty():
			var preferred: Dictionary = AppState.entities[0]
			for item in AppState.entities:
				if item is Dictionary and str(item.get("entity_kind", "")) == "THERAPEUTIC":
					preferred = item
					break
			AppState.select_entity(preferred)
		_refresh_bottom_charts()
		_tour_stage = 1
		_tour_elapsed = 0.0
	elif _tour_stage == 1 and _tour_elapsed >= 0.8:
		get_viewport().get_texture().get_image().save_png(_tour_out_dir + "/godot_01_chemical_universe.png")
		if _modals_ctrl != null:
			_modals_ctrl.open_ai_modal()
			if _modals_ctrl.has_method("select_provider_for_preview"):
				_modals_ctrl.call("select_provider_for_preview", "gemini")
		_tour_stage = 2
		_tour_elapsed = 0.0
	elif _tour_stage == 2 and _tour_elapsed >= 0.8:
		get_viewport().get_texture().get_image().save_png(_tour_out_dir + "/godot_02_ai_providers_gemini_openai.png")
		if _modals_ctrl != null:
			_modals_ctrl.close_all_modals()
		_switch_tab(2)
		_tour_stage = 3
		_tour_elapsed = 0.0
	elif _tour_stage == 3 and _tour_elapsed >= 0.8:
		get_viewport().get_texture().get_image().save_png(_tour_out_dir + "/godot_03_cancer_battle.png")
		_switch_tab(3)
		if _tab_family_editor != null and _tab_family_editor.has_method("_on_apply_mutation_clicked"):
			_tab_family_editor.call("_on_apply_mutation_clicked")
		_tour_stage = 4
		_tour_elapsed = 0.0
	elif _tour_stage == 4 and _tour_elapsed >= 1.1:
		get_viewport().get_texture().get_image().save_png(_tour_out_dir + "/godot_04_family_editor.png")
		_tour_stage = -1
		get_tree().quit(0)


func _update_interface_mode_styles() -> void:
	var is_lab: bool = (AppState.interface_mode == "LAB")
	if is_lab:
		_mode_lab_btn.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_EMERALD, 8, 2, 10, 6, 8, Color(0.20, 0.82, 0.60, 0.4)))
		_mode_play_btn.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 10, 6))
	else:
		_mode_lab_btn.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD, ThemeBuilder.BORDER_SUBTLE, 8, 1, 10, 6))
		_mode_play_btn.add_theme_stylebox_override("normal", ThemeBuilder.make_glass_box(ThemeBuilder.BG_CARD_ACTIVE, ThemeBuilder.BORDER_CYAN, 8, 2, 10, 6, 8, Color(0.13, 0.82, 0.93, 0.4)))


func _on_overlay_selected(idx: int) -> void:
	var modes := ["none", "substrate", "oxygen", "ecm", "drug"]
	var selected_mode: String = modes[idx]
	AppState.world_overlay_mode_changed.emit(selected_mode)
	if _world_view != null and _world_view.has_method("set_heatmap_overlay"):
		_world_view.call("set_heatmap_overlay", selected_mode)


func _on_play_pause_clicked() -> void:
	var new_action: String = "pause" if AppState.is_playing else "play"
	_play_pause_btn.text = "⏸ Pause" if new_action == "play" else "▶ Play"
	if has_node("/root/ApiClient"):
		get_node("/root/ApiClient").set_playback(new_action, AppState.playback_speed)


func _set_speed(spd: int) -> void:
	AppState.playback_speed = spd
	_speed_1x_btn.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN if spd == 1 else ThemeBuilder.TEXT_MUTED)
	_speed_10x_btn.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN if spd == 10 else ThemeBuilder.TEXT_MUTED)
	_speed_100x_btn.add_theme_color_override("font_color", ThemeBuilder.BORDER_CYAN if spd == 100 else ThemeBuilder.TEXT_MUTED)
	if has_node("/root/ApiClient") and AppState.is_playing:
		get_node("/root/ApiClient").set_playback("play", spd)


func _safe_instantiate_control(path: String) -> Control:
	if ResourceLoader.exists(path):
		var scr: Variant = load(path)
		if scr is GDScript and scr.can_instantiate():
			return scr.new() as Control
	return Control.new()

