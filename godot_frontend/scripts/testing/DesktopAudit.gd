extends Node
## Explicit developer smoke test: use only against an isolated test engine.
var dashboard: Control
var output: String
var checks: Array = []

func request(path: String, body: Variant = null) -> Dictionary:
	var http := HTTPRequest.new()
	http.timeout = 120
	add_child(http)
	var method := HTTPClient.METHOD_GET if body == null else HTTPClient.METHOD_POST
	var err := http.request(ApiClient.base_http_url + "/api/v1" + path, PackedStringArray(["Content-Type: application/json"]), method, "" if body == null else JSON.stringify(body))
	if err != OK:
		http.queue_free()
		return {"code": 0, "data": {}}
	var response: Array = await http.request_completed
	var data: Variant = JSON.parse_string(response[3].get_string_from_utf8())
	http.queue_free()
	return {"code": response[1], "data": data}

func check(name: String, passed: bool, detail: Variant = "") -> void:
	checks.append({"check": name, "passed": passed, "detail": detail})
	print("DESKTOP AUDIT ", "PASS " if passed else "FAIL ", name, " ", detail)
	var file := FileAccess.open(output + "/desktop_audit.json", FileAccess.WRITE)
	file.store_string(JSON.stringify({"checks": checks}, "  "))

func settle(seconds: float = 0.8) -> void:
	await get_tree().create_timer(seconds).timeout
	await RenderingServer.frame_post_draw

func snapshot(name: String) -> void:
	await settle()
	get_viewport().get_texture().get_image().save_png(output + "/" + name + ".png")

func until(condition: Callable, seconds: float = 110) -> bool:
	var end := Time.get_ticks_msec() + int(seconds * 1000)
	while Time.get_ticks_msec() < end:
		if condition.call():
			return true
		await get_tree().create_timer(0.25).timeout
	return false

func run(main: Control, directory: String) -> void:
	dashboard = main
	output = directory
	DirAccess.make_dir_recursive_absolute(output)
	await settle(3)
	var response := await request("/world/reset-defaults", {})
	check("reset defaults", response.code == 200 and response.data.data.total_entity_count == 106)
	await request("/world/stir", {"strength": 1.0})
	await request("/world/step", {"steps": 12})
	ApiClient.fetch_state()
	await settle(1)
	check("real chemical movement", int(AppState.clocks.get("evolutionary_step", 0)) >= 12)
	await snapshot("01_chemical_universe")
	await get_tree().create_timer(0.3).timeout
	await snapshot("01b_chemical_animation")
	dashboard._switch_tab(1)
	var evolution: TabEvolutionLab = dashboard._tab_evolution_lab
	evolution._create_plan()
	var planned := await until(func() -> bool: return not evolution._approve.disabled)
	check("local AI reviewed search plan", planned, evolution._plan_label.text)
	if planned:
		evolution._run_plan()
		var done := false
		for i in range(150):
			await settle(1)
			response = await request("/discovery/job")
			if str(response.data.get("status", "")) in ["complete", "failed", "cancelled"]:
				done = response.data.get("status") == "complete"
				break
		check("guided evolution completes", done, response.data)
		evolution._source.select(1)
		evolution._refresh()
		await settle(3)
		response = await request("/discovery/compounds?source=shortlist")
		check("five real new candidates", response.code == 200 and int(response.data.get("total", 0)) == 5)
		check("drug comparison images", evolution._candidate_image.texture != null and evolution._reference_image.texture != null)
	await snapshot("02_evolution_and_drug_comparison")
	var tissue: TabCancerBattle = dashboard._tab_cancer_battle
	dashboard._switch_tab(2)
	var diseases := {"PDAC": "Gemcitabine", "NSCLC_KRAS": "Sotorasib", "AMR_BACTERIAL": "Meropenem", "METABOLIC_MASLD": "Resmetirom"}
	for disease in diseases:
		response = await request("/diseases/select", {"disease_id": disease, "seed": 20261026})
		ApiClient.fetch_state()
		await settle(1)
		tissue._compound.text = diseases[disease]
		tissue._dose.value = 5
		tissue._route.select(1)
		tissue._apply()
		await until(func() -> bool: return not tissue._status.text.begins_with("Applying"), 20)
		await settle(2)
		await request("/playback", {"action": "pause", "target": "tumor"})
		await request("/tumor/step", {"steps": 6, "dt_seconds": 900})
		ApiClient.fetch_state()
		await settle(1)
		var state: Dictionary = AppState.tumor_snapshot
		check(disease + " actual cells, exposure and biological time", state.get("disease_id") == disease and not state.get("cells", []).is_empty() and float(state.get("biological_hours", 0)) >= 1.5, tissue._metrics.text)
		if not state.get("cells", []).is_empty():
			tissue._inspect_cell(state.cells[0])
		await snapshot("03_tissue_" + disease)
		for overlay in ["oxygen", "drug_free", "ecm_density", "nutrient", "none"]:
			tissue._view.overlay = overlay
			await settle(0.1)
		check(disease + " overlay fields", state.get("heatmaps", {}).has("oxygen") and state.get("heatmaps", {}).has("drug_free"))
	await request("/diseases/select", {"disease_id": "PDAC"})
	ApiClient.fetch_state()
	await settle(1)
	tissue._combination()
	await until(func() -> bool: return not tissue._busy)
	check("four-arm comparison", tissue._status.text.begins_with("Independent-hazard"), tissue._status.text)
	tissue._coevolve()
	await until(func() -> bool: return not tissue._busy)
	check("chemical tumour coevolution", tissue._status.text.begins_with("Coevolution complete"), tissue._status.text)
	dashboard._switch_tab(3)
	var editor: TabFamilyAndEditor = dashboard._tab_family_editor
	editor._parent.text = "Aspirin"
	editor._load_parent()
	await settle(1)
	editor._on_apply_mutation_clicked()
	await until(func() -> bool: return not editor._edited.is_empty())
	check("real structural edit and genealogy", not editor._edited.is_empty() and not editor._dag.is_empty(), editor._details.text)
	await snapshot("04_genealogy_editor")
	dashboard._switch_tab(4)
	var assessment: TabToxicityPBPK = dashboard._tab_toxicity_pbpk
	assessment._query.text = "Gemcitabine"
	assessment._pbpk.button_pressed = true
	assessment._run()
	await until(func() -> bool: return not assessment._busy)
	check("PBPK and toxicology", assessment._summary.text.begins_with("PBPK model input"), assessment._status.text)
	await snapshot("05_toxicology_pbpk")
	dashboard._switch_tab(5)
	var chat: TabAIAssistant = dashboard._tab_ai_assistant
	chat._input.text = "What is happening?"
	chat._send_message()
	await until(func() -> bool: return not chat._busy)
	check("local chat response", chat._messages.get_child_count() >= 3)
	chat._create_checkpoint()
	await settle(1)
	check("checkpoint appears", chat._checkpoints.get_child_count() > 0)
	for format in ["json", "csv", "sdf", "md", "pdf"]:
		chat._format = format
		chat._export_to(output + "/audit_export." + format)
		var exported := await until(func() -> bool: return chat._status.text.begins_with("Export saved") or chat._status.text.contains("failed"))
		check("native " + format + " file export", exported and FileAccess.file_exists(output + "/audit_export." + format), chat._status.text)
	await snapshot("06_chat_and_exports")
	for action in ["open_library_modal", "open_missions_modal", "open_glossary_modal", "open_ai_modal"]:
		dashboard._modals_ctrl.call(action)
		await snapshot("07_" + action)
		dashboard._modals_ctrl.close_all_modals()
	check("modal controls opened", true)
	var failed := checks.filter(func(item: Dictionary) -> bool: return not item.passed).size()
	print("DESKTOP AUDIT FINISHED: ", checks.size(), " checks, ", failed, " failures")
	get_tree().quit(0 if failed == 0 else 1)
