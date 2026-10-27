extends Node
## Central reactive state store for EVOCURE Godot 4.7 Frontend.
## Synchronizes live scientific simulation telemetry, entity selection,
## chart histories, active disease, PLAY/LAB mode, and AI provider settings.

signal state_updated(state: Dictionary)
signal entity_selected(entity_data: Dictionary)
signal compound_inspected(inspection_data: Dictionary)
signal lineage_loaded(lineage_data: Dictionary)
signal library_loaded(compounds: Array)
signal diseases_loaded(diseases: Array)
signal ai_providers_updated(providers_info: Dictionary)
signal workspace_changed(tab_index: int)
signal environment_preset_changed(preset_id: String)
signal toast_requested(message: String, severity: String)
signal world_overlay_mode_changed(overlay_mode: String)
signal spawn_tool_changed(compound_query: String)

var backend_connected: bool = false
var interface_mode: String = "PLAY" # "PLAY" or "LAB"
var universe_mode: String = "ARTIFICIAL_LIFE" # "ARTIFICIAL_LIFE" or "GROUNDED_CHEMISTRY"
var active_disease: String = "PDAC"
var active_environment_preset: String = "pancreatic_tissue"
var active_workspace_tab: int = 0
var is_playing: bool = false
var playback_speed: int = 1
var playback_target: String = "both"

# Visual & Accessibility preferences
var bloom_enabled: bool = true
var colorblind_palette: String = "bioluminescent" # "bioluminescent", "okabe_ito", "high_contrast"
var heatmap_overlay: String = "none" # "none", "substrate", "oxygen", "ecm", "drug"
var active_spawn_compound: String = "" # Non-empty when click-to-spawn is armed
var analysis_compound_query: String = ""
var tissue_compound_query: String = "Gemcitabine"

# Latest snapshots from Python scientific engine
var latest_state: Dictionary = {}
var clocks: Dictionary = {
	"computational_seconds": 0.0,
	"evolutionary_step": 0,
	"evolutionary_generation": 0,
	"virtual_years": 0.0,
	"biological_seconds": 0.0,
	"biological_hours": 0.0,
}
var environment: Dictionary = {
	"width": 1000.0,
	"height": 700.0,
	"temperature_k": 310.15,
	"ph": 7.4,
	"viscosity_pa_s": 0.00089,
	"oxygen_norm": 1.0,
}
var entities: Array = []
var tumor_snapshot: Dictionary = {}
var selected_entity: Dictionary = {}
var inspected_compound: Dictionary = {}
var selected_lineage_chain: Array = []

var compound_library: Array = []
var available_diseases: Array = []
var missions_glossary: Dictionary = {}
var ai_providers_status: Dictionary = {
	"active_provider": "auto_local",
	"active_model": "gemma-4-E2B-it-Q4_K_M",
	"providers": {}
}

# Rolling chart histories for the 3 bottom scientific charts
const MAX_CHART_POINTS: int = 120
var chart_years: Array[float] = []
var chart_family_series: Dictionary = {
	"Catalysts": [],
	"Transporters": [],
	"Degraders": [],
	"Antitumor": [],
	"Structural": [],
	"Others": [],
}
var chart_mean_toxicity: Array[float] = []
var chart_std_toxicity: Array[float] = []
var chart_mean_antitumor: Array[float] = []
var chart_std_antitumor: Array[float] = []
var latest_family_counts: Dictionary = {
	"Catalysts": 0,
	"Transporters": 0,
	"Degraders": 0,
	"Antitumor": 0,
	"Structural": 0,
	"Others": 0,
}


func ingest_server_state(data: Dictionary) -> void:
	if data.is_empty():
		return
	latest_state = data
	backend_connected = true
	if data.get("playback") is Dictionary:
		is_playing = bool(data["playback"].get("is_running", false))
		playback_speed = int(data["playback"].get("speed_multiplier", 1))

	if data.has("interface_mode"):
		interface_mode = str(data["interface_mode"])
	if data.has("universe_mode"):
		universe_mode = str(data["universe_mode"])
	if data.has("clocks") and data["clocks"] is Dictionary:
		clocks = data["clocks"]
	if data.has("environment") and data["environment"] is Dictionary:
		environment = data["environment"]
	if data.has("entities") and data["entities"] is Array:
		entities = data["entities"]
	if data.has("tumor") and data["tumor"] is Dictionary:
		tumor_snapshot = data["tumor"]
		if tumor_snapshot.has("disease_id"):
			active_disease = str(tumor_snapshot["disease_id"])
	if data.has("ai_provider_summary") and data["ai_provider_summary"] is Dictionary:
		var summary: Dictionary = data["ai_provider_summary"]
		if summary.has("active_provider"):
			ai_providers_status["active_provider"] = str(summary["active_provider"])
		if summary.has("active_model"):
			ai_providers_status["active_model"] = str(summary["active_model"])

	_update_chart_buffers(data)

	# Keep selected entity refreshed if still alive, or auto-select first entity once
	if selected_entity.is_empty() and not entities.is_empty():
		select_entity(entities[0])
	elif not selected_entity.is_empty():
		var sel_id: String = str(selected_entity.get("entity_id", ""))
		if not sel_id.is_empty():
			for ent in entities:
				if ent is Dictionary and str(ent.get("entity_id", "")) == sel_id:
					selected_entity = ent
					break

	state_updated.emit(latest_state)


func _update_chart_buffers(data: Dictionary) -> void:
	if data.has("chart_history") and data["chart_history"] is Array and not (data["chart_history"] as Array).is_empty():
		chart_years.clear()
		for k in chart_family_series.keys():
			(chart_family_series[k] as Array).clear()
		chart_mean_toxicity.clear()
		chart_std_toxicity.clear()
		chart_mean_antitumor.clear()
		chart_std_antitumor.clear()

		for pt in data["chart_history"]:
			if not (pt is Dictionary):
				continue
			chart_years.append(float(pt.get("virtual_years", 0.0)))
			var fp: Dictionary = pt.get("family_populations", {}) if pt.get("family_populations") is Dictionary else {}
			var c_cat: int = int(fp.get("catalysts", 0))
			var c_trn: int = int(fp.get("transporters", 0))
			var c_deg: int = int(fp.get("degraders", 0))
			var c_ant: int = int(fp.get("antitumor", 0))
			var c_str: int = int(fp.get("structural", 0))
			var c_oth: int = int(fp.get("others", 0))
			(chart_family_series["Catalysts"] as Array).append(float(c_cat))
			(chart_family_series["Transporters"] as Array).append(float(c_trn))
			(chart_family_series["Degraders"] as Array).append(float(c_deg))
			(chart_family_series["Antitumor"] as Array).append(float(c_ant))
			(chart_family_series["Structural"] as Array).append(float(c_str))
			(chart_family_series["Others"] as Array).append(float(c_oth))
			latest_family_counts = {
				"Catalysts": c_cat,
				"Transporters": c_trn,
				"Degraders": c_deg,
				"Antitumor": c_ant,
				"Structural": c_str,
				"Others": c_oth,
			}
			chart_mean_toxicity.append(float(pt.get("mean_energy_norm", 0.0)))
			chart_std_toxicity.append(float(0.0))
			chart_mean_antitumor.append(float(pt.get("species_diversity_ratio", 0.0)))
			chart_std_antitumor.append(float(0.0))
		return

	# Client-side computation from real entity array if server chart_history not yet populated
	var yr: float = float(clocks.get("virtual_years", float(chart_years.size()) * 0.1))
	if not chart_years.is_empty() and absf(chart_years[-1] - yr) < 0.0001 and chart_years.size() > 2:
		return

	var counts := {
		"Catalysts": 0,
		"Transporters": 0,
		"Degraders": 0,
		"Antitumor": 0,
		"Structural": 0,
		"Others": 0,
	}
	var sum_tox: float = 0.0
	var sum_anti: float = 0.0
	var n: int = maxi(1, entities.size())

	for ent in entities:
		if not (ent is Dictionary):
			continue
		var role: String = str(ent.get("ecological_role", ent.get("role", "neutral"))).to_lower()
		var kind: String = str(ent.get("entity_kind", ent.get("kind", "MOLECULE"))).to_upper()
		if role.contains("catalyst") or kind == "CATALYST":
			counts["Catalysts"] += 1
		elif role.contains("cooperator") or role.contains("transport"):
			counts["Transporters"] += 1
		elif role.contains("parasite") or role.contains("degrad"):
			counts["Degraders"] += 1
		elif kind == "THERAPEUTIC" or float(ent.get("antitumor_norm", 0.0)) > 0.55:
			counts["Antitumor"] += 1
		elif role.contains("autotroph") or kind == "PROTOCELL":
			counts["Structural"] += 1
		else:
			counts["Others"] += 1

		sum_tox += float(ent.get("toxicity_norm", 0.22))
		sum_anti += float(ent.get("antitumor_norm", 0.38))

	latest_family_counts = counts
	chart_years.append(yr)
	for k in counts.keys():
		var arr: Array = chart_family_series[k]
		arr.append(float(counts[k]))
		if arr.size() > MAX_CHART_POINTS:
			arr.pop_front()

	chart_mean_toxicity.append(clampf(sum_tox / float(n), 0.0, 1.0))
	chart_std_toxicity.append(0.08)
	chart_mean_antitumor.append(clampf(sum_anti / float(n), 0.0, 1.0))
	chart_std_antitumor.append(0.09)

	while chart_years.size() > MAX_CHART_POINTS:
		chart_years.pop_front()
		chart_mean_toxicity.pop_front()
		chart_std_toxicity.pop_front()
		chart_mean_antitumor.pop_front()
		chart_std_antitumor.pop_front()


func select_entity(entity_data: Dictionary) -> void:
	selected_entity = entity_data
	entity_selected.emit(entity_data)
	var cid: String = str(entity_data.get("compound_id", entity_data.get("name", "")))
	if not cid.is_empty() and has_node("/root/ApiClient"):
		get_node("/root/ApiClient").inspect_compound(cid)
		get_node("/root/ApiClient").fetch_entity_lineage(cid)


func set_workspace_tab(tab_idx: int) -> void:
	if active_workspace_tab == tab_idx:
		return
	active_workspace_tab = tab_idx
	workspace_changed.emit(tab_idx)


func notify_toast(msg: String, severity: String = "info") -> void:
	toast_requested.emit(msg, severity)
