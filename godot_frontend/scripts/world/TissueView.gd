class_name TissueView
extends Control

signal cell_selected(data: Dictionary)
var snapshot: Dictionary = {}
var overlay: String = "oxygen"
var selected_id: String = ""
var zoom: float = 1.0
var _elapsed: float = 0.0
var _offset := Vector2.ZERO
var _scale: float = 1.0
var _last_deaths: Dictionary = {}
var _death_flashes: Dictionary = {}
var _new_cells: Dictionary = {}
var _known_cells: Dictionary = {}

func _ready() -> void:
	clip_contents = true
	custom_minimum_size = Vector2(500, 400)
	mouse_filter = Control.MOUSE_FILTER_STOP

func update_snapshot(data: Dictionary) -> void:
	var reset: bool = float(data.get("biological_hours", 0)) < float(snapshot.get("biological_hours", 0)) or data.get("disease_id") != snapshot.get("disease_id")
	if reset:
		_known_cells.clear()
		_last_deaths.clear()
		_death_flashes.clear()
		_new_cells.clear()
	var initial: bool = _known_cells.is_empty()
	for cell in data.get("cells", []):
		var id: String = str(cell.get("id", ""))
		var death: String = str(cell.get("st", "ALIVE"))
		if not initial and not _known_cells.has(id):
			_new_cells[id] = _elapsed
		if death != "ALIVE" and _last_deaths.get(id, "ALIVE") == "ALIVE":
			_death_flashes[id] = _elapsed
		_known_cells[id] = true
		_last_deaths[id] = death
	snapshot = data
	queue_redraw()

func _process(delta: float) -> void:
	if not visible:
		return
	_elapsed += delta
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color("09151e"))
	var domain: Dictionary = snapshot.get("domain", {})
	var world := Vector2(float(domain.get("width", 800)), float(domain.get("height", 600)))
	_scale = minf((size.x - 40) / world.x, (size.y - 80) / world.y) * zoom
	_offset = (size - world * _scale) * 0.5 + Vector2(0, 12)
	draw_set_transform(_offset, 0, Vector2.ONE * _scale)
	draw_rect(Rect2(Vector2.ZERO, world), Color("142329"))
	_draw_field(world)
	_draw_ecm(world)
	for vessel in snapshot.get("vessels", []):
		var a := Vector2(float(vessel.get("x1", 0)), float(vessel.get("y1", 0)))
		var b := Vector2(float(vessel.get("x2", 0)), float(vessel.get("y2", 0)))
		var width: float = float(vessel.get("radius", 8)) * 1.4
		draw_line(a, b, Color(0.48, 0.14, 0.20, 0.8), width + 3, true)
		draw_line(a, b, Color(0.80, 0.30, 0.32, 0.7), width, true)
	var cells: Array = snapshot.get("cells", [])
	var stride: int = maxi(1, int(ceil(cells.size() / 1500.0)))
	for i in range(0, cells.size(), stride):
		_draw_cell(cells[i])
	draw_set_transform(Vector2.ZERO)
	var title: String = str(snapshot.get("display_name", "Live tissue"))
	draw_string(ThemeDB.fallback_font, Vector2(15, 24), title, HORIZONTAL_ALIGNMENT_LEFT, size.x - 30, 16, Color("e6f1ff"))
	draw_string(ThemeDB.fallback_font, Vector2(15, 45), "Biological time %.2f h · %d cells (%d shown) · wheel to zoom · click to inspect" % [float(snapshot.get("biological_hours", 0)), cells.size(), int(ceil(cells.size() / float(stride)))], HORIZONTAL_ALIGNMENT_LEFT, size.x - 30, 12, Color("a6bac8"))
	var disease: String = str(snapshot.get("disease_id", "PDAC"))
	var legend: String = "Coral: tumour · violet outline: resistant · blue: healthy · amber: stroma · green: immune · grey: dead"
	if disease == "AMR_BACTERIAL":
		legend = "Green rods: bacteria · violet: resistant bacteria · blue: host mucosa · amber: biofilm · grey: dead"
	elif disease == "METABOLIC_MASLD":
		legend = "Ochre: steatotic hepatocytes · dots: steatotic cell state · amber: stellate/fibrosis · blue: healthier cells"
	draw_string(ThemeDB.fallback_font, Vector2(15, size.y - 28), legend, HORIZONTAL_ALIGNMENT_LEFT, size.x - 30, 11, Color("c3d0dc"))
	draw_string(ThemeDB.fallback_font, Vector2(15, size.y - 10), "Spatial state from the model. Membrane animation and shapes are illustrative, not microscopy. Field scale: " + overlay, HORIZONTAL_ALIGNMENT_LEFT, size.x - 30, 11, Color("90a3b4"))

func _draw_field(world: Vector2) -> void:
	if overlay == "none":
		return
	var values: Array = snapshot.get("heatmaps", {}).get(overlay, [])
	if values.is_empty() or not values[0] is Array:
		return
	var peak: float = 1.0 if overlay in ["oxygen", "nutrient", "ecm_density"] else 0.001
	for row in values:
		for value in row:
			peak = maxf(peak, float(value))
	var dimensions := Vector2(world.x / values[0].size(), world.y / values.size())
	for y in range(values.size()):
		for x in range(values[y].size()):
			var value: float = clampf(float(values[y][x]) / peak, 0, 1)
			var color: Color = Color(0.15, 0.35, 0.75, value * 0.5)
			if overlay == "oxygen":
				color = Color(0.30, 0.10, 0.45, (1.0 - value) * 0.65)
			elif overlay == "ecm_density":
				color = Color(0.8, 0.58, 0.25, value * 0.45)
			elif overlay == "drug_free":
				color = Color(0.1, 0.7, 0.65, value * 0.28)
			draw_rect(Rect2(Vector2(x, y) * dimensions, dimensions), color)
	draw_string(ThemeDB.fallback_font, Vector2(12, world.y - 12), "Field maximum %.4f %s" % [peak, "uM" if overlay == "drug_free" else "model units"], HORIZONTAL_ALIGNMENT_LEFT, -1, 12 / _scale, Color("aec5cd"))

func _draw_ecm(world: Vector2) -> void:
	var values: Array = snapshot.get("heatmaps", {}).get("ecm_density", [])
	if values.is_empty():
		return
	var cell := Vector2(world.x / values[0].size(), world.y / values.size())
	for y in range(0, values.size(), 2):
		for x in range(0, values[y].size(), 2):
			var density: float = float(values[y][x])
			if density > 0.12:
				var point := Vector2(x + 0.5, y + 0.5) * cell
				draw_line(point - cell * Vector2(0.8, 0.2), point + cell * Vector2(0.8, 0.2), Color(0.72, 0.57, 0.3, minf(0.7, density)), 1.0, true)

func _draw_cell(cell: Dictionary) -> void:
	var point := Vector2(float(cell.get("x", 0)), float(cell.get("y", 0)))
	var radius: float = maxf(3.0, float(cell.get("r", 6)))
	var type: String = str(cell.get("t", ""))
	var death: String = str(cell.get("st", "ALIVE"))
	var id: String = str(cell.get("id", ""))
	var seed: int = absi(id.hash())
	var disease: String = str(snapshot.get("disease_id", "PDAC"))
	var is_pathology: bool = type in ["TUMOR_SENSITIVE", "TUMOR_RESISTANT"]
	var resistant: bool = type == "TUMOR_RESISTANT" or float(cell.get("res", 0)) > 0.45
	var stroma: bool = type in ["CAF", "FIBROBLAST"]
	var color := Color("4a91b0")
	if is_pathology:
		color = Color("c76d7a") if disease in ["PDAC", "NSCLC_KRAS"] else (Color("70af7a") if disease == "AMR_BACTERIAL" else Color("bd934d"))
	elif stroma:
		color = Color("b79a59")
	elif type in ["IMMUNE_T", "MACROPHAGE"]:
		color = Color("6fbc9a")
	var flash: float = clampf(1.0 - (_elapsed - float(_death_flashes.get(id, -99))) / 3.0, 0, 1)
	if death != "ALIVE":
		color = Color("657078") if death == "NECROSIS" else Color("725a71")
		color.a = 0.35 + flash * 0.5
		radius *= 0.65 if death == "APOPTOSIS" else 1.1
	var breath: float = sin(_elapsed * 1.3 + float(seed % 50)) * 0.025 if death == "ALIVE" else 0.0
	var points := PackedVector2Array()
	for i in range(16):
		var angle: float = TAU * i / 16.0
		var shape: float = 1.0 + sin(angle * 3 + float(seed % 7)) * 0.10 + breath
		var vector := Vector2(cos(angle), sin(angle)) * radius * shape
		if stroma:
			vector *= Vector2(1.8, 0.6)
		elif disease == "AMR_BACTERIAL" and is_pathology:
			vector *= Vector2(1.7, 0.7)
		points.append(point + vector.rotated(float(seed % 180) * PI / 180))
	draw_colored_polygon(points, Color(color, color.a * 0.65))
	points.append(points[0])
	draw_polyline(points, color.lightened(0.25), 1.0, true)
	if disease != "AMR_BACTERIAL" or not is_pathology:
		draw_circle(point + Vector2(radius * 0.12, -radius * 0.1), radius * 0.42, Color(color.darkened(0.48), color.a))
	if disease == "METABOLIC_MASLD" and is_pathology and death == "ALIVE":
		for i in range(3):
			var offset := Vector2(cos(float(i) * 2.1), sin(float(i) * 2.1)) * radius * 0.55
			draw_circle(point + offset, radius * 0.20, Color(0.91, 0.78, 0.42, 0.7))
	if resistant and death == "ALIVE":
		draw_arc(point, radius + 2, 0, TAU, 20, Color("b99ce5"), 1.2, true)
	var uptake: float = float(cell.get("drg", 0))
	if uptake > 0.001 and death == "ALIVE":
		draw_arc(point, radius + 3, 0, TAU * minf(1, uptake / 2.0), 20, Color(0.2, 1.0, 0.8, 0.85), 1.0, true)
	var born: float = clampf(1.0 - (_elapsed - float(_new_cells.get(id, -99))) / 3.0, 0, 1)
	if born > 0:
		draw_arc(point, radius + 5 + (1 - born) * 8, 0, TAU, 20, Color(0.4, 1, 0.6, born), 1.3, true)
	if flash > 0:
		draw_arc(point, radius + 4 + (1 - flash) * 10, 0, TAU, 20, Color(1.0, 0.7, 0.8, flash), 1.4, true)
	if id == selected_id:
		draw_arc(point, radius + 7, 0, TAU, 30, Color.WHITE, 1.8, true)

func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			zoom = minf(3, zoom * 1.15)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			zoom = maxf(0.6, zoom / 1.15)
		elif event.button_index == MOUSE_BUTTON_LEFT:
			var point: Vector2 = (event.position - _offset) / _scale
			var nearest: Dictionary = {}
			var distance: float = 30 / _scale
			for cell in snapshot.get("cells", []):
				var d: float = point.distance_to(Vector2(float(cell.get("x", 0)), float(cell.get("y", 0))))
				if d < distance:
					distance = d
					nearest = cell
			if not nearest.is_empty():
				selected_id = str(nearest.get("id", ""))
				cell_selected.emit(nearest)
		accept_event()
