class_name GenealogyGraphControl
extends Control
## Interactive Evolutionary Lineage DAG Chart for EVOCURE.
## Renders generational nodes (extant vs extinct), glowing Bezier lineage edges,
## interactive zoom & pan navigation, and node selection callbacks.

signal node_clicked(node_dict: Dictionary)

var nodes_data: Array = []
var edges_data: Array = []

var _node_positions: Dictionary = {} # id -> Vector2 (in graph space)
var _node_dict_by_id: Dictionary = {} # id -> Dictionary

# Camera & Navigation
var _zoom_scale: float = 1.0
var _pan_offset: Vector2 = Vector2(50.0, 50.0)
var _is_panning: bool = false
var _pan_start_mouse: Vector2 = Vector2.ZERO
var _pan_start_offset: Vector2 = Vector2.ZERO

# Hover and selection
var _hovered_node_id: String = ""
var _selected_node_id: String = ""


func _init() -> void:
	custom_minimum_size = Vector2(300.0, 180.0)
	clip_contents = true
	mouse_filter = MOUSE_FILTER_STOP


func _ready() -> void:
	mouse_exited.connect(_on_mouse_exited)


func set_genealogy_data(nodes: Array, edges: Array) -> void:
	nodes_data = []
	for node in nodes:
		var data: Dictionary = node.duplicate()
		data["id"] = data.get("compound_id", data.get("id", data.get("node_id", "")))
		data["extant"] = int(data.get("current_abundance", 0)) > 0
		nodes_data.append(data)
	edges_data = edges
	_rebuild_dag_layout()
	queue_redraw()


func _rebuild_dag_layout() -> void:
	_node_positions.clear()
	_node_dict_by_id.clear()

	var gen_map: Dictionary = {}
	for n in nodes_data:
		if not (n is Dictionary):
			continue
		var nd: Dictionary = n as Dictionary
		var nid: String = str(nd.get("id", nd.get("entity_id", nd.get("name", ""))))
		if nid.is_empty():
			continue
		_node_dict_by_id[nid] = nd

		var gen: int = int(nd.get("generation", 0))
		if not gen_map.has(gen):
			gen_map[gen] = []
		(gen_map[gen] as Array).append(nid)

	var col_spacing: float = 140.0
	var row_spacing: float = 65.0
	var sorted_gens: Array = gen_map.keys()
	sorted_gens.sort()

	for g in sorted_gens:
		var ids: Array = gen_map[g]
		var total_h: float = float(ids.size()) * row_spacing
		var start_y: float = -total_h * 0.5 + row_spacing * 0.5
		var gx: float = float(g) * col_spacing

		for idx in range(ids.size()):
			var nid: String = ids[idx]
			var gy: float = start_y + float(idx) * row_spacing
			_node_positions[nid] = Vector2(gx, gy)


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb: InputEventMouseButton = event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			_zoom_at(mb.position, 1.18)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_zoom_at(mb.position, 1.0 / 1.18)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_MIDDLE or mb.button_index == MOUSE_BUTTON_RIGHT:
			_is_panning = mb.pressed
			if mb.pressed:
				_pan_start_mouse = mb.position
				_pan_start_offset = _pan_offset
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				var hit_id: String = _hit_test(mb.position)
				if not hit_id.is_empty():
					_selected_node_id = hit_id
					if _node_dict_by_id.has(hit_id):
						node_clicked.emit(_node_dict_by_id[hit_id])
					queue_redraw()
					accept_event()
				else:
					# Start pan on background left click
					_is_panning = true
					_pan_start_mouse = mb.position
					_pan_start_offset = _pan_offset
					accept_event()
			else:
				_is_panning = false

	elif event is InputEventMouseMotion:
		var mm: InputEventMouseMotion = event as InputEventMouseMotion
		if _is_panning:
			_pan_offset = _pan_start_offset + (mm.position - _pan_start_mouse)
			queue_redraw()
			accept_event()
		else:
			var prev_hover: String = _hovered_node_id
			_hovered_node_id = _hit_test(mm.position)
			if _hovered_node_id != prev_hover:
				queue_redraw()


func _zoom_at(mouse_pos: Vector2, factor: float) -> void:
	var prev_zoom: float = _zoom_scale
	_zoom_scale = clampf(_zoom_scale * factor, 0.45, 2.8)
	# Zoom centered around cursor
	_pan_offset = mouse_pos - (mouse_pos - _pan_offset) * (_zoom_scale / prev_zoom)
	queue_redraw()


func _on_mouse_exited() -> void:
	if not _hovered_node_id.is_empty():
		_hovered_node_id = ""
		queue_redraw()


func _hit_test(screen_pt: Vector2) -> String:
	var hit_radius: float = 18.0 * _zoom_scale
	for nid in _node_positions.keys():
		var graph_pos: Vector2 = _node_positions[nid]
		var s_pos: Vector2 = graph_pos * _zoom_scale + _pan_offset
		if screen_pt.distance_to(s_pos) <= hit_radius:
			return nid
	return ""


func _draw() -> void:
	var font: Font = ThemeDB.fallback_font

	# 1. Dark glass background
	var bg_col := Color(0.027, 0.051, 0.106, 0.88)
	draw_rect(Rect2(Vector2.ZERO, size), bg_col, true)
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.165, 0.286, 0.467, 0.5), false, 1.0)

	# 2. Generational vertical guides
	_draw_generation_guides(font)

	# 3. Lineage Edges (Glowing Bezier curves)
	_draw_lineage_edges()

	# 4. Generational Nodes
	_draw_nodes(font)


func _draw_generation_guides(font: Font) -> void:
	if font == null:
		return
	var col_spacing: float = 140.0
	for gen in range(12):
		var gx: float = float(gen) * col_spacing
		var sx: float = gx * _zoom_scale + _pan_offset.x
		if sx >= -50.0 and sx <= size.x + 50.0:
			draw_line(Vector2(sx, 0.0), Vector2(sx, size.y), Color(0.165, 0.286, 0.467, 0.15), 1.0)
			draw_string(font, Vector2(sx - 16.0, 16.0), "Gen %d" % gen, HORIZONTAL_ALIGNMENT_CENTER, 36, 9, Color(0.580, 0.686, 0.835, 0.55))


func _draw_lineage_edges() -> void:
	for edge in edges_data:
		if not (edge is Dictionary):
			continue
		var ed: Dictionary = edge as Dictionary
		var f_id: String = str(ed.get("from", ed.get("source", ed.get("parent", ""))))
		var t_id: String = str(ed.get("to", ed.get("target", ed.get("child", ""))))

		if not _node_positions.has(f_id) or not _node_positions.has(t_id):
			continue

		var p1: Vector2 = (_node_positions[f_id] as Vector2) * _zoom_scale + _pan_offset
		var p2: Vector2 = (_node_positions[t_id] as Vector2) * _zoom_scale + _pan_offset

		# Bezier S-curve control points
		var mid_x: float = (p1.x + p2.x) * 0.5
		var ctrl1: Vector2 = Vector2(mid_x, p1.y)
		var ctrl2: Vector2 = Vector2(mid_x, p2.y)

		# Build polyline for smooth glowing Bezier curve
		var pts := PackedVector2Array()
		var steps: int = 14
		for s in range(steps + 1):
			var t: float = float(s) / float(steps)
			var pt: Vector2 = p1.bezier_interpolate(ctrl1, ctrl2, p2, t)
			pts.append(pt)

		var edge_col := Color(0.133, 0.827, 0.933, 0.35)
		draw_polyline(pts, edge_col, 3.5 * _zoom_scale, true)
		draw_polyline(pts, Color(0.204, 0.827, 0.600, 0.85), 1.5 * _zoom_scale, true)


func _draw_nodes(font: Font) -> void:
	var r: float = 14.0 * _zoom_scale

	for nid in _node_positions.keys():
		var pos: Vector2 = (_node_positions[nid] as Vector2) * _zoom_scale + _pan_offset
		var nd: Dictionary = _node_dict_by_id.get(nid, {})
		var extant: bool = bool(nd.get("extant", true))
		var role: String = str(nd.get("role", "")).to_lower()
		var n_name: String = str(nd.get("name", nid))

		var is_hover: bool = (nid == _hovered_node_id)
		var is_sel: bool = (nid == _selected_node_id)

		# Outer glow halo
		if is_hover or is_sel:
			var halo_col := Color(0.133, 0.827, 0.933, 0.35)
			draw_circle(pos, r + 7.0 * _zoom_scale, halo_col)

		# Base circle fill
		var fill_col := Color(0.06, 0.12, 0.22, 0.95)
		if not extant:
			fill_col = Color(0.04, 0.07, 0.11, 0.75)
		draw_circle(pos, r, fill_col)

		# Perimeter border
		var border_col := Color(0.204, 0.827, 0.600, 0.9) if extant else Color(0.40, 0.48, 0.58, 0.6)
		if role.contains("antitumor"):
			border_col = Color(0.984, 0.749, 0.141, 0.95)
		elif role.contains("catalyst"):
			border_col = Color(0.204, 0.827, 0.600, 0.95)

		if is_sel:
			border_col = Color(0.133, 0.827, 0.933, 1.0)

		draw_arc(pos, r, 0, TAU, 24, border_col, 2.0 * _zoom_scale, true)

		# Extant / Extinct status dot inside
		var status_dot_col: Color = Color(0.204, 0.827, 0.600) if extant else Color(0.5, 0.55, 0.65)
		draw_circle(pos, 3.5 * _zoom_scale, status_dot_col)

		# Labels
		if font != null:
			var label_col := Color(0.918, 0.957, 1.0, 0.95) if extant else Color(0.580, 0.686, 0.835, 0.7)
			var f_size: int = clampi(int(9.0 * _zoom_scale), 7, 12)
			draw_string(font, Vector2(pos.x - 45.0, pos.y + r + 13.0 * _zoom_scale), n_name, HORIZONTAL_ALIGNMENT_CENTER, 90, f_size, label_col)

			# Extant vs Extinct badge
			var badge_txt: String = "EXTANT" if extant else "EXTINCT"
			var badge_col: Color = Color(0.204, 0.827, 0.600, 0.8) if extant else Color(0.50, 0.55, 0.65, 0.6)
			draw_string(font, Vector2(pos.x - 30.0, pos.y - r - 4.0 * _zoom_scale), badge_txt, HORIZONTAL_ALIGNMENT_CENTER, 60, clampi(f_size - 2, 6, 9), badge_col)
