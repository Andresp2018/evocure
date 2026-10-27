class_name MinimapControl
extends Control
## Real-time Microscopic Minimap for EVOCURE.
## Displays live colored dots for chemical creatures, crimson cluster for tumor cells,
## and an interactive camera viewport rectangle supporting click-and-drag navigation.

signal camera_jump_requested(world_pos: Vector2)

var _creatures: Array = []
var _tumor_cells: Array = []
var _camera_rect: Rect2 = Rect2(0, 0, 1280, 720)
var _world_bounds: Rect2 = Rect2(0, 0, 1280, 720)

var _is_dragging: bool = false
var _inner_margin: float = 4.0


func _init() -> void:
	custom_minimum_size = Vector2(176.0, 108.0)
	mouse_filter = MOUSE_FILTER_STOP


func update_minimap(creatures: Array, tumor_cells: Array, camera_rect: Rect2, world_bounds: Rect2) -> void:
	_creatures = creatures
	_tumor_cells = tumor_cells
	_camera_rect = camera_rect
	if world_bounds.size.x > 10.0 and world_bounds.size.y > 10.0:
		_world_bounds = world_bounds
	queue_redraw()


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb: InputEventMouseButton = event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			_is_dragging = mb.pressed
			if mb.pressed:
				_handle_minimap_click(mb.position)
				accept_event()
	elif event is InputEventMouseMotion:
		var mm: InputEventMouseMotion = event as InputEventMouseMotion
		if _is_dragging:
			_handle_minimap_click(mm.position)
			accept_event()


func _handle_minimap_click(local_pos: Vector2) -> void:
	var inner := _get_inner_rect()
	if inner.size.x <= 1.0 or inner.size.y <= 1.0:
		return

	var norm_x: float = clampf((local_pos.x - inner.position.x) / inner.size.x, 0.0, 1.0)
	var norm_y: float = clampf((local_pos.y - inner.position.y) / inner.size.y, 0.0, 1.0)

	var world_x: float = _world_bounds.position.x + norm_x * _world_bounds.size.x
	var world_y: float = _world_bounds.position.y + norm_y * _world_bounds.size.y

	camera_jump_requested.emit(Vector2(world_x, world_y))


func _get_inner_rect() -> Rect2:
	return Rect2(_inner_margin, _inner_margin, size.x - _inner_margin * 2.0, size.y - _inner_margin * 2.0)


func _world_to_minimap(world_pt: Vector2, inner: Rect2) -> Vector2:
	var norm_x: float = (world_pt.x - _world_bounds.position.x) / _world_bounds.size.x
	var norm_y: float = (world_pt.y - _world_bounds.position.y) / _world_bounds.size.y
	return Vector2(
		inner.position.x + norm_x * inner.size.x,
		inner.position.y + norm_y * inner.size.y
	)


func _draw() -> void:
	var inner := _get_inner_rect()

	# 1. Dark glass frame
	var bg_col := Color(0.027, 0.051, 0.106, 0.88)
	draw_rect(Rect2(Vector2.ZERO, size), bg_col, true)

	# Subtle microscopic niche split guide (70% fluid left, 30% stroma right)
	var split_x: float = inner.position.x + inner.size.x * 0.70
	var niche_col := Color(0.957, 0.247, 0.369, 0.08)
	draw_rect(Rect2(split_x, inner.position.y, inner.end.x - split_x, inner.size.y), niche_col, true)

	# 2. Glowing cyan/neon border
	var border_col := Color(0.133, 0.827, 0.933, 0.75)
	draw_rect(Rect2(Vector2.ZERO, size), border_col, false, 1.2)

	# 3. Tumor cells in the stromal tissue niche
	for cell_item in _tumor_cells:
		var pos: Vector2 = Vector2.ZERO
		var is_res: bool = false
		if cell_item is Dictionary:
			var d: Dictionary = cell_item as Dictionary
			pos = Vector2(float(d.get("x", 0.0)), float(d.get("y", 0.0)))
			is_res = bool(d.get("is_resistant", false))
		elif cell_item is Node2D:
			var n: Node2D = cell_item as Node2D
			pos = n.position
			if n.has_method("setup_cell"):
				is_res = n.get("is_resistant")

		var m_pos: Vector2 = _world_to_minimap(pos, inner)
		if inner.has_point(m_pos):
			var col: Color = Color(0.659, 0.333, 0.969, 0.95) if is_res else Color(0.957, 0.247, 0.369, 0.9)
			draw_circle(m_pos, 2.2, col)

	# 4. Chemical creatures
	for cre_item in _creatures:
		var pos: Vector2 = Vector2.ZERO
		var col: Color = Color(0.133, 0.827, 0.933, 0.9)
		if cre_item is Dictionary:
			var d: Dictionary = cre_item as Dictionary
			pos = Vector2(float(d.get("x", 0.0)), float(d.get("y", 0.0)))
			var role: String = str(d.get("ecological_role", "")).to_lower()
			var anti: float = float(d.get("antitumor_norm", 0.0))
			if anti > 0.6:
				col = Color(0.984, 0.749, 0.141) # Amber / Antitumor
			elif role.contains("catalyst"):
				col = Color(0.204, 0.827, 0.600) # Emerald
			elif role.contains("degrad"):
				col = Color(0.976, 0.451, 0.086) # Flame
		elif cre_item is Node2D:
			var n: Node2D = cre_item as Node2D
			pos = n.position
			if "primary_color" in n:
				col = n.get("primary_color")

		var m_pos: Vector2 = _world_to_minimap(pos, inner)
		if inner.has_point(m_pos):
			draw_circle(m_pos, 1.8, col)

	# 5. Interactive Camera Viewport Rectangle
	var cam_min: Vector2 = _world_to_minimap(_camera_rect.position, inner)
	var cam_max: Vector2 = _world_to_minimap(_camera_rect.end, inner)
	var cam_rect := Rect2(
		clampf(cam_min.x, inner.position.x, inner.end.x),
		clampf(cam_min.y, inner.position.y, inner.end.y),
		clampf(cam_max.x - cam_min.x, 8.0, inner.size.x),
		clampf(cam_max.y - cam_min.y, 6.0, inner.size.y)
	)

	# Camera fill & border
	draw_rect(cam_rect, Color(1.0, 1.0, 1.0, 0.08), true)
	draw_rect(cam_rect, Color(1.0, 1.0, 1.0, 0.85), false, 1.0)

	# 6. Subtle corner watermark
	var font: Font = ThemeDB.fallback_font
	if font != null:
		draw_string(font, Vector2(inner.position.x + 3.0, inner.end.y - 3.0), "MINIMAP", HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.580, 0.686, 0.835, 0.55))
