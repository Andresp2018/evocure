class_name HeatmapGridControl
extends Control
## 2D PDE Scalar Field & Bliss/Loewe Combination Synergy Heatmap for EVOCURE.
## Renders interactive colored 2D grids, numeric cell values for matrices <= 10x10,
## continuous vertical colorbar, and mouse hover inspector readouts.

var chart_title: String = "PDE Substrate Concentration Field"
var palette_name: String = "viridis"
var grid_data: Array = []
var x_axis_labels: Array = []
var y_axis_labels: Array = []

var _hover_cell: Vector2i = Vector2i(-1, -1)
var _min_val: float = 0.0
var _max_val: float = 1.0

# Layout metrics
var _pad_top: float = 26.0
var _pad_bottom: float = 24.0
var _pad_left: float = 40.0
var _pad_right: float = 52.0 # Space for colorbar


func _init() -> void:
	custom_minimum_size = Vector2(240.0, 140.0)
	clip_contents = true
	mouse_filter = MOUSE_FILTER_STOP


func _ready() -> void:
	mouse_exited.connect(_on_mouse_exited)


func set_grid_data(title_text: String, grid_2d: Array, palette: String = "viridis", x_labels: Array = [], y_labels: Array = []) -> void:
	chart_title = title_text
	grid_data = grid_2d
	palette_name = palette
	x_axis_labels = x_labels
	y_axis_labels = y_labels

	_compute_extrema()
	queue_redraw()


func _compute_extrema() -> void:
	_min_val = 999999.0
	_max_val = -999999.0

	for row in grid_data:
		if not (row is Array):
			continue
		for val in (row as Array):
			var v: float = float(val)
			if v < _min_val:
				_min_val = v
			if v > _max_val:
				_max_val = v

	if _min_val >= _max_val:
		_min_val = 0.0
		_max_val = 1.0


func _get_grid_rect() -> Rect2:
	return Rect2(
		_pad_left,
		_pad_top,
		maxf(10.0, size.x - _pad_left - _pad_right),
		maxf(10.0, size.y - _pad_top - _pad_bottom)
	)


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		var mm: InputEventMouseMotion = event as InputEventMouseMotion
		var rect := _get_grid_rect()
		if rect.has_point(mm.position) and not grid_data.is_empty():
			var rows: int = grid_data.size()
			var cols: int = (grid_data[0] as Array).size() if rows > 0 and grid_data[0] is Array else 1
			var cell_w: float = rect.size.x / float(maxi(1, cols))
			var cell_h: float = rect.size.y / float(maxi(1, rows))

			var c: int = clampi(int((mm.position.x - rect.position.x) / cell_w), 0, cols - 1)
			var r: int = clampi(int((mm.position.y - rect.position.y) / cell_h), 0, rows - 1)
			if _hover_cell != Vector2i(c, r):
				_hover_cell = Vector2i(c, r)
				queue_redraw()
		else:
			if _hover_cell != Vector2i(-1, -1):
				_hover_cell = Vector2i(-1, -1)
				queue_redraw()


func _on_mouse_exited() -> void:
	if _hover_cell != Vector2i(-1, -1):
		_hover_cell = Vector2i(-1, -1)
		queue_redraw()


func _draw() -> void:
	var font: Font = ThemeDB.fallback_font

	# 1. Background
	var bg_col := Color(0.035, 0.065, 0.125, 0.88)
	draw_rect(Rect2(Vector2.ZERO, size), bg_col, true)
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.165, 0.286, 0.467, 0.45), false, 1.0)

	# 2. Title
	if font != null and not chart_title.is_empty():
		draw_string(font, Vector2(10.0, 17.0), chart_title, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.918, 0.957, 1.0, 0.95))

	if grid_data.is_empty():
		return

	var rows: int = grid_data.size()
	var cols: int = (grid_data[0] as Array).size() if rows > 0 and grid_data[0] is Array else 0
	if rows == 0 or cols == 0:
		return

	var rect := _get_grid_rect()
	var cell_w: float = rect.size.x / float(cols)
	var cell_h: float = rect.size.y / float(rows)
	var show_labels: bool = (rows <= 10 and cols <= 10)

	# 3. Draw Grid Cells
	for r in range(rows):
		var row_arr: Array = grid_data[r] as Array
		for c in range(cols):
			var val: float = float(row_arr[c]) if c < row_arr.size() else 0.0
			var norm: float = clampf((val - _min_val) / maxf(0.0001, _max_val - _min_val), 0.0, 1.0)
			var col: Color = _map_palette(norm, palette_name)
			var cell_rect := Rect2(rect.position.x + float(c) * cell_w, rect.position.y + float(r) * cell_h, cell_w, cell_h)

			draw_rect(cell_rect, col, true)
			# Subtle cell border for matrices
			if show_labels:
				draw_rect(cell_rect, Color(0.03, 0.06, 0.12, 0.4), false, 0.8)

			# Cell numeric label if <= 10x10
			if show_labels and font != null and cell_w >= 24.0 and cell_h >= 14.0:
				var txt_col: Color = Color.BLACK if norm > 0.65 else Color.WHITE
				var txt: String = "%+.2f" % val if val != 0.0 else "0.00"
				draw_string(font, Vector2(cell_rect.position.x + 2.0, cell_rect.position.y + cell_h * 0.65), txt, HORIZONTAL_ALIGNMENT_CENTER, int(cell_w - 4.0), 9, txt_col)

	# 4. Axis Labels if present
	if font != null:
		var label_col := Color(0.580, 0.686, 0.835, 0.85)
		for c in range(cols):
			if c < x_axis_labels.size():
				var lx: float = rect.position.x + float(c) * cell_w + cell_w * 0.5
				draw_string(font, Vector2(lx - 12.0, rect.end.y + 14.0), str(x_axis_labels[c]), HORIZONTAL_ALIGNMENT_CENTER, 24, 9, label_col)
		for r in range(rows):
			if r < y_axis_labels.size():
				var ly: float = rect.position.y + float(r) * cell_h + cell_h * 0.6
				draw_string(font, Vector2(4.0, ly), str(y_axis_labels[r]), HORIZONTAL_ALIGNMENT_RIGHT, 32, 9, label_col)

	# 5. Hover Cell Highlight
	if _hover_cell.x >= 0 and _hover_cell.x < cols and _hover_cell.y >= 0 and _hover_cell.y < rows:
		var h_rect := Rect2(
			rect.position.x + float(_hover_cell.x) * cell_w,
			rect.position.y + float(_hover_cell.y) * cell_h,
			cell_w,
			cell_h
		)
		draw_rect(h_rect, Color.WHITE, false, 1.8)

		# Floating hover tooltip
		if font != null:
			var h_val: float = float((grid_data[_hover_cell.y] as Array)[_hover_cell.x])
			var info: String = "[%d, %d]: %.3f" % [_hover_cell.x, _hover_cell.y, h_val]
			var t_pos: Vector2 = Vector2(clampf(h_rect.position.x - 20.0, 10.0, size.x - 90.0), maxf(h_rect.position.y - 14.0, _pad_top + 4.0))
			draw_rect(Rect2(t_pos.x - 4.0, t_pos.y - 10.0, 78.0, 14.0), Color(0.04, 0.08, 0.16, 0.95), true)
			draw_string(font, t_pos, info, HORIZONTAL_ALIGNMENT_LEFT, 78, 9, Color(0.133, 0.827, 0.933))

	# 6. Vertical Colorbar on the right
	_draw_colorbar(rect, font)


func _draw_colorbar(plot_rect: Rect2, font: Font) -> void:
	var cb_w: float = 12.0
	var cb_x: float = size.x - _pad_right + 12.0
	var cb_y: float = plot_rect.position.y
	var cb_h: float = plot_rect.size.y

	var steps: int = 24
	var step_h: float = cb_h / float(steps)

	for s in range(steps):
		var norm: float = 1.0 - (float(s) / float(steps))
		var col: Color = _map_palette(norm, palette_name)
		var seg_rect := Rect2(cb_x, cb_y + float(s) * step_h, cb_w, step_h + 0.5)
		draw_rect(seg_rect, col, true)

	draw_rect(Rect2(cb_x, cb_y, cb_w, cb_h), Color(0.165, 0.286, 0.467, 0.75), false, 1.0)

	if font != null:
		var label_col := Color(0.580, 0.686, 0.835, 0.85)
		draw_string(font, Vector2(cb_x + cb_w + 4.0, cb_y + 8.0), "%.1f" % _max_val, HORIZONTAL_ALIGNMENT_LEFT, 30, 8, label_col)
		draw_string(font, Vector2(cb_x + cb_w + 4.0, cb_y + cb_h * 0.5 + 4.0), "%.1f" % ((_max_val + _min_val) * 0.5), HORIZONTAL_ALIGNMENT_LEFT, 30, 8, label_col)
		draw_string(font, Vector2(cb_x + cb_w + 4.0, cb_y + cb_h), "%.1f" % _min_val, HORIZONTAL_ALIGNMENT_LEFT, 30, 8, label_col)


func _map_palette(norm: float, p_name: String) -> Color:
	var t: float = clampf(norm, 0.0, 1.0)
	match p_name:
		"viridis":
			# Viridis: purple (0.26, 0.00, 0.33) -> blue/teal (0.13, 0.57, 0.55) -> yellow (0.99, 0.90, 0.14)
			if t < 0.5:
				return Color(0.267, 0.004, 0.329).lerp(Color(0.125, 0.533, 0.553), t * 2.0)
			else:
				return Color(0.125, 0.533, 0.553).lerp(Color(0.992, 0.906, 0.145), (t - 0.5) * 2.0)
		"magma":
			# Magma: black/purple -> crimson -> orange -> pale yellow
			if t < 0.35:
				return Color(0.04, 0.03, 0.10).lerp(Color(0.45, 0.10, 0.42), t / 0.35)
			elif t < 0.7:
				return Color(0.45, 0.10, 0.42).lerp(Color(0.90, 0.28, 0.26), (t - 0.35) / 0.35)
			else:
				return Color(0.90, 0.28, 0.26).lerp(Color(0.99, 0.95, 0.65), (t - 0.7) / 0.3)
		"coolwarm":
			# Coolwarm: deep blue (0.23, 0.30, 0.75) -> white (0.86, 0.86, 0.86) -> deep red (0.70, 0.01, 0.15)
			if t < 0.5:
				return Color(0.23, 0.30, 0.75).lerp(Color(0.92, 0.92, 0.95), t * 2.0)
			else:
				return Color(0.92, 0.92, 0.95).lerp(Color(0.85, 0.15, 0.20), (t - 0.5) * 2.0)
		"bioluminescent":
			# Bioluminescent: abyss navy (0.03, 0.06, 0.14) -> electric cyan (0.13, 0.83, 0.93) -> emerald (0.20, 0.83, 0.60)
			if t < 0.5:
				return Color(0.03, 0.06, 0.14).lerp(Color(0.133, 0.827, 0.933), t * 2.0)
			else:
				return Color(0.133, 0.827, 0.933).lerp(Color(0.204, 0.827, 0.600), (t - 0.5) * 2.0)
		_:
			# Plasma fallback
			if t < 0.5:
				return Color(0.05, 0.03, 0.53).lerp(Color(0.80, 0.20, 0.50), t * 2.0)
			else:
				return Color(0.80, 0.20, 0.50).lerp(Color(0.94, 0.97, 0.13), (t - 0.5) * 2.0)
