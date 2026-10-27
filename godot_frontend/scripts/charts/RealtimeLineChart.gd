class_name RealtimeLineChart
extends Control
## Antialiased Real-time Scientific Line Chart for EVOCURE.
## Supports multi-series population dynamics with logarithmic or linear Y-axes,
## dynamic legends with live integer counts, and mean ± std confidence bands.

var chart_title: String = "Population History"
var use_log_scale: bool = false
var show_legend: bool = false

# Multi-series data
var _x_vals: Array = []
var x_axis_title: String = "Virtual years"
var _series_dict: Dictionary = {}
var _colors_dict: Dictionary = {}
var _latest_counts: Dictionary = {}

# Band series data (mean ± std)
var _is_band_mode: bool = false
var _band_mean: Array = []
var _band_std: Array = []
var _band_color: Color = Color(0.133, 0.827, 0.933, 1.0)

# Layout metrics
var _pad_top: float = 24.0
var _pad_bottom: float = 22.0
var _pad_left: float = 44.0
var _pad_right: float = 16.0


func _init() -> void:
	custom_minimum_size = Vector2(240.0, 110.0)
	clip_contents = true


func configure(title_text: String, log_scale: bool = false, with_legend: bool = false) -> void:
	chart_title = title_text
	use_log_scale = log_scale
	show_legend = with_legend
	_pad_right = 135.0 if show_legend else 16.0
	queue_redraw()


func set_multi_series(x_vals: Array, series_dict: Dictionary, colors_dict: Dictionary, latest_counts: Dictionary = {}) -> void:
	_is_band_mode = false
	_x_vals = x_vals
	_series_dict = series_dict
	_colors_dict = colors_dict
	_latest_counts = latest_counts
	queue_redraw()


func set_band_series(x_vals: Array, mean_vals: Array, std_vals: Array, line_color: Color) -> void:
	_is_band_mode = true
	_x_vals = x_vals
	_band_mean = mean_vals
	_band_std = std_vals
	_band_color = line_color
	queue_redraw()


func _get_plot_rect() -> Rect2:
	return Rect2(
		_pad_left,
		_pad_top,
		maxf(10.0, size.x - _pad_left - _pad_right),
		maxf(10.0, size.y - _pad_top - _pad_bottom)
	)


func _draw() -> void:
	var plot_rect := _get_plot_rect()
	var font: Font = ThemeDB.fallback_font

	# 1. Background dark panel
	var bg_col := Color(0.035, 0.065, 0.125, 0.85)
	draw_rect(Rect2(Vector2.ZERO, size), bg_col, true)
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.165, 0.286, 0.467, 0.45), false, 1.0)

	# 2. Chart Title
	if font != null and not chart_title.is_empty():
		draw_string(font, Vector2(10.0, 16.0), chart_title, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.918, 0.957, 1.0, 0.95))

	# 3. Y Range & Gridlines
	var y_min: float = 0.0
	var y_max: float = 1.0

	if use_log_scale:
		y_min = 1.0
		y_max = 1000.0
		# Find data peak
		for sname in _series_dict.keys():
			var arr: Array = _series_dict[sname]
			for v in arr:
				y_max = maxf(y_max, float(v))
		_draw_log_grid(plot_rect, font, y_max)
	else:
		if _is_band_mode:
			y_min = 0.0
			y_max = 1.0
		else:
			for sname in _series_dict.keys():
				var arr: Array = _series_dict[sname]
				for v in arr:
					y_max = maxf(y_max, float(v))
			y_max = maxf(1.0, y_max * 1.1)
		_draw_linear_grid(plot_rect, font, y_min, y_max)

	# 4. X-Axis Ticks & Label
	_draw_x_axis(plot_rect, font)

	# 5. Plot Data Series
	if _x_vals.size() >= 2:
		var x_min: float = float(_x_vals[0])
		var x_max: float = float(_x_vals[-1])
		if absf(x_max - x_min) < 0.001:
			x_max = x_min + 1.0

		if _is_band_mode:
			_draw_band_curve(plot_rect, x_min, x_max, y_min, y_max)
		else:
			_draw_multi_series_curves(plot_rect, x_min, x_max, y_min, y_max)

	# 6. Legend on the right
	if show_legend:
		_draw_legend(font)


func _draw_linear_grid(plot: Rect2, font: Font, y_min: float, y_max: float) -> void:
	var ticks := [0.0, 0.5, 1.0] if _is_band_mode else [0.0, y_max * 0.5, y_max]
	var grid_col := Color(0.165, 0.286, 0.467, 0.35)
	var label_col := Color(0.580, 0.686, 0.835, 0.85)

	for t in ticks:
		var norm_y: float = (t - y_min) / maxf(0.001, y_max - y_min)
		var py: float = plot.end.y - norm_y * plot.size.y
		draw_line(Vector2(plot.position.x, py), Vector2(plot.end.x, py), grid_col, 1.0)
		if font != null:
			var txt: String = "%.1f" % t if _is_band_mode else "%d" % int(t)
			draw_string(font, Vector2(6.0, py + 4.0), txt, HORIZONTAL_ALIGNMENT_RIGHT, 32, 9, label_col)


func _draw_log_grid(plot: Rect2, font: Font, y_max: float) -> void:
	var log_decades := [1.0, 10.0, 100.0, 1000.0]
	if y_max > 1000.0:
		log_decades.append(10000.0)

	var log_min: float = 0.0 # log10(1) = 0
	var log_max: float = log(maxf(10.0, y_max)) / log(10.0)
	var grid_col := Color(0.165, 0.286, 0.467, 0.35)
	var label_col := Color(0.580, 0.686, 0.835, 0.85)

	for d in log_decades:
		var ld: float = log(d) / log(10.0)
		if ld > log_max:
			continue
		var norm_y: float = (ld - log_min) / maxf(0.001, log_max - log_min)
		var py: float = plot.end.y - norm_y * plot.size.y
		draw_line(Vector2(plot.position.x, py), Vector2(plot.end.x, py), grid_col, 1.0)
		if font != null:
			var txt: String = "10^%d" % int(ld) if ld > 0 else "1"
			draw_string(font, Vector2(4.0, py + 4.0), txt, HORIZONTAL_ALIGNMENT_RIGHT, 34, 9, label_col)


func _draw_x_axis(plot: Rect2, font: Font) -> void:
	var axis_col := Color(0.20, 0.35, 0.55, 0.75)
	draw_line(Vector2(plot.position.x, plot.end.y), Vector2(plot.end.x, plot.end.y), axis_col, 1.2)

	if font != null and _x_vals.size() >= 2:
		var x0: float = float(_x_vals[0])
		var x1: float = float(_x_vals[-1])
		var label_col := Color(0.580, 0.686, 0.835, 0.85)
		draw_string(font, Vector2(plot.position.x, plot.end.y + 14.0), "%.1f" % x0, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, label_col)
		draw_string(font, Vector2(plot.end.x - 30.0, plot.end.y + 14.0), "%.1f" % x1, HORIZONTAL_ALIGNMENT_RIGHT, 30, 9, label_col)
		draw_string(font, Vector2((plot.position.x + plot.end.x) * 0.5 - 24.0, plot.end.y + 14.0), x_axis_title, HORIZONTAL_ALIGNMENT_CENTER, -1, 9, label_col)


func _draw_multi_series_curves(plot: Rect2, x_min: float, x_max: float, y_min: float, y_max: float) -> void:
	var log_min: float = 0.0
	var log_max: float = log(maxf(10.0, y_max)) / log(10.0)

	for sname in _series_dict.keys():
		var arr: Array = _series_dict[sname]
		if arr.size() < 2:
			continue

		var col: Color = _colors_dict.get(sname, Color(0.133, 0.827, 0.933))
		var pts := PackedVector2Array()

		var n: int = mini(_x_vals.size(), arr.size())
		for i in range(n):
			var x_norm: float = (float(_x_vals[i]) - x_min) / maxf(0.001, x_max - x_min)
			var px: float = plot.position.x + x_norm * plot.size.x
			var val: float = float(arr[i])
			var py: float = plot.end.y

			if use_log_scale:
				var ld: float = log(maxf(1.0, val)) / log(10.0)
				var y_norm: float = (ld - log_min) / maxf(0.001, log_max - log_min)
				py = plot.end.y - y_norm * plot.size.y
			else:
				var y_norm: float = (val - y_min) / maxf(0.001, y_max - y_min)
				py = plot.end.y - y_norm * plot.size.y

			pts.append(Vector2(px, py))

		# Glowing double-drawn polyline
		var halo_col: Color = col
		halo_col.a = 0.28
		draw_polyline(pts, halo_col, 4.0, true)
		draw_polyline(pts, col, 1.8, true)


func _draw_band_curve(plot: Rect2, x_min: float, x_max: float, y_min: float, y_max: float) -> void:
	var n: int = mini(_x_vals.size(), _band_mean.size())
	if n < 2:
		return

	var upper_pts := PackedVector2Array()
	var lower_pts := PackedVector2Array()
	var mean_pts := PackedVector2Array()

	for i in range(n):
		var x_norm: float = (float(_x_vals[i]) - x_min) / maxf(0.001, x_max - x_min)
		var px: float = plot.position.x + x_norm * plot.size.x
		var m: float = float(_band_mean[i])
		var s: float = float(_band_std[i]) if i < _band_std.size() else 0.05

		var u_val: float = clampf(m + s, y_min, y_max)
		var l_val: float = clampf(m - s, y_min, y_max)

		var py_m: float = plot.end.y - ((m - y_min) / maxf(0.001, y_max - y_min)) * plot.size.y
		var py_u: float = plot.end.y - ((u_val - y_min) / maxf(0.001, y_max - y_min)) * plot.size.y
		var py_l: float = plot.end.y - ((l_val - y_min) / maxf(0.001, y_max - y_min)) * plot.size.y

		mean_pts.append(Vector2(px, py_m))
		upper_pts.append(Vector2(px, py_u))
		lower_pts.append(Vector2(px, py_l))

	# Construct confidence band polygon (clockwise: upper left-to-right, then lower right-to-left)
	var band_poly := PackedVector2Array()
	band_poly.append_array(upper_pts)
	for i in range(lower_pts.size() - 1, -1, -1):
		band_poly.append(lower_pts[i])

	var band_fill_col: Color = _band_color
	band_fill_col.a = 0.22
	draw_colored_polygon(band_poly, band_fill_col)

	# Glowing mean polyline
	var halo_col: Color = _band_color
	halo_col.a = 0.35
	draw_polyline(mean_pts, halo_col, 4.0, true)
	draw_polyline(mean_pts, _band_color, 2.0, true)


func _draw_legend(font: Font) -> void:
	if font == null:
		return

	var start_x: float = size.x - _pad_right + 10.0
	var start_y: float = _pad_top + 4.0
	var line_h: float = 14.0

	var families := ["Catalysts", "Transporters", "Degraders", "Antitumor", "Structural", "Others"]
	for i in range(families.size()):
		var fname: String = families[i]
		var col: Color = _colors_dict.get(fname, Color(0.133, 0.827, 0.933))
		var cnt: int = int(_latest_counts.get(fname, 0))
		var y: float = start_y + float(i) * line_h

		# Dot
		draw_circle(Vector2(start_x, y), 3.0, col)
		# Label + live count
		var txt: String = "%s: %d" % [fname, cnt]
		draw_string(font, Vector2(start_x + 8.0, y + 3.0), txt, HORIZONTAL_ALIGNMENT_LEFT, 110, 9, Color(0.918, 0.957, 1.0, 0.88))
