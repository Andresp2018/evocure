class_name BioCellNode2D
extends Node2D
## Cellular Oncology & Microbial Entity Node for EVOCURE Microscopic World.
## Renders healthy, tumor (PDAC/NSCLC), resistant clones (amethyst hexagonal shield),
## immune, CAF stroma, AMR bacteria, and MASLD hepatocytes with membrane breathing,
## fierce tumor aura, and beam hit reaction flashes.

const CELL_SPRITES: Dictionary = {
	"healthy": "res://assets/sprites/cell_healthy.png",
	"cell_healthy": "res://assets/sprites/cell_healthy.png",
	"tumor_pdac": "res://assets/sprites/cell_tumor_pdac.png",
	"cell_tumor_pdac": "res://assets/sprites/cell_tumor_pdac.png",
	"pdac": "res://assets/sprites/cell_tumor_pdac.png",
	"tumor_resistant": "res://assets/sprites/cell_tumor_resistant.png",
	"cell_tumor_resistant": "res://assets/sprites/cell_tumor_resistant.png",
	"resistant": "res://assets/sprites/cell_tumor_resistant.png",
	"immune": "res://assets/sprites/cell_immune.png",
	"cell_immune": "res://assets/sprites/cell_immune.png",
	"stroma_caf": "res://assets/sprites/cell_stroma_caf.png",
	"cell_stroma_caf": "res://assets/sprites/cell_stroma_caf.png",
	"caf": "res://assets/sprites/cell_stroma_caf.png",
	"tumor_nsclc": "res://assets/sprites/cell_tumor_nsclc.png",
	"cell_tumor_nsclc": "res://assets/sprites/cell_tumor_nsclc.png",
	"nsclc": "res://assets/sprites/cell_tumor_nsclc.png",
	"bacteria_amr": "res://assets/sprites/cell_bacteria_amr.png",
	"cell_bacteria_amr": "res://assets/sprites/cell_bacteria_amr.png",
	"amr": "res://assets/sprites/cell_bacteria_amr.png",
	"hepatocyte_masld": "res://assets/sprites/cell_hepatocyte_masld.png",
	"cell_hepatocyte_masld": "res://assets/sprites/cell_hepatocyte_masld.png",
	"masld": "res://assets/sprites/cell_hepatocyte_masld.png",
}

const SHADER_RES_PATH: String = "res://assets/shaders/bioluminescent_entity.gdshader"

var cell_data: Dictionary = {}
var cell_type: String = "tumor_pdac"
var clone_id: String = ""
var is_resistant: bool = false
var is_tumor: bool = false
var is_selected: bool = false
var viability: float = 1.0

# Animation timers
var _anim_time: float = 0.0
var _hit_flash: float = 0.0
var _shield_rot: float = 0.0
var _base_scale: Vector2 = Vector2(0.24, 0.24)
var _aura_color: Color = Color(0.957, 0.247, 0.369, 1.0) # Crimson tumor default

# Node references
var _sprite_node: Sprite2D = null
var _shader_mat: ShaderMaterial = null


func _ready() -> void:
	_setup_sprite()
	if not cell_data.is_empty():
		setup_cell(cell_data)


func _setup_sprite() -> void:
	if _sprite_node == null:
		_sprite_node = Sprite2D.new()
		_sprite_node.name = "CellSprite"
		_sprite_node.centered = true
		_sprite_node.scale = _base_scale
		add_child(_sprite_node)

	var shader_res: Shader = load(SHADER_RES_PATH) as Shader
	if shader_res != null:
		_shader_mat = ShaderMaterial.new()
		_shader_mat.shader = shader_res
		_sprite_node.material = _shader_mat


func setup_cell(data: Dictionary) -> void:
	cell_data = data
	cell_type = str(data.get("cell_type", data.get("type", "tumor_pdac"))).to_lower()
	clone_id = str(data.get("clone_id", ""))
	viability = clampf(float(data.get("viability", 1.0)), 0.0, 1.0)

	var clone_upper: String = clone_id.to_upper()
	is_resistant = bool(data.get("is_resistant", false)) or cell_type.contains("resistant") or clone_upper.contains("BASAL") or clone_upper.contains("PERSISTER")
	is_tumor = cell_type.contains("tumor") or cell_type.contains("pdac") or cell_type.contains("nsclc") or cell_type.contains("resistant")

	# Position initialization
	if data.has("x") and data.has("y"):
		position = Vector2(float(data["x"]), float(data["y"]))

	# Texture lookup
	var sprite_key: String = cell_type
	if not CELL_SPRITES.has(sprite_key):
		if is_resistant:
			sprite_key = "tumor_resistant"
		elif cell_type.contains("pdac"):
			sprite_key = "tumor_pdac"
		elif cell_type.contains("nsclc"):
			sprite_key = "tumor_nsclc"
		elif cell_type.contains("immune"):
			sprite_key = "immune"
		elif cell_type.contains("stroma") or cell_type.contains("caf"):
			sprite_key = "stroma_caf"
		elif cell_type.contains("bacteria") or cell_type.contains("amr"):
			sprite_key = "bacteria_amr"
		elif cell_type.contains("hepa") or cell_type.contains("masld"):
			sprite_key = "hepatocyte_masld"
		elif cell_type.contains("healthy"):
			sprite_key = "healthy"
		else:
			sprite_key = "tumor_pdac"

	var tex_path: String = CELL_SPRITES.get(sprite_key, "res://assets/sprites/cell_tumor_pdac.png")
	var tex: Texture2D = load(tex_path) as Texture2D
	if _sprite_node != null and tex != null:
		_sprite_node.texture = tex

	_resolve_aura_color()
	_update_shader_parameters()
	queue_redraw()


func _resolve_aura_color() -> void:
	if is_resistant:
		_aura_color = Color(0.659, 0.333, 0.969, 1.0) # Amethyst Violet
	elif is_tumor:
		_aura_color = Color(0.957, 0.247, 0.369, 1.0) # Fierce Crimson
	elif cell_type.contains("immune"):
		_aura_color = Color(0.204, 0.827, 0.600, 1.0) # Emerald Guard
	elif cell_type.contains("stroma") or cell_type.contains("caf"):
		_aura_color = Color(0.984, 0.749, 0.141, 1.0) # Amber Collagen
	elif cell_type.contains("bacteria") or cell_type.contains("amr"):
		_aura_color = Color(0.325, 0.898, 0.247, 1.0) # Biofilm Green
	elif cell_type.contains("hepa") or cell_type.contains("masld"):
		_aura_color = Color(0.965, 0.584, 0.235, 1.0) # Hepatic Ochre
	else:
		_aura_color = Color(0.220, 0.741, 0.973, 1.0) # Healthy Blue


func _update_shader_parameters() -> void:
	if _shader_mat == null:
		return
	_shader_mat.set_shader_parameter("glow_color", _aura_color)
	_shader_mat.set_shader_parameter("selected_intensity", 1.0 if is_selected else 0.0)
	_shader_mat.set_shader_parameter("hit_flash", clampf(_hit_flash, 0.0, 1.0))
	_shader_mat.set_shader_parameter("pulse_speed", 3.2 if is_tumor else 1.8)


func set_selected(selected: bool) -> void:
	is_selected = selected
	_update_shader_parameters()
	queue_redraw()


func trigger_beam_hit(intensity: float = 1.0) -> void:
	_hit_flash = clampf(intensity, 0.5, 1.0)
	_update_shader_parameters()
	queue_redraw()


func _process(delta: float) -> void:
	_anim_time += delta
	_shield_rot += delta * 0.8

	# Decay hit flash smoothly
	if _hit_flash > 0.0:
		_hit_flash = maxf(0.0, _hit_flash - delta * 2.8)
		if _shader_mat != null:
			_shader_mat.set_shader_parameter("hit_flash", _hit_flash)

	# Gentle membrane breathing (pulsing scale and subtle rotational wobble)
	var breath_speed: float = 3.2 if is_tumor else 1.8
	var breath_scale: float = 1.0 + sin(_anim_time * breath_speed) * (0.045 if is_tumor else 0.025)
	if _hit_flash > 0.0:
		breath_scale += _hit_flash * 0.12
	if _sprite_node != null:
		_sprite_node.scale = _base_scale * breath_scale
		_sprite_node.rotation = sin(_anim_time * 1.2) * 0.05

	queue_redraw()


func _draw() -> void:
	# 1. Outer bioluminescent aura
	_draw_cell_aura()

	# 2. Translucent Amethyst Hexagonal Shield for resistant clones
	if is_resistant:
		_draw_resistant_hex_shield()

	# 3. Hit burst rays when struck by antitumor beams
	if _hit_flash > 0.05:
		_draw_hit_flash_rays()

	# 4. Viability / Health Ring Indicator
	_draw_viability_arc()

	# 5. Selection Ring
	if is_selected:
		_draw_selection_ring()


func _draw_cell_aura() -> void:
	var pulse: float = sin(_anim_time * (3.5 if is_tumor else 1.8)) * 0.5 + 0.5
	var aura_r: float = 34.0 + (6.0 * pulse if is_tumor else 3.0 * pulse)
	var col: Color = _aura_color
	col.a = (0.24 + 0.16 * pulse) if is_tumor else (0.12 + 0.06 * pulse)
	draw_circle(Vector2.ZERO, aura_r, col)
	var inner_col: Color = col.lightened(0.2)
	inner_col.a = col.a * 1.4
	draw_circle(Vector2.ZERO, aura_r * 0.65, inner_col)


func _draw_resistant_hex_shield() -> void:
	var shield_r: float = 40.0 + sin(_anim_time * 2.0) * 1.5
	var shield_col: Color = Color(0.659, 0.333, 0.969, 0.65)
	var hex_pts := PackedVector2Array()
	for i in range(6):
		var a: float = _shield_rot + float(i) * TAU / 6.0
		hex_pts.append(Vector2(cos(a) * shield_r, sin(a) * shield_r))

	# Hexagon translucent fill
	var fill_col: Color = shield_col
	fill_col.a = 0.14 + 0.06 * sin(_anim_time * 3.0)
	draw_colored_polygon(hex_pts, fill_col)
	# Hexagon perimeter
	draw_polyline(hex_pts, shield_col, 2.0, true)

	# 6 shield node dots
	for pt in hex_pts:
		draw_circle(pt, 2.5, Color.WHITE)


func _draw_hit_flash_rays() -> void:
	var flash_col: Color = Color.WHITE
	flash_col.a = _hit_flash * 0.9
	var ray_count: int = 8
	var len_outer: float = 42.0 + _hit_flash * 18.0
	for i in range(ray_count):
		var a: float = float(i) * TAU / float(ray_count) + (_anim_time * 2.0)
		var p1: Vector2 = Vector2(cos(a), sin(a)) * 22.0
		var p2: Vector2 = Vector2(cos(a), sin(a)) * len_outer
		draw_line(p1, p2, flash_col, 2.0 * _hit_flash, true)


func _draw_viability_arc() -> void:
	if viability >= 0.99 and not is_selected and not is_tumor:
		return
	var arc_r: float = 33.0
	var health_col: Color = Color(0.204, 0.827, 0.600) # Emerald
	if viability < 0.4:
		health_col = Color(0.957, 0.247, 0.369) # Crimson
	elif viability < 0.7:
		health_col = Color(0.984, 0.749, 0.141) # Amber

	# Background dark arc
	draw_arc(Vector2.ZERO, arc_r, -PI * 0.8, PI * 0.8, 16, Color(0.04, 0.08, 0.16, 0.5), 2.5, true)
	# Health foreground arc
	var sweep: float = (PI * 1.6) * viability
	draw_arc(Vector2.ZERO, arc_r, -PI * 0.8, -PI * 0.8 + sweep, 16, health_col, 2.5, true)


func _draw_selection_ring() -> void:
	var ring_col: Color = Color(0.133, 0.827, 0.933, 0.95)
	var radius: float = 42.0
	var segments: int = 8
	var arc_len: float = TAU / float(segments)
	for i in range(segments):
		var start_a: float = _shield_rot * 1.5 + float(i) * arc_len
		var end_a: float = start_a + arc_len * 0.55
		draw_arc(Vector2.ZERO, radius, start_a, end_a, 8, ring_col, 2.0, true)

	draw_arc(Vector2.ZERO, radius - 6.0, 0, TAU, 24, Color(0.133, 0.827, 0.933, 0.4), 1.0, true)


func is_point_inside(pt: Vector2) -> bool:
	return global_position.distance_to(pt) <= 38.0
