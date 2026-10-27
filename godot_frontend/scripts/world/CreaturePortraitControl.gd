class_name CreaturePortraitControl
extends Control
## Centered Animated Portrait Control for EVOCURE.
## Renders a live 60 FPS CreatureRig2D or BioCellNode2D with a dark radial
## bioluminescent backdrop glow in both large inspection panels (110x110)
## and compact evolutionary lineage strips (44x44).

var is_compact: bool = false
var current_entity_data: Dictionary = {}

var _creature_rig: CreatureRig2D = null
var _cell_node: BioCellNode2D = null
var _glow_phase: float = 0.0
var _target_scale: float = 1.0


func _init() -> void:
	clip_contents = true
	custom_minimum_size = Vector2(44.0, 44.0)


func _ready() -> void:
	resized.connect(_on_resized)
	_ensure_rig_nodes()


func _ensure_rig_nodes() -> void:
	if _creature_rig == null:
		_creature_rig = CreatureRig2D.new()
		_creature_rig.name = "PortraitCreature"
		_creature_rig.visible = false
		add_child(_creature_rig)

	if _cell_node == null:
		_cell_node = BioCellNode2D.new()
		_cell_node.name = "PortraitCell"
		_cell_node.visible = false
		add_child(_cell_node)

	_update_positions()


func set_compact_mode(compact: bool) -> void:
	is_compact = compact
	_update_scale()
	queue_redraw()


func set_entity_data(data: Dictionary) -> void:
	current_entity_data = data
	_ensure_rig_nodes()

	if data.is_empty():
		if _creature_rig != null:
			_creature_rig.visible = false
		if _cell_node != null:
			_cell_node.visible = false
		queue_redraw()
		return

	if data.has("cell_type") or data.has("clone_id"):
		if _creature_rig != null:
			_creature_rig.visible = false
		if _cell_node != null:
			var c_copy: Dictionary = data.duplicate()
			c_copy.erase("x")
			c_copy.erase("y")
			_cell_node.visible = true
			_cell_node.setup_cell(c_copy)
	else:
		if _cell_node != null:
			_cell_node.visible = false
		if _creature_rig != null:
			var e_copy: Dictionary = data.duplicate()
			e_copy.erase("x")
			e_copy.erase("y")
			e_copy.erase("vx")
			e_copy.erase("vy")
			e_copy["r"] = 16.0
			_creature_rig.visible = true
			_creature_rig.setup_entity(e_copy)

	_update_scale()
	_update_positions()
	queue_redraw()


func _on_resized() -> void:
	_update_scale()
	_update_positions()
	queue_redraw()


func _update_scale() -> void:
	var min_dim: float = minf(size.x, size.y)
	if min_dim <= 4.0:
		min_dim = 42.0 if is_compact else 96.0

	if is_compact or min_dim <= 60.0:
		_target_scale = (min_dim / 96.0) * 1.15
	else:
		_target_scale = (min_dim / 96.0) * 1.08

	if _creature_rig != null:
		_creature_rig.scale = Vector2(_target_scale, _target_scale)
	if _cell_node != null:
		_cell_node.scale = Vector2(_target_scale, _target_scale)


func _update_positions() -> void:
	var eff_size: Vector2 = size
	if eff_size.x <= 4.0 or eff_size.y <= 4.0:
		eff_size = Vector2(42.0, 42.0) if is_compact else Vector2(96.0, 96.0)
	var center: Vector2 = eff_size * 0.5
	if _creature_rig != null:
		_creature_rig.position = center
		_creature_rig.set_target_pos(center)
	if _cell_node != null:
		_cell_node.position = center


func _process(delta: float) -> void:
	_glow_phase += delta * 2.2
	queue_redraw()


func _draw() -> void:
	var center: Vector2 = size * 0.5
	var r: float = minf(size.x, size.y) * 0.48

	if r <= 2.0:
		return

	# Dark radial bioluminescent base
	var bg_col: Color = Color(0.035, 0.065, 0.13, 0.92)
	draw_circle(center, r, bg_col)

	# Luminous radial gradient aura
	var pulse: float = sin(_glow_phase) * 0.5 + 0.5
	var aura_col: Color = Color(0.133, 0.827, 0.933, 0.12 + 0.08 * pulse)
	if current_entity_data.has("cell_type") or current_entity_data.has("clone_id"):
		aura_col = Color(0.957, 0.247, 0.369, 0.14 + 0.08 * pulse) # Crimson for tumor portrait

	draw_circle(center, r * 0.85, aura_col)
	aura_col.a *= 1.5
	draw_circle(center, r * 0.45, aura_col)

	# Luminous border ring
	var border_col: Color = Color(0.165, 0.286, 0.467, 0.75)
	if not current_entity_data.is_empty():
		border_col = Color(0.133, 0.827, 0.933, 0.85) if not aura_col.r > 0.8 else Color(0.957, 0.247, 0.369, 0.85)

	draw_arc(center, r - 1.0, 0, TAU, 32, border_col, 1.2 if is_compact else 1.8, true)
