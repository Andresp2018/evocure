class_name CreatureRig2D
extends Node2D
## Hybrid Studio-Sprite + Modular 2D Chemical Creature Rig for EVOCURE.
## Combines 8 iconic chemical archetypes with custom procedural modular creature assemblies,
## animated at 60 FPS with bioluminescent glow, generational traits, and interactive combat reactions.

const ARCHETYPE_SPRITES: Dictionary = {
	"water": "res://assets/sprites/creature_water.png",
	"methane": "res://assets/sprites/creature_methane.png",
	"benzene": "res://assets/sprites/creature_benzene.png",
	"glucose": "res://assets/sprites/creature_glucose.png",
	"phosphate": "res://assets/sprites/creature_phosphate.png",
	"lipid": "res://assets/sprites/creature_lipid.png",
	"dioxygen": "res://assets/sprites/creature_dioxygen.png",
	"amino": "res://assets/sprites/creature_amino.png",
}

const SHADER_RES_PATH: String = "res://assets/shaders/bioluminescent_entity.gdshader"

# Entity state
var entity_data: Dictionary = {}
var archetype: String = "modular"
var is_selected: bool = false
var target_pos: Vector2 = Vector2.ZERO
var velocity: Vector2 = Vector2.ZERO
var current_state: String = "IDLE" # "IDLE", "MOVE", "ATTACK", "SHIELD_IMPACT", "ATTACK_REACT"

# Generational and modular gene indices
var generation: int = 0
var head_id: int = 0
var body_id: int = 0
var legs_id: int = 0
var appendage_id: int = 0
var crystal_id: int = 0
var primary_color: Color = Color(0.133, 0.827, 0.933, 1.0)
var secondary_color: Color = Color(0.204, 0.827, 0.600, 1.0)

# Animation timers and phase accumulators
var _anim_time: float = 0.0
var _color_data: Dictionary = {}
var _last_palette: String = ""
var _attack_timer: float = 0.0
var _hit_flash_timer: float = 0.0
var _shield_impact_timer: float = 0.0
var _orbital_phase: float = 0.0
var _orbit_count: int = 3
var _dash_rotation: float = 0.0
var _base_scale: Vector2 = Vector2(1.0, 1.0)

# Nodes
var _sprite_node: Sprite2D = null
var _shader_mat: ShaderMaterial = null


func _ready() -> void:
	_setup_sprite_and_shader()
	if not entity_data.is_empty():
		setup_entity(entity_data)


func _setup_sprite_and_shader() -> void:
	if _sprite_node == null:
		_sprite_node = Sprite2D.new()
		_sprite_node.name = "ArchetypeSprite"
		_sprite_node.centered = true
		_sprite_node.scale = Vector2(0.22, 0.22)
		add_child(_sprite_node)

	var shader_res: Shader = load(SHADER_RES_PATH) as Shader
	if shader_res != null:
		_shader_mat = ShaderMaterial.new()
		_shader_mat.shader = shader_res
		_sprite_node.material = _shader_mat


func setup_entity(data: Dictionary) -> void:
	_color_data = data
	_last_palette = AppState.colorblind_palette
	entity_data = data
	archetype = _resolve_archetype(data)
	generation = int(data.get("generation", 0))

	# Modular genetic attributes
	head_id = int(data.get("head_id", _hash_int(data.get("entity_id", 0), 12, 1)))
	body_id = int(data.get("body_id", _hash_int(data.get("entity_id", 0), 12, 2)))
	legs_id = int(data.get("legs_id", _hash_int(data.get("entity_id", 0), 8, 3)))
	appendage_id = int(data.get("appendage_id", _hash_int(data.get("entity_id", 0), 8, 4)))
	crystal_id = int(data.get("crystal_id", _hash_int(data.get("entity_id", 0), 6, 5)))

	_resolve_colors(data)

	# Target position init
	if data.has("x") and data.has("y"):
		target_pos = Vector2(float(data["x"]), float(data["y"]))
		if position == Vector2.ZERO:
			position = target_pos

	if data.has("vx") and data.has("vy"):
		velocity = Vector2(float(data["vx"]), float(data["vy"]))

	_orbit_count = clampi(2 + (generation % 3), 2, 5)
	_base_scale = Vector2.ONE * clampf(float(data.get("r", data.get("radius", 5.0))) / 16.0, 0.24, 0.65)

	# Configure Sprite vs Procedural
	if ARCHETYPE_SPRITES.has(archetype):
		var tex_path: String = ARCHETYPE_SPRITES[archetype]
		var tex: Texture2D = load(tex_path) as Texture2D
		if _sprite_node != null:
			_sprite_node.texture = tex
			_sprite_node.visible = true
	else:
		if _sprite_node != null:
			_sprite_node.visible = false

	_update_shader_parameters()
	queue_redraw()


func _resolve_archetype(data: Dictionary) -> String:
	var raw_arch: String = str(data.get("archetype", "")).to_lower().strip_edges()
	if ARCHETYPE_SPRITES.has(raw_arch):
		return raw_arch

	var qname: String = str(data.get("name", "")).to_lower()
	var formula: String = str(data.get("formula", "")).to_lower()
	var query: String = qname + " " + formula

	if query.contains("h2o") or query.contains("water") or query.contains("aqua"):
		return "water"
	if query.contains("ch4") or query.contains("methane"):
		return "methane"
	if query.contains("c6h6") or query.contains("benzene"):
		return "benzene"
	if query.contains("glucose") or query.contains("c6h12o6") or query.contains("sugar"):
		return "glucose"
	if query.contains("phosphate") or query.contains("po4") or query.contains("atp"):
		return "phosphate"
	if query.contains("lipid") or query.contains("palmitic") or query.contains("membrane"):
		return "lipid"
	if query.contains("dioxygen") or query.contains("o2") or query.contains("oxygen"):
		return "dioxygen"
	if query.contains("amino") or query.contains("alanine") or query.contains("glycine"):
		return "amino"

	return "modular"


func _resolve_colors(data: Dictionary) -> void:
	if AppState.colorblind_palette != "bioluminescent":
		var palette: Array[Color] = [Color("0072b2"), Color("e69f00"), Color("56b4e9"), Color("009e73"), Color("cc79a7")]
		if AppState.colorblind_palette == "high_contrast":
			palette = [Color.WHITE, Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.GREEN]
		primary_color = palette[absi(str(data.get("family_id", data.get("fam", ""))).hash()) % palette.size()]
		secondary_color = primary_color.lightened(0.4)
		return
	if data.has("color"):
		var col_val: Variant = data["color"]
		if col_val is Color:
			primary_color = col_val
			secondary_color = primary_color.lightened(0.25)
			return
		elif col_val is String and not (col_val as String).is_empty():
			primary_color = Color.from_string(col_val, Color(0.133, 0.827, 0.933))
			secondary_color = primary_color.lightened(0.25)
			return

	var role: String = str(data.get("ecological_role", data.get("role", "neutral"))).to_lower()
	var kind: String = str(data.get("entity_kind", data.get("kind", "MOLECULE"))).to_upper()
	var anti_val: float = float(data.get("antitumor_norm", 0.0))

	if anti_val > 0.65 or kind == "THERAPEUTIC":
		primary_color = Color(0.984, 0.749, 0.141, 1.0) # Electric Amber Gold
		secondary_color = Color(0.957, 0.247, 0.369, 1.0) # Coral
	elif role.contains("catalyst") or kind == "CATALYST":
		primary_color = Color(0.204, 0.827, 0.600, 1.0) # Vibrant Emerald
		secondary_color = Color(0.325, 0.965, 0.725, 1.0)
	elif role.contains("cooperator") or role.contains("transport"):
		primary_color = Color(0.133, 0.827, 0.933, 1.0) # Electric Cyan
		secondary_color = Color(0.482, 0.914, 0.988, 1.0)
	elif role.contains("parasite") or role.contains("degrad"):
		primary_color = Color(0.976, 0.451, 0.086, 1.0) # Amber Flame
		secondary_color = Color(0.992, 0.733, 0.247, 1.0)
	elif role.contains("autotroph") or kind == "PROTOCELL":
		primary_color = Color(0.659, 0.333, 0.969, 1.0) # Bioluminescent Violet
		secondary_color = Color(0.812, 0.584, 0.984, 1.0)
	else:
		primary_color = Color(0.220, 0.741, 0.973, 1.0) # Ice Blue
		secondary_color = Color(0.565, 0.855, 0.984, 1.0)


func _update_shader_parameters() -> void:
	if _shader_mat == null:
		return
	_shader_mat.set_shader_parameter("glow_color", primary_color)
	_shader_mat.set_shader_parameter("selected_intensity", 1.0 if is_selected else 0.0)
	_shader_mat.set_shader_parameter("hit_flash", clampf(_hit_flash_timer, 0.0, 1.0))
	_shader_mat.set_shader_parameter("pulse_speed", 2.2 + float(generation) * 0.2)


func set_selected(selected: bool) -> void:
	is_selected = selected
	_update_shader_parameters()
	queue_redraw()


func set_target_pos(pos: Vector2) -> void:
	target_pos = pos


func trigger_attack_react() -> void:
	_attack_timer = 0.45
	_hit_flash_timer = 0.55
	current_state = "ATTACK_REACT"
	_update_shader_parameters()
	queue_redraw()


func trigger_shield_impact() -> void:
	_shield_impact_timer = 0.60
	current_state = "SHIELD_IMPACT"
	queue_redraw()


func trigger_attack() -> void:
	_attack_timer = 0.50
	current_state = "ATTACK"
	queue_redraw()


func _process(delta: float) -> void:
	if _last_palette != AppState.colorblind_palette:
		_last_palette = AppState.colorblind_palette
		_resolve_colors(_color_data)
		_update_shader_parameters()
	_anim_time += delta
	if _shader_mat != null:
		_shader_mat.set_shader_parameter("glow_enabled", AppState.bloom_enabled)
	_orbital_phase += delta * (1.8 + float(generation) * 0.15)
	_dash_rotation += delta * 1.5

	# Decay flash and combat timers
	if _hit_flash_timer > 0.0:
		_hit_flash_timer = maxf(0.0, _hit_flash_timer - delta * 2.5)
		if _shader_mat != null:
			_shader_mat.set_shader_parameter("hit_flash", _hit_flash_timer)

	if _attack_timer > 0.0:
		_attack_timer = maxf(0.0, _attack_timer - delta * 2.0)
		if _attack_timer == 0.0 and current_state in ["ATTACK", "ATTACK_REACT"]:
			current_state = "IDLE"

	if _shield_impact_timer > 0.0:
		_shield_impact_timer = maxf(0.0, _shield_impact_timer - delta * 1.8)
		if _shield_impact_timer == 0.0 and current_state == "SHIELD_IMPACT":
			current_state = "IDLE"

	# Smooth 60 FPS interpolation to target position
	var dist: float = position.distance_to(target_pos)
	if dist > 0.4:
		var lerp_factor: float = clampf(12.0 * delta, 0.0, 1.0)
		position = position.lerp(target_pos, lerp_factor)
		velocity = (target_pos - position) * 8.0
		current_state = "MOVE" if _attack_timer <= 0.0 else current_state
		# Velocity-aligned smooth tilt
		var target_rot: float = velocity.angle()
		rotation = lerp_angle(rotation, target_rot, clampf(9.0 * delta, 0.0, 1.0))
	else:
		if current_state == "MOVE":
			current_state = "IDLE"
		# Gentle sinusoidal bobbing & breathing in idle
		var idle_rot: float = sin(_anim_time * 1.5 + float(head_id)) * 0.08
		rotation = lerp_angle(rotation, idle_rot, clampf(4.0 * delta, 0.0, 1.0))

	# Sinusoidal breathing scale
	var breath: float = sin(_anim_time * 2.4 + float(body_id)) * 0.04
	var scale_mult: float = 1.0 + breath
	if _attack_timer > 0.0:
		scale_mult += sin(_attack_timer * PI * 2.0) * 0.18
	scale = _base_scale * scale_mult

	queue_redraw()


func _draw() -> void:
	if AppState.interface_mode == "LAB" and entity_data.get("structure") is Dictionary:
		if _sprite_node != null:
			_sprite_node.visible = false
		_draw_molecular_structure(entity_data["structure"])
		return
	if _sprite_node != null:
		_sprite_node.visible = ARCHETYPE_SPRITES.has(archetype)
	# 1. Bioluminescent outer aura
	if AppState.bloom_enabled:
		_draw_bioluminescent_aura()

	# 2. Generational accessories (underneath or dorsal)
	if generation > 0:
		_draw_generational_accessories()

	# 3. Modular procedural creature body if not using studio sprite
	if not ARCHETYPE_SPRITES.has(archetype):
		_draw_modular_creature()

	# 4. Floating orbital atoms/droplets
	_draw_orbital_particles()

	# 5. Combat shockwaves / attack flare
	if _attack_timer > 0.0:
		_draw_attack_flare()
	if _shield_impact_timer > 0.0:
		_draw_shield_impact()

	# 6. Selection reticle
	if is_selected:
		_draw_selection_ring()


func _draw_bioluminescent_aura() -> void:
	var pulse: float = 0.5 + 0.5 * sin(_anim_time * 3.0)
	var aura_radius: float = 30.0 + 4.0 * pulse
	var aura_col: Color = primary_color
	aura_col.a = 0.12 + 0.08 * pulse
	draw_circle(Vector2.ZERO, aura_radius, aura_col)
	var inner_col: Color = secondary_color
	inner_col.a = 0.18 + 0.06 * pulse
	draw_circle(Vector2.ZERO, aura_radius * 0.65, inner_col)


func _draw_orbital_particles() -> void:
	var base_orbit_r: float = 32.0 + float(generation) * 2.0
	for i in range(_orbit_count):
		var angle: float = _orbital_phase * (1.0 + float(i) * 0.35) + (float(i) * TAU / float(_orbit_count))
		var pt: Vector2 = Vector2(cos(angle), sin(angle) * 0.7) * base_orbit_r
		var pcol: Color = secondary_color
		pcol.a = 0.85
		draw_circle(pt, 2.6, pcol)
		# Trailing glowing ghost dot
		var trail_pt: Vector2 = Vector2(cos(angle - 0.25), sin(angle - 0.25) * 0.7) * base_orbit_r
		pcol.a = 0.35
		draw_circle(trail_pt, 1.8, pcol)


func _draw_generational_accessories() -> void:
	# Dorsal crystals and sensory antennae for evolved creatures
	if generation >= 1:
		# Dorsal crystal spire
		var crystal_col: Color = Color(0.85, 0.95, 1.0, 0.85)
		var pts: PackedVector2Array = [
			Vector2(-4.0, -20.0),
			Vector2(0.0, -32.0 - float(generation) * 2.0),
			Vector2(4.0, -20.0),
		]
		draw_colored_polygon(pts, crystal_col)
		draw_polyline(pts, primary_color, 1.2, true)

	if generation >= 2:
		# Sensory antennae whiskers
		var whisker_phase: float = sin(_anim_time * 4.0) * 0.2
		var left_tip: Vector2 = Vector2(24.0, -14.0).rotated(whisker_phase)
		var right_tip: Vector2 = Vector2(24.0, 14.0).rotated(-whisker_phase)
		draw_line(Vector2(10.0, -4.0), left_tip, secondary_color, 1.4, true)
		draw_line(Vector2(10.0, 4.0), right_tip, secondary_color, 1.4, true)
		draw_circle(left_tip, 2.2, Color.WHITE)
		draw_circle(right_tip, 2.2, Color.WHITE)

	if generation >= 3:
		# Catalytic orbital halo ring
		var halo_col: Color = primary_color
		halo_col.a = 0.45
		draw_arc(Vector2.ZERO, 38.0, _dash_rotation, _dash_rotation + PI * 0.75, 16, halo_col, 1.5, true)
		draw_arc(Vector2.ZERO, 38.0, _dash_rotation + PI, _dash_rotation + PI * 1.75, 16, halo_col, 1.5, true)


func _draw_modular_creature() -> void:
	# Deterministic modular procedural rendering
	var time: float = _anim_time

	# 1. Legs / Cilia / Tendrils (underneath)
	_draw_modular_legs(legs_id, time)

	# 2. Appendages (fins, claspers, plumes)
	_draw_modular_appendage(appendage_id, time)

	# 3. Body Core
	_draw_modular_body(body_id, time)

	# 4. Translucent Subsurface Core Organelle
	var core_col: Color = secondary_color
	core_col.a = 0.55 + 0.25 * sin(time * 3.5)
	draw_circle(Vector2(-2.0, 0.0), 9.0, core_col)
	draw_circle(Vector2(-2.0, 0.0), 4.5, Color.WHITE * 0.8)

	# 5. Head / Rostrum / Crest
	_draw_modular_head(head_id, time)

	# 6. Glossy Eyes with White Specular Highlights
	_draw_modular_eyes()

	# 7. Dorsal Crystal / Fullerene Core
	if crystal_id > 0:
		_draw_modular_crystal(crystal_id)


func _draw_modular_body(id: int, time: float) -> void:
	var col: Color = primary_color
	col.a = 0.88
	var edge_col: Color = secondary_color

	match id % 12:
		0: # Oval ellipsoid
			var pts := PackedVector2Array()
			for i in range(16):
				var a: float = float(i) * TAU / 16.0
				pts.append(Vector2(cos(a) * 22.0, sin(a) * 14.0))
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 1.6, true)
		1: # Segmented annelid (3 segments)
			draw_circle(Vector2(-12.0, 0.0), 10.0, col)
			draw_circle(Vector2(0.0, 0.0), 12.0, col)
			draw_circle(Vector2(12.0, 0.0), 9.0, col)
			draw_arc(Vector2(0.0, 0.0), 12.5, 0, TAU, 16, edge_col, 1.4, true)
		2: # 5-lobed radiolarian star
			var pts := PackedVector2Array()
			for i in range(20):
				var a: float = float(i) * TAU / 20.0
				var r: float = 20.0 if (i % 2 == 0) else 10.0
				pts.append(Vector2(cos(a) * r, sin(a) * r))
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 1.5, true)
		3: # Amoebic undulating droplet
			var pts := PackedVector2Array()
			for i in range(16):
				var a: float = float(i) * TAU / 16.0
				var ripple: float = sin(a * 4.0 + time * 3.0) * 3.5
				pts.append(Vector2(cos(a) * (18.0 + ripple), sin(a) * (14.0 + ripple)))
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 1.6, true)
		4: # Hexagonal chitinous carapace
			var pts := PackedVector2Array()
			for i in range(6):
				var a: float = float(i) * TAU / 6.0
				pts.append(Vector2(cos(a) * 19.0, sin(a) * 15.0))
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 2.0, true)
		5: # Diatom spindle / needle
			var pts := PackedVector2Array([
				Vector2(-24.0, 0.0), Vector2(-10.0, -12.0),
				Vector2(10.0, -12.0), Vector2(24.0, 0.0),
				Vector2(10.0, 12.0), Vector2(-10.0, 12.0)
			])
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 1.8, true)
		6: # Reniform / kidney bean
			var pts := PackedVector2Array()
			for i in range(16):
				var a: float = float(i) * TAU / 16.0
				var r: float = 18.0 - (sin(a) * 5.0 if sin(a) < 0 else 0.0)
				pts.append(Vector2(cos(a) * r, sin(a) * 13.0))
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 1.6, true)
		7: # Pyriform / teardrop flagellate
			var pts := PackedVector2Array([
				Vector2(-20.0, 0.0), Vector2(-6.0, -14.0),
				Vector2(14.0, -10.0), Vector2(22.0, 0.0),
				Vector2(14.0, 10.0), Vector2(-6.0, 14.0)
			])
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 1.6, true)
		8: # Rhomboid diamond
			var pts := PackedVector2Array([
				Vector2(-22.0, 0.0), Vector2(0.0, -17.0),
				Vector2(22.0, 0.0), Vector2(0.0, 17.0)
			])
			draw_colored_polygon(pts, col)
			draw_polyline(pts, edge_col, 1.8, true)
		9: # Diplococcus twin-capsule
			draw_circle(Vector2(-9.0, 0.0), 12.0, col)
			draw_circle(Vector2(9.0, 0.0), 12.0, col)
			draw_arc(Vector2(-9.0, 0.0), 12.0, 0, TAU, 16, edge_col, 1.4, true)
			draw_arc(Vector2(9.0, 0.0), 12.0, 0, TAU, 16, edge_col, 1.4, true)
		10: # Spirillum sinuous wave
			var pts := PackedVector2Array()
			for i in range(16):
				var x: float = -20.0 + float(i) * 2.8
				var y: float = sin(x * 0.15 + time * 3.0) * 8.0
				pts.append(Vector2(x, y))
			draw_polyline(pts, col, 10.0, true)
			draw_polyline(pts, edge_col, 2.0, true)
		11: # Torus annular ring
			draw_arc(Vector2.ZERO, 16.0, 0, TAU, 24, col, 8.0, true)
			draw_arc(Vector2.ZERO, 20.0, 0, TAU, 24, edge_col, 1.5, true)
			draw_arc(Vector2.ZERO, 12.0, 0, TAU, 24, edge_col, 1.5, true)


func _draw_modular_head(id: int, time: float) -> void:
	var hpos: Vector2 = Vector2(16.0, 0.0)
	var col: Color = primary_color.lightened(0.15)
	col.a = 0.92

	match id % 12:
		0: # Bulbous rostrum
			draw_circle(hpos, 8.0, col)
		1: # Crown of micro-cilia
			for i in range(5):
				var a: float = -0.6 + float(i) * 0.3 + sin(time * 6.0 + float(i)) * 0.12
				var tip: Vector2 = hpos + Vector2(cos(a), sin(a)) * 12.0
				draw_line(hpos, tip, secondary_color, 1.4, true)
		2: # Armored chitin cephalon
			var pts := PackedVector2Array([
				hpos + Vector2(0.0, -8.0),
				hpos + Vector2(10.0, 0.0),
				hpos + Vector2(0.0, 8.0)
			])
			draw_colored_polygon(pts, col)
		3: # Transparent dome
			var dome_col: Color = secondary_color
			dome_col.a = 0.45
			draw_circle(hpos + Vector2(2.0, 0.0), 9.0, dome_col)
			draw_arc(hpos + Vector2(2.0, 0.0), 9.0, -PI * 0.5, PI * 0.5, 12, Color.WHITE, 1.2, true)
		4: # Bifurcated sensory horns
			draw_line(hpos, hpos + Vector2(8.0, -9.0), secondary_color, 2.0, true)
			draw_line(hpos, hpos + Vector2(8.0, 9.0), secondary_color, 2.0, true)
		5: # Tapered proboscis / catalytic micro-syringe
			draw_line(hpos, hpos + Vector2(16.0, 0.0), Color.WHITE, 2.0, true)
			draw_circle(hpos + Vector2(16.0, 0.0), 2.0, primary_color)
		6: # Bioluminescent lantern stalk
			var tip: Vector2 = hpos + Vector2(10.0, -10.0 + sin(time * 3.0) * 3.0)
			draw_line(hpos, tip, secondary_color, 1.5, true)
			draw_circle(tip, 4.0, Color.WHITE)
			draw_circle(tip, 7.0, Color(primary_color.r, primary_color.g, primary_color.b, 0.4))
		_:
			draw_circle(hpos, 7.5, col)


func _draw_modular_legs(id: int, time: float) -> void:
	var leg_col: Color = secondary_color
	leg_col.a = 0.85

	match id % 8:
		0: # 4 waving cilia
			for i in range(4):
				var px: float = -12.0 + float(i) * 8.0
				var sway: float = sin(time * 5.0 + float(i)) * 4.0
				draw_line(Vector2(px, 8.0), Vector2(px - 4.0, 18.0 + sway), leg_col, 1.4, true)
				draw_line(Vector2(px, -8.0), Vector2(px - 4.0, -18.0 - sway), leg_col, 1.4, true)
		1: # 6 segmented micro-legs
			for i in range(3):
				var px: float = -8.0 + float(i) * 8.0
				var angle_off: float = sin(time * 6.0 + float(i) * 1.2) * 0.2
				# Bottom leg
				var knee_b: Vector2 = Vector2(px, 8.0) + Vector2(4.0, 8.0).rotated(angle_off)
				var foot_b: Vector2 = knee_b + Vector2(4.0, 6.0)
				draw_line(Vector2(px, 8.0), knee_b, leg_col, 1.6, true)
				draw_line(knee_b, foot_b, leg_col, 1.4, true)
				# Top leg
				var knee_t: Vector2 = Vector2(px, -8.0) + Vector2(4.0, -8.0).rotated(-angle_off)
				var foot_t: Vector2 = knee_t + Vector2(4.0, -6.0)
				draw_line(Vector2(px, -8.0), knee_t, leg_col, 1.6, true)
				draw_line(knee_t, foot_t, leg_col, 1.4, true)
		2: # Twin posterior swimming flagella
			for flag_i in range(2):
				var sign_y: float = -1.0 if flag_i == 0 else 1.0
				var pts := PackedVector2Array()
				for seg in range(7):
					var x: float = -16.0 - float(seg) * 5.5
					var y: float = (sign_y * 4.0) + sin(time * 8.0 - float(seg) * 0.8) * (3.0 + float(seg) * 1.2)
					pts.append(Vector2(x, y))
				draw_polyline(pts, leg_col, 1.8, true)
		3: # 8 radiating tentacular tendrils
			for i in range(8):
				var a: float = float(i) * TAU / 8.0
				var wave: float = sin(time * 4.0 + float(i)) * 0.25
				var tip: Vector2 = Vector2(cos(a + wave), sin(a + wave)) * 26.0
				draw_line(Vector2.ZERO, tip, leg_col, 1.3, true)
		_:
			# 4 paddle-like swimmerets
			for i in range(2):
				var px: float = -6.0 + float(i) * 12.0
				var swing: float = sin(time * 6.0 + float(i) * PI) * 5.0
				draw_line(Vector2(px, 10.0), Vector2(px + swing, 20.0), leg_col, 2.2, true)
				draw_line(Vector2(px, -10.0), Vector2(px + swing, -20.0), leg_col, 2.2, true)


func _draw_modular_appendage(id: int, time: float) -> void:
	match id % 8:
		0: # Dorsal luminous fin
			var pts := PackedVector2Array([
				Vector2(-10.0, -10.0),
				Vector2(0.0, -22.0 + sin(time * 4.0) * 2.0),
				Vector2(10.0, -10.0)
			])
			var col: Color = secondary_color
			col.a = 0.55
			draw_colored_polygon(pts, col)
		1: # Lateral claspers
			var claw_phase: float = sin(time * 3.5) * 0.2
			draw_arc(Vector2(8.0, -12.0), 8.0, -0.5 + claw_phase, 1.2 + claw_phase, 8, primary_color, 2.0, true)
			draw_arc(Vector2(8.0, 12.0), 8.0, -1.2 - claw_phase, 0.5 - claw_phase, 8, primary_color, 2.0, true)
		_:
			pass


func _draw_modular_eyes() -> void:
	var eye_pos1: Vector2 = Vector2(12.0, -5.0)
	var eye_pos2: Vector2 = Vector2(12.0, 5.0)
	# Sclera / eye background
	draw_circle(eye_pos1, 4.0, Color(0.04, 0.08, 0.16, 0.95))
	draw_circle(eye_pos2, 4.0, Color(0.04, 0.08, 0.16, 0.95))
	# Glowing Iris
	draw_circle(eye_pos1, 2.6, secondary_color)
	draw_circle(eye_pos2, 2.6, secondary_color)
	# Pupil
	draw_circle(eye_pos1, 1.2, Color.BLACK)
	draw_circle(eye_pos2, 1.2, Color.BLACK)
	# Glossy Specular Catchlights (Crisp White)
	draw_circle(eye_pos1 + Vector2(1.0, -1.0), 1.0, Color.WHITE)
	draw_circle(eye_pos2 + Vector2(1.0, -1.0), 1.0, Color.WHITE)


func _draw_modular_crystal(id: int) -> void:
	var pts := PackedVector2Array([
		Vector2(0.0, -18.0),
		Vector2(5.0, -24.0),
		Vector2(0.0, -30.0),
		Vector2(-5.0, -24.0)
	])
	var col: Color = Color(0.9, 0.98, 1.0, 0.85)
	draw_colored_polygon(pts, col)
	draw_polyline(pts, primary_color, 1.2, true)


func _draw_attack_flare() -> void:
	var flare_r: float = 38.0 + (1.0 - _attack_timer) * 22.0
	var flare_col: Color = primary_color
	flare_col.a = _attack_timer * 0.8
	draw_arc(Vector2.ZERO, flare_r, 0, TAU, 24, flare_col, 2.5, true)
	# Muzzle burst forward
	var tip: Vector2 = Vector2(flare_r + 12.0, 0.0)
	draw_line(Vector2(14.0, 0.0), tip, Color.WHITE, 3.0, true)


func _draw_shield_impact() -> void:
	var impact_r: float = 32.0 + (1.0 - _shield_impact_timer) * 16.0
	var shield_col: Color = Color(0.66, 0.33, 0.97, _shield_impact_timer * 0.8)
	# Hexagonal shield ripple
	var pts := PackedVector2Array()
	for i in range(6):
		var a: float = float(i) * TAU / 6.0
		pts.append(Vector2(cos(a) * impact_r, sin(a) * impact_r))
	draw_polyline(pts, shield_col, 2.2, true)


func _draw_selection_ring() -> void:
	var ring_col: Color = Color(0.133, 0.827, 0.933, 0.95)
	var radius: float = 38.0
	var segments: int = 8
	var arc_len: float = TAU / float(segments)
	# Rotating dashed cyan ring
	for i in range(segments):
		var start_a: float = _dash_rotation + float(i) * arc_len
		var end_a: float = start_a + arc_len * 0.55
		draw_arc(Vector2.ZERO, radius, start_a, end_a, 8, ring_col, 1.8, true)

	# Inner glowing concentric circle
	var inner_col: Color = ring_col
	inner_col.a = 0.35 + 0.2 * sin(_anim_time * 5.0)
	draw_arc(Vector2.ZERO, radius - 6.0, 0, TAU, 24, inner_col, 1.0, true)

	# 4 Corner reticle brackets
	var bracket_len: float = 6.0
	var corner_r: float = radius + 4.0
	for b in range(4):
		var corner_a: float = float(b) * (PI * 0.5) + (PI * 0.25)
		var center_pt: Vector2 = Vector2(cos(corner_a), sin(corner_a)) * corner_r
		draw_circle(center_pt, 2.0, ring_col)


func is_point_inside(pt: Vector2) -> bool:
	return global_position.distance_to(pt) <= 36.0 * scale.x


func _hash_int(seed_val: Variant, modulus: int, salt: int = 1) -> int:
	var h: int = absi(hash(str(seed_val) + "_" + str(salt)))
	return (h % maxi(1, modulus))

func _draw_molecular_structure(structure: Dictionary) -> void:
	var atoms: Array = structure.get("atoms", [])
	for bond in structure.get("bonds", []):
		if int(bond[0]) < atoms.size() and int(bond[1]) < atoms.size():
			var a: Array = atoms[int(bond[0])]
			var b: Array = atoms[int(bond[1])]
			var start := Vector2(float(a[0]), float(a[1]))
			var end := Vector2(float(b[0]), float(b[1]))
			var normal := (end - start).orthogonal().normalized() * 1.5
			var order: float = float(bond[2]) if bond.size() > 2 else 1.0
			if order >= 2:
				draw_line(start + normal, end + normal, Color("98aabb"), 1.0, true)
				draw_line(start - normal, end - normal, Color("98aabb"), 1.0, true)
				if order >= 3:
					draw_line(start, end, Color("98aabb"), 1.0, true)
			else:
				draw_line(start, end, Color("98aabb"), 1.5, true)
				if order > 1:
					draw_dashed_line(start + normal * 2, end + normal * 2, Color("98aabb"), 1.0, 2.0, true, true)
	for atom in atoms:
		var position := Vector2(float(atom[0]), float(atom[1]))
		var color: Color = {"O": Color("ed7777"), "N": Color("669ef5"), "S": Color("e4cc57"), "P": Color("f4a64f"), "F": Color("8ae0a0"), "Cl": Color("8ae0a0")}.get(str(atom[2]), Color("b3c2cf"))
		draw_circle(position, 4.5 if str(atom[2]) == "C" else 5.5, color)
	if is_selected:
		draw_arc(Vector2.ZERO, 44, 0, TAU, 32, Color.WHITE, 2.0, true)
