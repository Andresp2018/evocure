class_name ThemeBuilder
extends RefCounted
## Programmatic Bioluminescent Dark-Glass Scientific Theme for Godot 4.7.
## Matches the futuristic laboratory HUD aesthetic with cyan/emerald/violet neon accents.

const BG_ABYSS := Color(0.027, 0.051, 0.106, 1.0)       # #070D1B
const BG_PANEL := Color(0.051, 0.094, 0.180, 0.90)      # #0D182E
const BG_CARD := Color(0.075, 0.133, 0.247, 0.88)       # #13223F
const BG_CARD_HOVER := Color(0.106, 0.192, 0.345, 0.95) # #1B3158
const BG_CARD_ACTIVE := Color(0.082, 0.243, 0.420, 0.96)# #153E6B

const BORDER_SUBTLE := Color(0.165, 0.286, 0.467, 0.65)
const BORDER_CYAN := Color(0.133, 0.827, 0.933, 0.95)   # #22D3EE
const BORDER_EMERALD := Color(0.204, 0.827, 0.600, 0.95)# #34D399
const BORDER_VIOLET := Color(0.659, 0.333, 0.969, 0.95) # #A855F7
const BORDER_CRIMSON := Color(0.957, 0.247, 0.369, 0.95)# #F43F5E

const TEXT_PRIMARY := Color(0.918, 0.957, 1.0, 1.0)
const TEXT_MUTED := Color(0.580, 0.686, 0.835, 1.0)
const TEXT_CYAN := Color(0.220, 0.741, 0.973, 1.0)
const TEXT_EMERALD := Color(0.290, 0.871, 0.502, 1.0)
const TEXT_AMBER := Color(0.984, 0.749, 0.141, 1.0)
const TEXT_CORAL := Color(0.984, 0.443, 0.522, 1.0)


static func build_main_theme() -> Theme:
	var theme := Theme.new()
	theme.default_font_size = 13

	# Label
	theme.set_color("font_color", "Label", TEXT_PRIMARY)

	# PanelContainer
	theme.set_stylebox("panel", "PanelContainer", make_glass_box(BG_PANEL, BORDER_SUBTLE, 10, 1))

	# Button
	var btn_normal := make_glass_box(BG_CARD, BORDER_SUBTLE, 8, 1, 10, 7)
	var btn_hover := make_glass_box(BG_CARD_HOVER, BORDER_CYAN, 8, 1, 10, 7)
	var btn_pressed := make_glass_box(BG_CARD_ACTIVE, BORDER_CYAN, 8, 2, 10, 7)
	var btn_disabled := make_glass_box(Color(0.05, 0.08, 0.14, 0.55), Color(0.15, 0.22, 0.32, 0.35), 8, 1, 10, 7)

	theme.set_stylebox("normal", "Button", btn_normal)
	theme.set_stylebox("hover", "Button", btn_hover)
	theme.set_stylebox("pressed", "Button", btn_pressed)
	theme.set_stylebox("focus", "Button", btn_hover)
	theme.set_stylebox("disabled", "Button", btn_disabled)
	theme.set_color("font_color", "Button", TEXT_PRIMARY)
	theme.set_color("font_hover_color", "Button", Color.WHITE)
	theme.set_color("font_pressed_color", "Button", BORDER_CYAN)

	# LineEdit & TextEdit
	var input_normal := make_glass_box(Color(0.035, 0.068, 0.135, 0.95), BORDER_SUBTLE, 7, 1, 9, 6)
	var input_focus := make_glass_box(Color(0.045, 0.088, 0.175, 0.98), BORDER_CYAN, 7, 2, 9, 6)
	theme.set_stylebox("normal", "LineEdit", input_normal)
	theme.set_stylebox("focus", "LineEdit", input_focus)
	theme.set_color("font_color", "LineEdit", TEXT_PRIMARY)
	theme.set_color("font_placeholder_color", "LineEdit", TEXT_MUTED)

	theme.set_stylebox("normal", "TextEdit", input_normal)
	theme.set_stylebox("focus", "TextEdit", input_focus)
	theme.set_color("font_color", "TextEdit", TEXT_PRIMARY)

	# OptionButton & PopupMenu
	theme.set_stylebox("normal", "OptionButton", btn_normal)
	theme.set_stylebox("hover", "OptionButton", btn_hover)
	theme.set_stylebox("pressed", "OptionButton", btn_pressed)
	var popup_box := make_glass_box(Color(0.04, 0.08, 0.16, 0.98), BORDER_CYAN, 8, 1, 8, 8)
	theme.set_stylebox("panel", "PopupMenu", popup_box)
	theme.set_stylebox("hover", "PopupMenu", make_glass_box(BG_CARD_ACTIVE, BORDER_CYAN, 5, 1, 6, 4))
	theme.set_color("font_color", "PopupMenu", TEXT_PRIMARY)

	# ProgressBar
	var pb_bg := make_glass_box(Color(0.03, 0.055, 0.11, 0.95), Color(0.14, 0.23, 0.38, 0.5), 5, 1, 0, 0)
	var pb_fill := make_glass_box(BORDER_CYAN, BORDER_CYAN, 5, 0, 0, 0)
	theme.set_stylebox("background", "ProgressBar", pb_bg)
	theme.set_stylebox("fill", "ProgressBar", pb_fill)

	# ScrollBar
	var sb_bg := make_glass_box(Color(0.03, 0.05, 0.10, 0.5), Color.TRANSPARENT, 4, 0, 2, 2)
	var sb_grab := make_glass_box(Color(0.18, 0.34, 0.56, 0.8), Color.TRANSPARENT, 4, 0, 2, 2)
	theme.set_stylebox("scroll", "VScrollBar", sb_bg)
	theme.set_stylebox("grabber", "VScrollBar", sb_grab)
	theme.set_stylebox("grabber_highlight", "VScrollBar", make_glass_box(BORDER_CYAN, Color.TRANSPARENT, 4, 0, 2, 2))

	return theme


static func make_glass_box(
	bg: Color,
	border: Color,
	radius: int = 8,
	border_w: int = 1,
	pad_h: int = 10,
	pad_v: int = 8,
	glow_size: int = 0,
	glow_color: Color = Color(0.13, 0.82, 0.93, 0.22)
) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = bg
	sb.border_color = border
	sb.set_border_width_all(border_w)
	sb.set_corner_radius_all(radius)
	sb.content_margin_left = pad_h
	sb.content_margin_right = pad_h
	sb.content_margin_top = pad_v
	sb.content_margin_bottom = pad_v
	if glow_size > 0:
		sb.shadow_color = glow_color
		sb.shadow_size = glow_size
	return sb


static func style_accent_button(btn: Button, bg: Color, border: Color, glow_col: Color) -> void:
	btn.add_theme_stylebox_override("normal", make_glass_box(bg, border, 9, 1, 14, 8, 6, glow_col))
	btn.add_theme_stylebox_override("hover", make_glass_box(bg.lightened(0.14), border.lightened(0.2), 9, 2, 14, 8, 10, glow_col))
	btn.add_theme_stylebox_override("pressed", make_glass_box(bg.darkened(0.10), Color.WHITE, 9, 2, 14, 8, 8, glow_col))


static func fill_rect(ctrl: Control) -> void:
	ctrl.anchor_left = 0.0
	ctrl.anchor_top = 0.0
	ctrl.anchor_right = 1.0
	ctrl.anchor_bottom = 1.0
	ctrl.offset_left = 0.0
	ctrl.offset_top = 0.0
	ctrl.offset_right = 0.0
	ctrl.offset_bottom = 0.0
	ctrl.grow_horizontal = Control.GROW_DIRECTION_BOTH
	ctrl.grow_vertical = Control.GROW_DIRECTION_BOTH
