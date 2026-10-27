shader_type canvas_item;

uniform vec4 glow_color : source_color = vec4(0.133, 0.827, 0.933, 1.0);
uniform float selected_intensity : hint_range(0.0, 2.0) = 0.0;
uniform float hit_flash : hint_range(0.0, 1.0) = 0.0;
uniform float pulse_speed : hint_range(0.0, 10.0) = 2.5;
uniform bool glow_enabled = true;
uniform float inner_glow_boost : hint_range(0.0, 2.0) = 0.35;

void fragment() {
	vec4 col = texture(TEXTURE, UV);
	if (!glow_enabled) { COLOR = col; } else {
	vec2 ps = TEXTURE_PIXEL_SIZE;

	// Multi-sample neighbor alpha to compute rim/outer glow halo
	float alpha_max = col.a;
	float radius = 3.0 + selected_intensity * 3.5;

	vec2 offsets[8] = vec2[](
		vec2(-1.0,  0.0), vec2( 1.0,  0.0),
		vec2( 0.0, -1.0), vec2( 0.0,  1.0),
		vec2(-0.707, -0.707), vec2( 0.707, -0.707),
		vec2(-0.707,  0.707), vec2( 0.707,  0.707)
	);

	for (int i = 0; i < 8; i++) {
		float a = texture(TEXTURE, UV + offsets[i] * ps * radius).a;
		alpha_max = max(alpha_max, a);
	}
	float rim = clamp(alpha_max - col.a, 0.0, 1.0);

	// Bioluminescent breathing pulse
	float pulse = sin(TIME * pulse_speed) * 0.5 + 0.5;
	float aura_power = 0.45 + 0.35 * pulse + selected_intensity * 0.9;

	// Outer glow rim
	vec3 out_glow = glow_color.rgb * rim * aura_power;

	// Inner body bioluminescent vibrancy
	vec3 inner = col.rgb;
	inner = mix(inner, inner * glow_color.rgb * 1.6, inner_glow_boost * (pulse * 0.4 + 0.6));

	// Selection accent
	if (selected_intensity > 0.0) {
		inner = mix(inner, glow_color.rgb * 1.4, selected_intensity * 0.3);
	}

	// Antitumor / reaction hit flash
	if (hit_flash > 0.001) {
		inner = mix(inner, vec3(1.0, 1.0, 1.0), hit_flash);
	}

	// Composite final color and alpha
	float final_a = max(col.a, rim * glow_color.a * (0.65 + selected_intensity * 0.35));
	vec3 final_rgb = mix(out_glow, inner, col.a);
	if (col.a > 0.01) {
		final_rgb = inner + out_glow * 0.6;
	}

	COLOR = vec4(final_rgb, final_a);
	}
}
