shader_type canvas_item;

uniform vec4 env_tint : source_color = vec4(1.0, 1.0, 1.0, 1.0);
uniform float flow_speed : hint_range(0.0, 3.0) = 0.35;
uniform float distortion_strength : hint_range(0.0, 0.05) = 0.007;
uniform float vignette_intensity : hint_range(0.0, 1.0) = 0.35;
uniform float spore_intensity : hint_range(0.0, 2.0) = 0.70;
uniform vec4 spore_color : source_color = vec4(0.2, 0.85, 0.95, 0.75);

// Deterministic 2D hash for procedural bokeh spores
float hash21(vec2 p) {
	p = fract(p * vec2(234.34, 435.345));
	p += dot(p, p + 34.23);
	return fract(p.x * p.y);
}

vec2 hash22(vec2 p) {
	float n = sin(dot(p, vec2(41.0, 289.0)));
	return fract(vec2(262144.0, 32768.0) * n);
}

// Procedural bioluminescent floating bokeh spores
float bokeh_spores(vec2 uv, float time) {
	float spores = 0.0;
	// 2 distinct depth layers
	for (int layer = 0; layer < 2; layer++) {
		float scale = (layer == 0) ? 14.0 : 24.0;
		float speed = (layer == 0) ? 0.06 : 0.12;
		vec2 st = uv * scale;
		st.y += time * speed;
		st.x += sin(st.y * 0.45 + float(layer) * 1.5) * 0.35;

		vec2 i_st = floor(st);
		vec2 f_st = fract(st);

		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				vec2 neighbor = vec2(float(x), float(y));
				vec2 p = hash22(i_st + neighbor);
				vec2 diff = neighbor + p - f_st;
				float dist = length(diff);
				float radius = 0.10 + 0.16 * hash21(i_st + neighbor);
				float pulse = 0.65 + 0.35 * sin(time * 2.2 + hash21(i_st + neighbor) * 6.2831);
				float glow = smoothstep(radius, 0.0, dist) * pulse;
				spores += glow * (0.35 + 0.25 * float(layer));
			}
		}
	}
	return spores;
}

void fragment() {
	// Subtle fluid wave distortion across microscopic liquid
	float t = TIME * flow_speed;
	vec2 wave = vec2(
		sin(UV.y * 11.0 + t) + sin(UV.y * 23.0 - t * 0.6) * 0.45,
		cos(UV.x * 9.0 + t * 0.85) + cos(UV.x * 19.0 - t * 0.45) * 0.45
	) * distortion_strength;

	vec2 distorted_uv = UV + wave;
	vec4 base_color = texture(TEXTURE, distorted_uv);

	// Biophysical environmental preset tint (hypoxia, low pH, oxidative stress, bloodstream)
	base_color.rgb *= env_tint.rgb;

	// Floating bioluminescent spores (denser in abyssal fluid, softer in stromal tumor region)
	float spores = bokeh_spores(UV, TIME) * spore_intensity;
	float fluid_bias = smoothstep(0.95, 0.25, UV.x);
	base_color.rgb += spore_color.rgb * spores * fluid_bias * 0.45;

	// Microscopic lens vignette depth
	vec2 center_diff = (UV - vec2(0.5)) * vec2(1.25, 1.0);
	float vignette = 1.0 - dot(center_diff, center_diff) * vignette_intensity;
	vignette = clamp(vignette, 0.0, 1.0);
	base_color.rgb *= vignette;

	COLOR = base_color;
}
