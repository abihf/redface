#version 320 es
precision highp float;

in vec2 v_local;
in vec2 v_half_size;
in vec4 v_color;
in vec2 v_radii;
out vec4 out_color;

float sd_rounded_box(vec2 p, vec2 b, float r) {
	vec2 q = abs(p) - b + vec2(r, r);
	return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0, 0.0))) - r;
}

void main() {
	float sd = sd_rounded_box(v_local, v_half_size, v_radii.x);
	float cov = smoothstep(0.5, -0.5, sd);
	if (v_radii.y > 0.0) {
		// Hollow ring of width (radius - inner_radius), measured on the SDF.
		float w = v_radii.x - v_radii.y;
		cov *= smoothstep(-w - 0.5, -w + 0.5, sd);
	}
	float a = v_color.a * cov;
	// Premultiplied alpha output.
	out_color = vec4(v_color.rgb * a, a);
}
