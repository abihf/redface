#version 320 es
precision highp float;

in vec2 v_uv;
in vec4 v_color;
out vec4 out_color;

uniform sampler2D u_tex;

void main() {
	float cov = texture(u_tex, v_uv).r;
	float a = v_color.a * cov;
	// Premultiplied alpha output.
	out_color = vec4(v_color.rgb * a, a);
}
