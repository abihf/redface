#version 320 es

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_size;
layout(location = 2) in vec2 a_uv_pos;
layout(location = 3) in vec2 a_uv_size;
layout(location = 4) in vec4 a_color;

uniform vec2 u_surface_size;

out vec2 v_uv;
out vec4 v_color;

void main() {
	vec2 corners[6] = vec2[6](
		vec2(0.0, 0.0),
		vec2(1.0, 0.0),
		vec2(1.0, 1.0),
		vec2(0.0, 0.0),
		vec2(1.0, 1.0),
		vec2(0.0, 1.0)
	);
	vec2 corner = corners[gl_VertexID];
	vec2 p = a_pos + corner * a_size;
	vec2 clip = vec2(
		p.x / u_surface_size.x * 2.0 - 1.0,
		1.0 - p.y / u_surface_size.y * 2.0
	);
	gl_Position = vec4(clip, 0.0, 1.0);
	v_uv = a_uv_pos + corner * a_uv_size;
	v_color = a_color;
}
