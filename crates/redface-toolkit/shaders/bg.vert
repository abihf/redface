#version 320 es
// Fullscreen triangle — no vertex attributes, position from gl_VertexID.
out vec2 v_uv;

void main() {
	float x = float((gl_VertexID << 1) & 2);
	float y = float(gl_VertexID & 2);
	gl_Position = vec4(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
	v_uv = vec2(x, y);
}
