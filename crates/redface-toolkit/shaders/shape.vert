#version 320 es
#define KIND_SHAKE      1u
#define KIND_DOT        2u
#define KIND_FACE_DISC  3u
#define KIND_FACE_GLYPH 4u
#define KIND_PULSE_RING 5u
#define TAU 6.283185307179586

layout(location = 0) in vec2 a_center;
layout(location = 1) in vec2 a_half_size;
layout(location = 2) in vec4 a_color;
layout(location = 3) in float a_radius;
layout(location = 4) in float a_inner_radius;
layout(location = 5) in float a_birth_time;
layout(location = 6) in uint a_kind;

uniform vec2 u_surface_size;
uniform float u_time;
uniform float u_shake_start;
uniform float u_face_toggled_at;
uniform float u_face_active;
uniform vec4 u_box_color;
uniform vec4 u_text_color;
uniform vec4 u_accent_color;

out vec2 v_local;
out vec2 v_half_size;
out vec4 v_color;
out vec2 v_radii;

float ease_out_cubic(float t) {
	float c = clamp(t, 0.0, 1.0);
	float m = 1.0 - c;
	return 1.0 - m * m * m;
}

void main() {
	vec2 corners[6] = vec2[6](
		vec2(-1.0, -1.0),
		vec2( 1.0, -1.0),
		vec2( 1.0,  1.0),
		vec2(-1.0, -1.0),
		vec2( 1.0,  1.0),
		vec2(-1.0,  1.0)
	);
	vec2 corner = corners[gl_VertexID];
	vec2 c = a_center;
	vec4 col = a_color;
	vec2 hs = a_half_size;
	float r = a_radius;
	float ri = a_inner_radius;

	// Password dots pop in over 150 ms with an ease-out-cubic scale/alpha.
	if (a_kind == KIND_DOT && a_birth_time >= 0.0) {
		float e = ease_out_cubic((u_time - a_birth_time) / 0.15);
		corner *= e;
		col.a *= e;
	}
	// Failed-auth shake: 400 ms decaying sine, 4 cycles, 14 px amplitude.
	if ((a_kind == KIND_SHAKE || a_kind == KIND_DOT) && u_shake_start >= 0.0) {
		float t = (u_time - u_shake_start) / 0.4;
		if (t >= 0.0 && t < 1.0) {
			c.x += 14.0 * sin(TAU * 4.0 * t) * (1.0 - t);
		}
	}
	// Face toggle: 200 ms crossfade towards the accent color.
	if (a_kind == KIND_FACE_DISC || a_kind == KIND_FACE_GLYPH) {
		float e = ease_out_cubic((u_time - u_face_toggled_at) / 0.2);
		float amount = 1.0 - e;
		if (u_face_active > 0.5) {
			amount = e;
		}
		if (a_kind == KIND_FACE_DISC) {
			col = vec4(mix(u_box_color.rgb, u_accent_color.rgb, amount * 0.35), col.a);
		} else {
			col = vec4(mix(u_text_color.rgb, u_accent_color.rgb, amount), col.a);
		}
	}
	// Pulse ring while face recognition is active: 1200 ms sine on the alpha and radius.
	if (a_kind == KIND_PULSE_RING) {
		if (u_face_active < 0.5) {
			col.a = 0.0;
		} else {
			float s = sin(TAU * (u_time - u_face_toggled_at) / 1.2);
			col.a *= 0.45 + 0.35 * s;
			float o = 6.0 + 2.0 * s;
			hs += vec2(o, o);
			r += o;
			ri += o;
		}
	}

	vec2 p = c + corner * hs;
	vec2 clip = vec2(
		p.x / u_surface_size.x * 2.0 - 1.0,
		1.0 - p.y / u_surface_size.y * 2.0
	);
	gl_Position = vec4(clip, 0.0, 1.0);
	v_local = corner * hs;
	v_half_size = hs;
	v_color = col;
	v_radii = vec2(r, ri);
}
