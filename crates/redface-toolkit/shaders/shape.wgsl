struct Uniforms {
	surface_size: vec2<f32>,
	bg_image_size: vec2<f32>,
	bg_color: vec4<f32>,
	text_color: vec4<f32>,
	box_color: vec4<f32>,
	accent_color: vec4<f32>,
	time: f32,
	shake_start: f32,
	face_toggled_at: f32,
	face_active: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

const KIND_SHAKE: u32 = 1u;
const KIND_DOT: u32 = 2u;
const KIND_FACE_DISC: u32 = 3u;
const KIND_FACE_GLYPH: u32 = 4u;
const KIND_PULSE_RING: u32 = 5u;
const TAU: f32 = 6.283185307179586;

fn ease_out_cubic(t: f32) -> f32 {
	let c = clamp(t, 0.0, 1.0);
	let m = 1.0 - c;
	return 1.0 - m * m * m;
}

struct VsOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) local: vec2<f32>,
	@location(1) half_size: vec2<f32>,
	@location(2) color: vec4<f32>,
	// x = outer corner radius, y = inner radius (0 = filled).
	@location(3) radii: vec2<f32>,
};

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@location(0) center: vec2<f32>,
	@location(1) half_size: vec2<f32>,
	@location(2) color: vec4<f32>,
	@location(3) radius: f32,
	@location(4) inner_radius: f32,
	@location(5) birth_time: f32,
	@location(6) kind: u32,
) -> VsOut {
	var corners = array<vec2<f32>, 6>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(1.0, -1.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(-1.0, 1.0),
	);
	var corner = corners[vi];
	var c = center;
	var col = color;
	var hs = half_size;
	var r = radius;
	var ri = inner_radius;

	// Password dots pop in over 150 ms with an ease-out-cubic scale/alpha.
	if (kind == KIND_DOT && birth_time >= 0.0) {
		let e = ease_out_cubic((u.time - birth_time) / 0.15);
		corner *= e;
		col.a *= e;
	}
	// Failed-auth shake: 400 ms decaying sine, 4 cycles, 14 px amplitude.
	if ((kind == KIND_SHAKE || kind == KIND_DOT) && u.shake_start >= 0.0) {
		let t = (u.time - u.shake_start) / 0.4;
		if (t >= 0.0 && t < 1.0) {
			c.x += 14.0 * sin(TAU * 4.0 * t) * (1.0 - t);
		}
	}
	// Face toggle: 200 ms crossfade towards the accent color.
	if (kind == KIND_FACE_DISC || kind == KIND_FACE_GLYPH) {
		let e = ease_out_cubic((u.time - u.face_toggled_at) / 0.2);
		var amount = 1.0 - e;
		if (u.face_active > 0.5) {
			amount = e;
		}
		if (kind == KIND_FACE_DISC) {
			col = vec4<f32>(mix(u.box_color.rgb, u.accent_color.rgb, amount * 0.35), col.a);
		} else {
			col = vec4<f32>(mix(u.text_color.rgb, u.accent_color.rgb, amount), col.a);
		}
	}
	// Pulse ring while face recognition is active: 1200 ms sine on the
	// alpha and on the radius (+6 px base, +/-2 px).
	if (kind == KIND_PULSE_RING) {
		if (u.face_active < 0.5) {
			col.a = 0.0;
		} else {
			let s = sin(TAU * (u.time - u.face_toggled_at) / 1.2);
			col.a *= 0.45 + 0.35 * s;
			let o = 6.0 + 2.0 * s;
			hs += vec2<f32>(o, o);
			r += o;
			ri += o;
		}
	}

	let p = c + corner * hs;
	let clip = vec2<f32>(
		p.x / u.surface_size.x * 2.0 - 1.0,
		1.0 - p.y / u.surface_size.y * 2.0,
	);
	var out: VsOut;
	out.pos = vec4<f32>(clip, 0.0, 1.0);
	out.local = corner * hs;
	out.half_size = hs;
	out.color = col;
	out.radii = vec2<f32>(r, ri);
	return out;
}

// Signed distance to a rounded box of half-size `b` and corner radius `r`;
// a full circle when r == min(b).
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
	let q = abs(p) - b + vec2<f32>(r, r);
	return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0, 0.0))) - r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
	let sd = sd_rounded_box(in.local, in.half_size, in.radii.x);
	var cov = smoothstep(0.5, -0.5, sd);
	if (in.radii.y > 0.0) {
		// Hollow ring of width (radius - inner_radius), measured on the SDF.
		let w = in.radii.x - in.radii.y;
		cov *= smoothstep(-w - 0.5, -w + 0.5, sd);
	}
	let a = in.color.a * cov;
	// Premultiplied alpha output.
	return vec4<f32>(in.color.rgb * a, a);
}
