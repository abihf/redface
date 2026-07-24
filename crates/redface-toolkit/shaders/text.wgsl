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
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;

struct VsOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@location(0) pos: vec2<f32>,
	@location(1) size: vec2<f32>,
	@location(2) uv_pos: vec2<f32>,
	@location(3) uv_size: vec2<f32>,
	@location(4) color: vec4<f32>,
) -> VsOut {
	var corners = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(0.0, 0.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(0.0, 1.0),
	);
	let corner = corners[vi];
	let p = pos + corner * size;
	let clip = vec2<f32>(
		p.x / u.surface_size.x * 2.0 - 1.0,
		1.0 - p.y / u.surface_size.y * 2.0,
	);
	var out: VsOut;
	out.pos = vec4<f32>(clip, 0.0, 1.0);
	out.uv = uv_pos + corner * uv_size;
	out.color = color;
	return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
	let cov = textureSample(atlas, atlas_samp, in.uv).r;
	let a = in.color.a * cov;
	// Premultiplied alpha output.
	return vec4<f32>(in.color.rgb * a, a);
}
