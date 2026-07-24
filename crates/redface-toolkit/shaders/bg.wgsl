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
@group(0) @binding(1) var bg_tex: texture_2d<f32>;
@group(0) @binding(2) var bg_samp: sampler;

struct VsOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

// Fullscreen triangle: positions and uvs derived from the vertex index.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
	let x = f32((vi << 1u) & 2u);
	let y = f32(vi & 2u);
	var out: VsOut;
	out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
	out.uv = vec2<f32>(x, y);
	return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
	if (u.bg_image_size.x <= 0.0 || u.bg_image_size.y <= 0.0) {
		return u.bg_color;
	}
	// Cover fit: scale so the image covers the surface, crop the overflow
	// centered (same math as scale_cover in ui.rs).
	let scale = max(u.surface_size.x / u.bg_image_size.x, u.surface_size.y / u.bg_image_size.y);
	let shown = u.surface_size / scale;
	let origin = (u.bg_image_size - shown) * 0.5;
	let img_uv = (origin + in.uv * shown) / u.bg_image_size;
	return textureSample(bg_tex, bg_samp, img_uv);
}
