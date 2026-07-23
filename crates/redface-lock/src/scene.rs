//! Shared rendering contract between the scene builder (`ui.rs`) and the
//! Vulkan renderer (`gpu.rs`). All types are plain data, `repr(C)` where they
//! cross into GPU buffers.

// Shape kinds (ShapeInstance::kind).
/// Part of the password row: shake offset is applied in the shader.
pub const SHAPE_SHAKE: u32 = 1;
/// Password dot: pops in from `birth_time`, also shakes.
pub const SHAPE_DOT: u32 = 2;
/// Face button disc: crossfades box_color -> accent (0.35 mix) on toggle.
pub const SHAPE_FACE_DISC: u32 = 3;
/// Face glyph parts: crossfade text_color -> accent on toggle.
pub const SHAPE_FACE_GLYPH: u32 = 4;
/// Pulse ring: alpha/radius sine while face_active is 1.
pub const SHAPE_PULSE_RING: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShapeInstance {
	pub center: [f32; 2],
	pub half_size: [f32; 2],
	pub color: [f32; 4],
	/// Outer corner radius in px; a full circle when it equals min(half_size).
	pub radius: f32,
	/// Inner radius for hollow rings in px; 0 = filled shape.
	pub inner_radius: f32,
	/// Seconds since app epoch; negative = no pop animation.
	pub birth_time: f32,
	pub kind: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextInstance {
	/// Top-left corner in px.
	pub pos: [f32; 2],
	/// Quad size in px.
	pub size: [f32; 2],
	/// Atlas UV of the top-left corner (0..1).
	pub uv_pos: [f32; 2],
	/// Atlas UV size (0..1).
	pub uv_size: [f32; 2],
	pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
	pub surface_size: [f32; 2],
	/// Background image pixel size; [0, 0] means solid `bg_color` instead.
	pub bg_image_size: [f32; 2],
	pub bg_color: [f32; 4],
	pub text_color: [f32; 4],
	pub box_color: [f32; 4],
	pub accent_color: [f32; 4],
	/// Seconds since app epoch.
	pub time: f32,
	/// Seconds since app epoch; negative = no shake.
	pub shake_start: f32,
	pub face_toggled_at: f32,
	/// 1.0 while face recognition is active, 0.0 otherwise.
	pub face_active: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
	pub shapes: Vec<ShapeInstance>,
	pub texts: Vec<TextInstance>,
}

/// Where one rasterized glyph lives in the atlas texture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasRect {
	pub x: u32,
	pub y: u32,
	pub w: u32,
	pub h: u32,
	/// Offset from the pen position to the glyph's top-left corner, in px.
	pub bearing: [f32; 2],
}

/// Side length of the square glyph-atlas texture (R8).
pub const ATLAS_SIZE: u32 = 2048;
