//! Scene construction and hit-testing geometry for the OSD.
//!
//! All shapes are plain data consumed by the toolkit's Vulkan renderer; the
//! pulse animation runs in the shader off `Uniforms::face_active`/`time`, so
//! positions here are always the rest positions.

use redface_toolkit::scene::{ATLAS_SIZE, SHAPE_FACE_GLYPH, SHAPE_PULSE_RING, Scene, ShapeInstance, TextInstance};
use redface_toolkit::text::{Fonts, GlyphAtlas};

// #e6e6e6 / #26262e / #7aa2f7, straight alpha.
pub const TEXT_COLOR: [f32; 4] = [230.0 / 255.0, 230.0 / 255.0, 230.0 / 255.0, 1.0];
pub const BOX_COLOR: [f32; 4] = [38.0 / 255.0, 38.0 / 255.0, 46.0 / 255.0, 1.0];
pub const ACCENT_COLOR: [f32; 4] = [122.0 / 255.0, 162.0 / 255.0, 247.0 / 255.0, 1.0];

const PANEL_COLOR: [f32; 4] = [0.06, 0.06, 0.08, 0.85];

/// Screen geometry shared by the scene builder and pointer hit-testing.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
	/// (x, y, width, height) of the cancel button.
	pub cancel_button: (f32, f32, f32, f32),
	pub face_center: (f32, f32),
	pub face_radius: f32,
	pub panel_radius: f32,
	pub text_size: f32,
	pub scale: f32,
}

/// `width`/`height` are in the coordinate space being laid out (buffer pixels
/// when building the scene, logical surface coordinates for hit-testing) and
/// `scale` multiplies the fixed-size elements accordingly.
pub fn layout(width: u32, height: u32, scale: f32) -> Layout {
	let (w, h) = (width as f32, height as f32);
	let btn_w = 132.0 * scale;
	let btn_h = 38.0 * scale;
	let btn_y = h - btn_h - 16.0 * scale;
	Layout {
		cancel_button: ((w - btn_w) / 2.0, btn_y, btn_w, btn_h),
		face_center: (w / 2.0, btn_y / 2.0),
		face_radius: 30.0 * scale,
		panel_radius: 18.0 * scale,
		text_size: 16.0 * scale,
		scale,
	}
}

pub fn hit_cancel(layout: &Layout, x: f64, y: f64) -> bool {
	let (bx, by, bw, bh) = layout.cancel_button;
	x as f32 >= bx && x as f32 <= bx + bw && y as f32 >= by && y as f32 <= by + bh
}

fn mix(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
	[
		a[0] + (b[0] - a[0]) * t,
		a[1] + (b[1] - a[1]) * t,
		a[2] + (b[2] - a[2]) * t,
		a[3] + (b[3] - a[3]) * t,
	]
}

/// Shapes for one frame: translucent panel, pulsing face glyph, cancel
/// button. Kept font-free so it is testable without a system font.
pub fn build_shapes(lay: &Layout, hover_cancel: bool, width: u32, height: u32) -> Vec<ShapeInstance> {
	let (w, h) = (width as f32, height as f32);
	let (fx, fy) = lay.face_center;
	let fr = lay.face_radius;
	let mut shapes = Vec::new();

	// Dark translucent panel; the surface background itself stays fully
	// transparent so the rounded corners show through.
	shapes.push(ShapeInstance {
		center: [w / 2.0, h / 2.0],
		half_size: [w / 2.0, h / 2.0],
		color: PANEL_COLOR,
		radius: lay.panel_radius,
		inner_radius: 0.0,
		birth_time: -1.0,
		kind: 0,
	});

	// Pulse ring: the shader animates alpha/radius while face_active is 1.
	let ring_radius = fr + 6.0 * lay.scale;
	shapes.push(ShapeInstance {
		center: [fx, fy],
		half_size: [ring_radius, ring_radius],
		color: [ACCENT_COLOR[0], ACCENT_COLOR[1], ACCENT_COLOR[2], 0.45],
		radius: ring_radius,
		inner_radius: (ring_radius - 3.0 * lay.scale).max(0.0),
		birth_time: -1.0,
		kind: SHAPE_PULSE_RING,
	});

	// Face glyph: hollow head circle plus two eyes.
	let glyph_color = [TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2], 0.9];
	let scale = lay.scale;
	let head_radius = fr * 0.42;
	shapes.push(ShapeInstance {
		center: [fx, fy],
		half_size: [head_radius, head_radius],
		color: glyph_color,
		radius: head_radius,
		inner_radius: (head_radius - 3.0 * scale).max(0.0),
		birth_time: -1.0,
		kind: SHAPE_FACE_GLYPH,
	});
	for side in [-1.0f32, 1.0] {
		let eye_radius = fr * 0.055;
		shapes.push(ShapeInstance {
			center: [fx + side * fr * 0.18, fy - fr * 0.08],
			half_size: [eye_radius, eye_radius],
			color: glyph_color,
			radius: eye_radius,
			inner_radius: 0.0,
			birth_time: -1.0,
			kind: SHAPE_FACE_GLYPH,
		});
	}

	// Cancel button; accent-tinted while hovered.
	let (bx, by, bw, bh) = lay.cancel_button;
	let button_color = if hover_cancel {
		mix(BOX_COLOR, ACCENT_COLOR, 0.35)
	} else {
		BOX_COLOR
	};
	shapes.push(ShapeInstance {
		center: [bx + bw / 2.0, by + bh / 2.0],
		half_size: [bw / 2.0, bh / 2.0],
		color: [button_color[0], button_color[1], button_color[2], 0.95],
		radius: 10.0 * scale,
		inner_radius: 0.0,
		birth_time: -1.0,
		kind: 0,
	});
	shapes
}

/// Emits the quads of one text pass (per glyph) into `scene.texts`.
#[allow(clippy::too_many_arguments)]
fn push_text_pass(
	scene: &mut Scene,
	fonts: &Fonts,
	atlas: &mut GlyphAtlas,
	size: f32,
	mut x: f32,
	baseline: f32,
	text: &str,
	color: [f32; 4],
) {
	let mut prev = None;
	for c in text.chars() {
		if let Some(prev) = prev {
			x += fonts.kern(prev, c, size);
		}
		match atlas.ensure(fonts, c, size) {
			Some((rect, advance)) => {
				let atlas_size = ATLAS_SIZE as f32;
				scene.texts.push(TextInstance {
					pos: [x + rect.bearing[0], baseline + rect.bearing[1]],
					size: [rect.w as f32, rect.h as f32],
					uv_pos: [rect.x as f32 / atlas_size, rect.y as f32 / atlas_size],
					uv_size: [rect.w as f32 / atlas_size, rect.h as f32 / atlas_size],
					color,
				});
				x += advance;
			}
			None => x += fonts.h_advance(c, size),
		}
		prev = Some(c);
	}
}

/// Emits `text` centered at (center_x, center_y): the drop shadow pass first
/// (keeps the text legible), then the text.
#[allow(clippy::too_many_arguments)]
fn push_text(
	scene: &mut Scene,
	fonts: &Fonts,
	atlas: &mut GlyphAtlas,
	size: f32,
	center_x: f32,
	center_y: f32,
	text: &str,
	color: [f32; 4],
) {
	let width = fonts.text_width(text, size);
	let baseline = center_y + (fonts.ascent(size) + fonts.descent(size)) / 2.0;
	let x = center_x - width / 2.0;
	let offset = (size * 0.045).max(1.0);
	let shadow = [0.0, 0.0, 0.0, 0.55 * color[3]];
	push_text_pass(scene, fonts, atlas, size, x + offset, baseline + offset, text, shadow);
	push_text_pass(scene, fonts, atlas, size, x, baseline, text, color);
}

/// Builds the frame as GPU data: shapes and text quads. The background is
/// fully transparent (`Uniforms::bg_color` alpha 0) and the pulse animation
/// runs in the shader, so positions here are always the rest positions.
pub fn build_scene(
	hover_cancel: bool,
	fonts: &Fonts,
	atlas: &mut GlyphAtlas,
	width: u32,
	height: u32,
	scale: f32,
) -> Scene {
	let lay = layout(width, height, scale);
	let mut scene = Scene {
		shapes: build_shapes(&lay, hover_cancel, width, height),
		texts: Vec::new(),
	};
	let (bx, by, bw, bh) = lay.cancel_button;
	push_text(
		&mut scene,
		fonts,
		atlas,
		lay.text_size,
		bx + bw / 2.0,
		by + bh / 2.0,
		"Cancel",
		TEXT_COLOR,
	);
	scene
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn layout_fits_surface() {
		let lay = layout(380, 210, 1.0);
		let (bx, by, bw, bh) = lay.cancel_button;
		assert!(bx >= 0.0 && by >= 0.0);
		assert!(bx + bw <= 380.0 && by + bh <= 210.0);
		// Horizontally centered.
		assert!((bx + bw / 2.0 - 190.0).abs() < 1e-6);
		// Face glyph sits above the button.
		assert!(lay.face_center.1 + lay.face_radius < by);
	}

	#[test]
	fn layout_scales_fixed_elements() {
		let lay = layout(760, 420, 2.0);
		assert_eq!(lay.panel_radius, 36.0);
		assert_eq!(lay.face_radius, 60.0);
		let (_, _, bw, bh) = lay.cancel_button;
		assert_eq!((bw, bh), (264.0, 76.0));
	}

	#[test]
	fn hit_cancel_inside_and_outside() {
		let lay = layout(380, 210, 1.0);
		let (bx, by, bw, bh) = lay.cancel_button;
		assert!(hit_cancel(&lay, (bx + bw / 2.0) as f64, (by + bh / 2.0) as f64));
		assert!(hit_cancel(&lay, bx as f64, by as f64));
		assert!(hit_cancel(&lay, (bx + bw) as f64, (by + bh) as f64));
		assert!(!hit_cancel(&lay, (bx - 1.0) as f64, (by + bh / 2.0) as f64));
		assert!(!hit_cancel(&lay, (bx + bw / 2.0) as f64, (by - 1.0) as f64));
		assert!(!hit_cancel(&lay, 0.0, 0.0));
	}

	#[test]
	fn shapes_count_and_kinds() {
		let lay = layout(380, 210, 1.0);
		let shapes = build_shapes(&lay, false, 380, 210);
		// Panel + pulse ring + head + 2 eyes + button.
		assert_eq!(shapes.len(), 6);
		assert_eq!(shapes.iter().filter(|s| s.kind == SHAPE_PULSE_RING).count(), 1);
		assert_eq!(shapes.iter().filter(|s| s.kind == SHAPE_FACE_GLYPH).count(), 3);
	}

	#[test]
	fn panel_is_translucent_and_rounded() {
		let lay = layout(380, 210, 1.0);
		let shapes = build_shapes(&lay, false, 380, 210);
		let panel = &shapes[0];
		assert_eq!(panel.kind, 0);
		assert_eq!(panel.center, [190.0, 105.0]);
		assert_eq!(panel.half_size, [190.0, 105.0]);
		assert!((panel.color[3] - 0.85).abs() < 1e-6);
		assert!((panel.radius - 18.0).abs() < 1e-6);
		assert_eq!(panel.inner_radius, 0.0);
	}

	#[test]
	fn face_glyph_head_is_hollow() {
		let lay = layout(380, 210, 1.0);
		let shapes = build_shapes(&lay, false, 380, 210);
		let head = shapes
			.iter()
			.find(|s| s.kind == SHAPE_FACE_GLYPH && s.inner_radius > 0.0)
			.expect("hollow head circle");
		assert!(head.inner_radius < head.radius);
		let eyes: Vec<_> = shapes
			.iter()
			.filter(|s| s.kind == SHAPE_FACE_GLYPH && s.inner_radius == 0.0)
			.collect();
		assert_eq!(eyes.len(), 2);
		// Eyes are symmetric around the face center.
		assert!((eyes[0].center[0] + eyes[1].center[0]) / 2.0 - lay.face_center.0 < 1e-6);
	}

	#[test]
	fn hover_tints_button_toward_accent() {
		let lay = layout(380, 210, 1.0);
		let plain = build_shapes(&lay, false, 380, 210);
		let hovered = build_shapes(&lay, true, 380, 210);
		let button = plain.last().unwrap().color;
		let hover = hovered.last().unwrap().color;
		assert_eq!(button[..3], BOX_COLOR[..3]);
		assert!(hover[2] > button[2]); // accent is much bluer
		for i in 0..3 {
			let expected = BOX_COLOR[i] + (ACCENT_COLOR[i] - BOX_COLOR[i]) * 0.35;
			assert!((hover[i] - expected).abs() < 1e-6);
		}
		// Only the button changes on hover.
		for (a, b) in plain[..5].iter().zip(&hovered[..5]) {
			assert_eq!(a.center, b.center);
			assert_eq!(a.color, b.color);
			assert_eq!(a.kind, b.kind);
		}
	}

	#[test]
	fn scene_emits_cancel_text_with_shadow() {
		let Ok(fonts) = Fonts::load() else { return };
		let mut atlas = GlyphAtlas::new();
		let scene = build_scene(false, &fonts, &mut atlas, 380, 210, 1.0);
		assert_eq!(scene.shapes.len(), 6);
		// Shadow + main pass per glyph.
		assert!(!scene.texts.is_empty());
		assert_eq!(scene.texts.len() % 2, 0);
		let lay = layout(380, 210, 1.0);
		let (bx, by, bw, bh) = lay.cancel_button;
		assert!(
			scene
				.texts
				.iter()
				.all(|t| t.pos[0] >= bx && t.pos[0] <= bx + bw && t.pos[1] >= by && t.pos[1] <= by + bh)
		);
		// Shadow pass: black at 0.55 * text alpha, offset from the main pass.
		assert!(
			scene
				.texts
				.iter()
				.any(|t| t.color == [0.0, 0.0, 0.0, 0.55 * TEXT_COLOR[3]])
		);
		assert!(!atlas.drain_pending().is_empty());
	}
}
