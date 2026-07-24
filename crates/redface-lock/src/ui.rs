use std::time::Instant;

use redface_toolkit::scene::{
	ATLAS_SIZE, SHAPE_DOT, SHAPE_FACE_DISC, SHAPE_FACE_GLYPH, SHAPE_PULSE_RING, SHAPE_SHAKE, Scene, ShapeInstance,
	TextInstance,
};
use redface_toolkit::text::{Fonts, GlyphAtlas, local_now};

use crate::config::{Color as UiColor, LockConfig};

pub const DOT_POP_MS: u128 = 150;
pub const SHAKE_MS: u128 = 400;
pub const TOGGLE_MS: u128 = 200;

#[derive(Debug)]
pub struct UiState {
	password: Vec<(char, Instant)>,
	pub face_active: bool,
	face_toggled_at: Instant,
	shake_start: Option<Instant>,
	pub message: Option<String>,
	pub hover_face: bool,
}

impl Default for UiState {
	fn default() -> Self {
		Self::new()
	}
}

impl UiState {
	pub fn new() -> Self {
		Self {
			password: Vec::new(),
			face_active: false,
			face_toggled_at: Instant::now(),
			shake_start: None,
			message: None,
			hover_face: false,
		}
	}

	pub fn push_char(&mut self, c: char) {
		self.message = None;
		self.password.push((c, Instant::now()));
	}

	pub fn backspace(&mut self) {
		self.password.pop();
	}

	pub fn clear(&mut self) {
		self.password.clear();
	}

	pub fn is_empty(&self) -> bool {
		self.password.is_empty()
	}

	pub fn password_len(&self) -> usize {
		self.password.len()
	}

	/// Collects the password and clears the input buffer.
	pub fn take_password(&mut self) -> String {
		let password: String = self.password.iter().map(|(c, _)| c).collect();
		self.password.clear();
		password
	}

	pub fn set_face_active(&mut self, active: bool) {
		if self.face_active != active {
			self.face_active = active;
			self.face_toggled_at = Instant::now();
		}
	}

	/// Marks a failed authentication: clears the password, shows the message
	/// and starts the shake animation.
	pub fn fail(&mut self, msg: impl Into<String>) {
		self.password.clear();
		self.message = Some(msg.into());
		self.shake_start = Some(Instant::now());
	}

	/// Whether any animation is still running and the surface should keep
	/// repainting at the native refresh rate.
	pub fn animating(&self, now: Instant) -> bool {
		// The pulse runs continuously while face recognition is active.
		if self.face_active {
			return true;
		}
		if self
			.shake_start
			.is_some_and(|t| now.duration_since(t).as_millis() < SHAKE_MS)
		{
			return true;
		}
		if now.duration_since(self.face_toggled_at).as_millis() < TOGGLE_MS {
			return true;
		}
		self.password
			.iter()
			.any(|(_, t)| now.duration_since(*t).as_millis() < DOT_POP_MS)
	}

	/// Seconds since `epoch` of the last face-toggle (for `Uniforms::face_toggled_at`).
	pub fn face_toggled_at_secs(&self, epoch: Instant) -> f32 {
		self.face_toggled_at.saturating_duration_since(epoch).as_secs_f32()
	}

	/// Seconds since `epoch` of the last failed-auth shake start, -1.0 when
	/// the shake never ran (for `Uniforms::shake_start`).
	pub fn shake_start_secs(&self, epoch: Instant) -> f32 {
		self.shake_start
			.map_or(-1.0, |t| t.saturating_duration_since(epoch).as_secs_f32())
	}

	/// Birth time (seconds since `epoch`) of each password dot, in the order
	/// the characters were typed.
	pub fn dot_births(&self, epoch: Instant) -> Vec<f32> {
		self.password
			.iter()
			.map(|(_, t)| t.saturating_duration_since(epoch).as_secs_f32())
			.collect()
	}
}

/// Screen geometry shared by the scene builder and pointer hit-testing.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
	/// (x, y, width, height) of the password input box.
	pub password_box: (f32, f32, f32, f32),
	pub face_center: (f32, f32),
	pub face_radius: f32,
	pub time_center_y: f32,
	pub date_center_y: f32,
	pub message_center_y: f32,
}

/// `width`/`height` are in the coordinate space being laid out (buffer pixels
/// when building the scene, logical surface coordinates for hit-testing) and
/// `scale` multiplies the fixed-size elements accordingly.
pub fn layout(width: u32, height: u32, scale: f32) -> Layout {
	let (w, h) = (width as f32, height as f32);
	let cx = w / 2.0;
	let box_w = (w * 0.45).clamp(260.0 * scale, 420.0 * scale);
	let box_h = 52.0 * scale;
	let box_y = h * 0.78 - box_h / 2.0;
	Layout {
		password_box: (cx - box_w / 2.0, box_y, box_w, box_h),
		face_center: (cx, box_y + box_h + 88.0 * scale),
		face_radius: 34.0 * scale,
		time_center_y: h * 0.12,
		date_center_y: h * 0.12 + h * 0.11 * 0.75,
		message_center_y: box_y + box_h + 30.0 * scale,
	}
}

pub fn hit_face(layout: &Layout, x: f64, y: f64) -> bool {
	let dx = x as f32 - layout.face_center.0;
	let dy = y as f32 - layout.face_center.1;
	(dx * dx + dy * dy).sqrt() <= layout.face_radius
}

/// Straight-alpha RGBA for the scene instances.
fn rgba(c: UiColor, alpha: f32) -> [f32; 4] {
	[c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, alpha]
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
				let atlas = ATLAS_SIZE as f32;
				scene.texts.push(TextInstance {
					pos: [x + rect.bearing[0], baseline + rect.bearing[1]],
					size: [rect.w as f32, rect.h as f32],
					uv_pos: [rect.x as f32 / atlas, rect.y as f32 / atlas],
					uv_size: [rect.w as f32 / atlas, rect.h as f32 / atlas],
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
/// (keeps the text legible over arbitrary background images), then the text.
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
/// handled by the renderer (see `Uniforms`) and all animations (shake, dot
/// pop, toggle crossfade, pulse) run in the shader, so positions here are
/// always the rest positions.
#[allow(clippy::too_many_arguments)]
pub fn build_scene(
	state: &UiState,
	config: &LockConfig,
	fonts: &Fonts,
	atlas: &mut GlyphAtlas,
	width: u32,
	height: u32,
	scale: f32,
	epoch: Instant,
) -> Scene {
	let mut scene = Scene::default();
	let lay = layout(width, height, scale);
	let cx = width as f32 / 2.0;
	let time = local_now();

	let time_size = (height as f32 * 0.11).clamp(40.0 * scale, 160.0 * scale);
	let date_size = (time_size * 0.28).max(16.0 * scale);
	let time_text = time.clock_text();
	let date_text = time.date_text();

	// Clock and date sit on a semi-transparent black backdrop for legibility.
	let pad_x = 28.0 * scale;
	let pad_y = 16.0 * scale;
	let text_w = fonts
		.text_width(&time_text, time_size)
		.max(fonts.text_width(&date_text, date_size));
	let top = lay.time_center_y - time_size * 0.6 - pad_y;
	let bottom = lay.date_center_y + date_size * 0.6 + pad_y;
	scene.shapes.push(ShapeInstance {
		center: [cx, (top + bottom) / 2.0],
		half_size: [text_w / 2.0 + pad_x, (bottom - top) / 2.0],
		color: [0.0, 0.0, 0.0, 0.5],
		radius: 16.0 * scale,
		inner_radius: 0.0,
		birth_time: -1.0,
		kind: 0,
	});

	push_text(
		&mut scene,
		fonts,
		atlas,
		time_size,
		cx,
		lay.time_center_y,
		&time_text,
		rgba(config.text_color, 1.0),
	);
	push_text(
		&mut scene,
		fonts,
		atlas,
		date_size,
		cx,
		lay.date_center_y,
		&date_text,
		rgba(config.text_color, 0.7),
	);

	// Password box; the shake offset is applied in the shader.
	let (bx, by, bw, bh) = lay.password_box;
	let box_center = [bx + bw / 2.0, by + bh / 2.0];
	let box_half = [bw / 2.0, bh / 2.0];
	let corner = 12.0 * scale;
	scene.shapes.push(ShapeInstance {
		center: box_center,
		half_size: box_half,
		color: rgba(config.box_color, 0.9),
		radius: corner,
		inner_radius: 0.0,
		birth_time: -1.0,
		kind: SHAPE_SHAKE,
	});
	let border_color = if state.message.is_some() {
		[0.9, 0.35, 0.35, 0.9]
	} else {
		rgba(config.text_color, 0.25)
	};
	scene.shapes.push(ShapeInstance {
		center: box_center,
		half_size: box_half,
		color: border_color,
		radius: corner,
		inner_radius: (corner - 1.5 * scale).max(0.0),
		birth_time: -1.0,
		kind: SHAPE_SHAKE,
	});

	if state.is_empty() {
		push_text(
			&mut scene,
			fonts,
			atlas,
			18.0 * scale,
			cx,
			by + bh / 2.0,
			"Password",
			rgba(config.text_color, 0.35),
		);
	} else {
		// One dot per char; each pops in from its birth time in the shader.
		let spacing = 22.0 * scale;
		let total = (state.password_len() as f32 - 1.0) * spacing;
		let start_x = cx - total / 2.0;
		let radius = 5.0 * scale;
		for (i, birth) in state.dot_births(epoch).into_iter().enumerate() {
			scene.shapes.push(ShapeInstance {
				center: [start_x + i as f32 * spacing, by + bh / 2.0],
				half_size: [radius, radius],
				color: rgba(config.text_color, 1.0),
				radius,
				inner_radius: 0.0,
				birth_time: birth,
				kind: SHAPE_DOT,
			});
		}
	}

	if let Some(message) = &state.message {
		push_text(
			&mut scene,
			fonts,
			atlas,
			18.0 * scale,
			cx,
			lay.message_center_y,
			message,
			[0.9, 0.35, 0.35, 1.0],
		);
	}

	// Face toggle button: the shader crossfades box_color -> accent on toggle
	// and animates the pulse ring while face recognition is active.
	let (fx, fy) = lay.face_center;
	let fr = lay.face_radius;
	let disc_alpha = if state.hover_face { 0.98 } else { 0.9 };
	scene.shapes.push(ShapeInstance {
		center: [fx, fy],
		half_size: [fr, fr],
		color: rgba(config.box_color, disc_alpha),
		radius: fr,
		inner_radius: 0.0,
		birth_time: -1.0,
		kind: SHAPE_FACE_DISC,
	});
	let ring_radius = fr + 6.0 * scale;
	scene.shapes.push(ShapeInstance {
		center: [fx, fy],
		half_size: [ring_radius, ring_radius],
		color: rgba(config.accent_color, 0.45),
		radius: ring_radius,
		inner_radius: (ring_radius - 3.0 * scale).max(0.0),
		birth_time: -1.0,
		kind: SHAPE_PULSE_RING,
	});

	// Face glyph: head circle plus two eyes.
	let glyph_color = rgba(config.text_color, 0.9);
	let head_radius = fr * 0.42;
	scene.shapes.push(ShapeInstance {
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
		scene.shapes.push(ShapeInstance {
			center: [fx + side * fr * 0.18, fy - fr * 0.08],
			half_size: [eye_radius, eye_radius],
			color: glyph_color,
			radius: eye_radius,
			inner_radius: 0.0,
			birth_time: -1.0,
			kind: SHAPE_FACE_GLYPH,
		});
	}
	scene
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::time::Duration;

	#[test]
	fn password_append_backspace_clear() {
		let mut ui = UiState::new();
		assert!(ui.is_empty());
		ui.push_char('a');
		ui.push_char('b');
		assert_eq!(ui.password_len(), 2);
		assert_eq!(ui.take_password(), "ab");
		assert!(ui.is_empty());

		ui.push_char('x');
		ui.push_char('y');
		ui.backspace();
		assert_eq!(ui.take_password(), "x");
		ui.push_char('z');
		ui.clear();
		assert!(ui.is_empty());
	}

	#[test]
	fn push_char_clears_message() {
		let mut ui = UiState::new();
		ui.fail("nope");
		assert!(ui.message.is_some());
		ui.push_char('a');
		assert!(ui.message.is_none());
	}

	#[test]
	fn fail_sets_message_and_shake() {
		let mut ui = UiState::new();
		ui.push_char('a');
		ui.fail("wrong password");
		assert!(ui.is_empty());
		assert_eq!(ui.message.as_deref(), Some("wrong password"));
		assert!(ui.animating(Instant::now()));
	}

	#[test]
	fn animating_states() {
		let mut ui = UiState::new();
		let now = Instant::now();
		// Fresh state: the initial face_toggled_at is within TOGGLE_MS.
		assert!(ui.animating(now));
		ui.face_toggled_at = now - Duration::from_millis(500);
		assert!(!ui.animating(now));

		ui.push_char('a');
		assert!(ui.animating(now));

		let mut ui = UiState::new();
		ui.face_toggled_at = now - Duration::from_millis(500);
		ui.set_face_active(true);
		assert!(ui.animating(now));
	}

	#[test]
	fn set_face_active_retriggers_toggle_animation() {
		let mut ui = UiState::new();
		ui.face_toggled_at = Instant::now() - Duration::from_secs(10);
		ui.set_face_active(true);
		assert!(ui.face_active);
		assert!(Instant::now().duration_since(ui.face_toggled_at).as_millis() < TOGGLE_MS);
		// No state change: timestamp must not move.
		let ts = ui.face_toggled_at;
		ui.set_face_active(true);
		assert_eq!(ui.face_toggled_at, ts);
	}

	#[test]
	fn hit_face_uses_button_circle() {
		let lay = layout(1920, 1080, 1.0);
		assert!(hit_face(&lay, lay.face_center.0 as f64, lay.face_center.1 as f64));
		assert!(!hit_face(&lay, 10.0, 10.0));
		let edge = lay.face_center.0 as f64 + lay.face_radius as f64 * 2.0;
		assert!(!hit_face(&lay, edge, lay.face_center.1 as f64));
	}

	#[test]
	fn dot_births_track_password_chars() {
		let epoch = Instant::now();
		let mut ui = UiState::new();
		assert!(ui.dot_births(epoch).is_empty());
		ui.push_char('a');
		ui.push_char('b');
		ui.push_char('c');
		let births = ui.dot_births(epoch);
		assert_eq!(births.len(), ui.password_len());
		assert!(births.iter().all(|b| *b >= 0.0));
		assert!(births.windows(2).all(|w| w[0] <= w[1]));
		ui.backspace();
		assert_eq!(ui.dot_births(epoch).len(), 2);
	}

	#[test]
	fn animation_timestamps_relative_to_epoch() {
		let epoch = Instant::now();
		let mut ui = UiState::new();
		assert_eq!(ui.shake_start_secs(epoch), -1.0);
		assert!(ui.face_toggled_at_secs(epoch) >= 0.0);
		ui.fail("nope");
		assert!(ui.shake_start_secs(epoch) >= 0.0);
	}

	#[test]
	fn build_scene_emits_dots_and_text_for_password() {
		let Ok(fonts) = Fonts::load() else { return };
		let mut atlas = GlyphAtlas::new();
		let config = LockConfig::default();
		let epoch = Instant::now();
		let mut state = UiState::new();
		state.push_char('a');
		state.push_char('b');
		let scene = build_scene(&state, &config, &fonts, &mut atlas, 1920, 1080, 1.0, epoch);
		// Backdrop + box + border + disc + ring + head + 2 eyes + 2 dots.
		assert_eq!(scene.shapes.len(), 10);
		let births: Vec<f32> = scene
			.shapes
			.iter()
			.filter(|s| s.kind == SHAPE_DOT)
			.map(|s| s.birth_time)
			.collect();
		assert_eq!(births.len(), 2);
		assert!(births.iter().all(|b| *b >= 0.0));
		assert!(births.windows(2).all(|w| w[0] <= w[1]));
		// Time and date strings produce glyphs (shadow + main passes).
		assert!(!scene.texts.is_empty());
		assert!(!atlas.drain_pending().is_empty());
	}

	#[test]
	fn build_scene_empty_password_shows_placeholder() {
		let Ok(fonts) = Fonts::load() else { return };
		let mut atlas = GlyphAtlas::new();
		let config = LockConfig::default();
		let state = UiState::new();
		let scene = build_scene(&state, &config, &fonts, &mut atlas, 1920, 1080, 1.0, Instant::now());
		assert!(!scene.shapes.iter().any(|s| s.kind == SHAPE_DOT));
		// Placeholder quads (alpha 0.35) sit inside the password box.
		let (_, by, _, bh) = layout(1920, 1080, 1.0).password_box;
		assert!(
			scene
				.texts
				.iter()
				.any(|t| (t.color[3] - 0.35).abs() < 1e-6 && t.pos[1] >= by && t.pos[1] <= by + bh)
		);
	}
}
