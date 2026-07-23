use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};

use crate::config::{Color as UiColor, LockConfig};
use crate::scene::{
	ATLAS_SIZE, AtlasRect, SHAPE_DOT, SHAPE_FACE_DISC, SHAPE_FACE_GLYPH, SHAPE_PULSE_RING, SHAPE_SHAKE, Scene,
	ShapeInstance, TextInstance,
};

pub const DOT_POP_MS: u128 = 150;
pub const SHAKE_MS: u128 = 400;
pub const TOGGLE_MS: u128 = 200;

const WEEKDAYS: [&str; 7] = [
	"Sunday",
	"Monday",
	"Tuesday",
	"Wednesday",
	"Thursday",
	"Friday",
	"Saturday",
];
const MONTHS: [&str; 12] = [
	"January",
	"February",
	"March",
	"April",
	"May",
	"June",
	"July",
	"August",
	"September",
	"October",
	"November",
	"December",
];

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

#[derive(Debug)]
pub struct FontError(String);

impl fmt::Display for FontError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "failed to load a system sans-serif font: {}", self.0)
	}
}

impl std::error::Error for FontError {}

pub struct Fonts {
	font: FontVec,
}

impl Fonts {
	pub fn load() -> Result<Self, FontError> {
		let mut db = fontdb::Database::new();
		db.load_system_fonts();
		let query = fontdb::Query {
			families: &[fontdb::Family::SansSerif],
			..Default::default()
		};
		let id = db
			.query(&query)
			.ok_or_else(|| FontError("no matching face found".to_owned()))?;
		let (data, index) = db
			.with_face_data(id, |data: &[u8], index| (data.to_vec(), index))
			.ok_or_else(|| FontError("face data unavailable".to_owned()))?;
		let font = FontVec::try_from_vec_and_index(data, index)
			.map_err(|err| FontError(format!("failed to parse face: {err}")))?;
		Ok(Self { font })
	}

	fn scaled(&self, size_px: f32) -> ab_glyph::PxScaleFont<&FontVec> {
		self.font.as_scaled(PxScale::from(size_px))
	}

	/// Ascent in px at `size_px`.
	pub fn ascent(&self, size_px: f32) -> f32 {
		self.scaled(size_px).ascent()
	}

	/// Descent in px at `size_px` (negative).
	pub fn descent(&self, size_px: f32) -> f32 {
		self.scaled(size_px).descent()
	}

	/// Horizontal advance of `c` in px at `size_px`.
	pub fn h_advance(&self, c: char, size_px: f32) -> f32 {
		let sf = self.scaled(size_px);
		sf.h_advance(sf.glyph_id(c))
	}

	/// Kerning between two consecutive chars in px at `size_px`.
	pub fn kern(&self, prev: char, next: char, size_px: f32) -> f32 {
		let sf = self.scaled(size_px);
		sf.kern(sf.glyph_id(prev), sf.glyph_id(next))
	}

	/// Total advance width of `text` in px at `size_px`, including kerning.
	pub fn text_width(&self, text: &str, size_px: f32) -> f32 {
		let sf = self.scaled(size_px);
		let mut width = 0.0;
		let mut prev = None;
		for c in text.chars() {
			let id = sf.glyph_id(c);
			if let Some(prev) = prev {
				width += sf.kern(prev, id);
			}
			width += sf.h_advance(id);
			prev = Some(id);
		}
		width
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalTime {
	pub hour: i32,
	pub min: i32,
	pub mday: i32,
	/// 0 = January
	pub month: i32,
	/// 0 = Sunday
	pub wday: i32,
}

impl LocalTime {
	pub fn clock_text(&self) -> String {
		format!("{:02}:{:02}", self.hour, self.min)
	}

	pub fn date_text(&self) -> String {
		let wday = WEEKDAYS[self.wday.rem_euclid(7) as usize];
		let month = MONTHS[self.month.rem_euclid(12) as usize];
		format!("{wday}, {} {month}", self.mday)
	}
}

pub fn local_now() -> LocalTime {
	unsafe {
		let t = libc::time(std::ptr::null_mut());
		let mut tm: libc::tm = std::mem::zeroed();
		libc::localtime_r(&t, &mut tm);
		LocalTime {
			hour: tm.tm_hour,
			min: tm.tm_min,
			mday: tm.tm_mday,
			month: tm.tm_mon,
			wday: tm.tm_wday,
		}
	}
}

/// Straight-alpha RGBA for the scene instances.
fn rgba(c: UiColor, alpha: f32) -> [f32; 4] {
	[c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, alpha]
}

#[allow(dead_code)] // test mirror of the shader easing
pub fn ease_out_cubic(t: f32) -> f32 {
	let t = t.clamp(0.0, 1.0);
	1.0 - (1.0 - t) * (1.0 - t) * (1.0 - t)
}

/// CPU-side bookkeeping for the R8 glyph atlas: rasterizes glyphs on demand,
/// shelf-packs them with 1px padding and records the regions the renderer
/// still has to upload.
pub struct GlyphAtlas {
	/// (glyph id, rounded size in px) -> atlas location.
	glyphs: HashMap<(u16, u32), AtlasRect>,
	/// Rasterized coverage not yet uploaded to the atlas texture.
	pending: Vec<(AtlasRect, Vec<u8>)>,
	cursor_x: u32,
	cursor_y: u32,
	shelf_height: u32,
}

impl Default for GlyphAtlas {
	fn default() -> Self {
		Self::new()
	}
}

impl GlyphAtlas {
	pub fn new() -> Self {
		Self {
			glyphs: HashMap::new(),
			pending: Vec::new(),
			cursor_x: 0,
			cursor_y: 0,
			shelf_height: 0,
		}
	}

	fn allocate(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
		const PAD: u32 = 1;
		if self.cursor_x + w + PAD > ATLAS_SIZE {
			self.cursor_x = 0;
			self.cursor_y += self.shelf_height + PAD;
			self.shelf_height = 0;
		}
		if self.cursor_y + h + PAD > ATLAS_SIZE {
			return None;
		}
		let pos = (self.cursor_x, self.cursor_y);
		self.cursor_x += w + PAD;
		self.shelf_height = self.shelf_height.max(h);
		Some(pos)
	}

	/// Returns the atlas rect (with pen-relative bearing) and scaled advance
	/// for `c` at `size_px`, rasterizing and queueing it for upload on first
	/// use. `None` for glyphs without an outline (whitespace) or when the
	/// atlas is full.
	pub fn ensure(&mut self, fonts: &Fonts, c: char, size_px: f32) -> Option<(AtlasRect, f32)> {
		let font = &fonts.font;
		let sf = font.as_scaled(PxScale::from(size_px));
		let id = sf.glyph_id(c);
		let advance = sf.h_advance(id);
		let key = (id.0, size_px.round() as u32);
		if let Some(rect) = self.glyphs.get(&key) {
			return Some((*rect, advance));
		}
		let glyph = id.with_scale_and_position(PxScale::from(size_px), ab_glyph::point(0.0, 0.0));
		let outlined = font.outline_glyph(glyph)?;
		let bounds = outlined.px_bounds();
		let (w, h) = (
			(bounds.max.x - bounds.min.x) as u32,
			(bounds.max.y - bounds.min.y) as u32,
		);
		if w == 0 || h == 0 {
			return None;
		}
		let (x, y) = self.allocate(w, h)?;
		let mut coverage = vec![0u8; (w * h) as usize];
		outlined.draw(|gx, gy, c| {
			coverage[(gy * w + gx) as usize] = (c * 255.0).round() as u8;
		});
		let rect = AtlasRect {
			x,
			y,
			w,
			h,
			bearing: [bounds.min.x, bounds.min.y],
		};
		self.glyphs.insert(key, rect);
		self.pending.push((rect, coverage));
		Some((rect, advance))
	}

	/// Takes the rasterized glyphs queued since the last drain; the renderer
	/// uploads each bitmap at its atlas rect.
	pub fn drain_pending(&mut self) -> Vec<(AtlasRect, Vec<u8>)> {
		std::mem::take(&mut self.pending)
	}
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
	use crate::scene::SHAPE_DOT;
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
	fn clock_and_date_text() {
		let t = LocalTime {
			hour: 9,
			min: 5,
			mday: 3,
			month: 0,
			wday: 5,
		};
		assert_eq!(t.clock_text(), "09:05");
		assert_eq!(t.date_text(), "Friday, 3 January");
	}

	#[test]
	fn ease_out_cubic_bounds_and_shape() {
		assert_eq!(ease_out_cubic(0.0), 0.0);
		assert_eq!(ease_out_cubic(1.0), 1.0);
		assert!((ease_out_cubic(0.5) - 0.875).abs() < 1e-6);
		// Clamps outside [0, 1].
		assert_eq!(ease_out_cubic(-1.0), 0.0);
		assert_eq!(ease_out_cubic(2.0), 1.0);
		// Fast start, slow end.
		assert!(ease_out_cubic(0.25) > 0.25);
		assert!(ease_out_cubic(0.9) < 1.0);
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
	fn atlas_caches_glyphs() {
		let Ok(fonts) = Fonts::load() else { return };
		let mut atlas = GlyphAtlas::new();
		let (first, _) = atlas.ensure(&fonts, 'a', 24.0).expect("glyph 'a'");
		let (second, _) = atlas.ensure(&fonts, 'a', 24.0).expect("glyph 'a'");
		assert_eq!(first, second);
		// One rasterization for two lookups; sizes that round together share it.
		atlas.ensure(&fonts, 'a', 24.2);
		let pending = atlas.drain_pending();
		assert_eq!(pending.len(), 1);
		assert!(pending[0].1.len() >= (first.w * first.h) as usize);
		assert!(atlas.drain_pending().is_empty());
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
