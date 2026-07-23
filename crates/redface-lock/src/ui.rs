use std::fmt;
use std::time::Instant;

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, PixmapMut, PixmapPaint, Shader, Stroke, Transform};

use crate::config::{Color as UiColor, LockConfig};

const DOT_POP_MS: u128 = 150;
const SHAKE_MS: u128 = 400;
const TOGGLE_MS: u128 = 200;
const PULSE_MS: u128 = 1200;

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
}

/// Screen geometry shared by the renderer and pointer hit-testing.
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
/// when rendering, logical surface coordinates for hit-testing) and `scale`
/// multiplies the fixed-size elements accordingly.
pub fn layout(width: u32, height: u32, scale: f32) -> Layout {
	let (w, h) = (width as f32, height as f32);
	let cx = w / 2.0;
	let box_w = (w * 0.45).clamp(260.0 * scale, 420.0 * scale);
	let box_h = 52.0 * scale;
	let box_y = h * 0.55 - box_h / 2.0;
	Layout {
		password_box: (cx - box_w / 2.0, box_y, box_w, box_h),
		face_center: (cx, box_y + box_h + 88.0 * scale),
		face_radius: 34.0 * scale,
		time_center_y: h * 0.28,
		date_center_y: h * 0.28 + h * 0.11 * 0.75,
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

/// Composites the static background for a surface: the cover-scaled image, or
/// the solid configured color. Cached per surface and memcpy'd each frame —
/// re-blitting the image every frame is the dominant render cost.
pub fn compose_background(width: u32, height: u32, image: Option<&Pixmap>, config: &LockConfig) -> Option<Pixmap> {
	let mut pixmap = Pixmap::new(width, height)?;
	match image.and_then(|img| scale_cover(img, width, height)) {
		Some(scaled) => {
			pixmap.draw_pixmap(
				0,
				0,
				scaled.as_ref(),
				&PixmapPaint::default(),
				Transform::default(),
				None,
			);
		}
		None => pixmap.fill(ts_color(config.background, 1.0)),
	}
	Some(pixmap)
}

/// Scales the decoded background image to cover (and crop to) the surface size.
pub fn scale_cover(src: &Pixmap, width: u32, height: u32) -> Option<Pixmap> {
	let mut dst = Pixmap::new(width, height)?;
	let (sw, sh) = (src.width() as f32, src.height() as f32);
	let scale = (width as f32 / sw).max(height as f32 / sh);
	let tx = (width as f32 - sw * scale) / 2.0;
	let ty = (height as f32 - sh * scale) / 2.0;
	let transform = Transform::from_row(scale, 0.0, 0.0, scale, tx, ty);
	dst.draw_pixmap(0, 0, src.as_ref(), &PixmapPaint::default(), transform, None);
	Some(dst)
}

fn ts_color(c: UiColor, alpha: f32) -> Color {
	Color::from_rgba(c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, alpha).unwrap_or(Color::BLACK)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
	a + (b - a) * t
}

fn ease_out_cubic(t: f32) -> f32 {
	let t = t.clamp(0.0, 1.0);
	1.0 - (1.0 - t) * (1.0 - t) * (1.0 - t)
}

fn lerp_color(a: UiColor, b: UiColor, t: f32, alpha: f32) -> Color {
	ts_color(
		UiColor::new(
			lerp(a.r as f32, b.r as f32, t) as u8,
			lerp(a.g as f32, b.g as f32, t) as u8,
			lerp(a.b as f32, b.b as f32, t) as u8,
		),
		alpha,
	)
}

fn solid(color: Color) -> Paint<'static> {
	Paint {
		shader: Shader::SolidColor(color),
		anti_alias: true,
		..Default::default()
	}
}

fn circle_path(x: f32, y: f32, r: f32) -> tiny_skia::Path {
	let mut pb = PathBuilder::new();
	pb.push_circle(x, y, r);
	pb.finish().expect("circle path is never empty")
}

fn rounded_rect_path(x: f32, y: f32, w: f32, h: f32, r: f32) -> tiny_skia::Path {
	let mut pb = PathBuilder::new();
	pb.move_to(x + r, y);
	pb.line_to(x + w - r, y);
	pb.quad_to(x + w, y, x + w, y + r);
	pb.line_to(x + w, y + h - r);
	pb.quad_to(x + w, y + h, x + w - r, y + h);
	pb.line_to(x + r, y + h);
	pb.quad_to(x, y + h, x, y + h - r);
	pb.line_to(x, y + r);
	pb.quad_to(x, y, x + r, y);
	pb.close();
	pb.finish().expect("rounded rect path is never empty")
}

/// Blends one coverage pixel of a glyph into the premultiplied pixmap.
fn blend_pixel(pixmap: &mut PixmapMut<'_>, x: i32, y: i32, color: Color, coverage: f32) {
	if x < 0 || y < 0 || x >= pixmap.width() as i32 || y >= pixmap.height() as i32 {
		return;
	}
	let sa = coverage * color.alpha();
	let inv = 1.0 - sa;
	let idx = (y as usize * pixmap.width() as usize + x as usize) * 4;
	let data = pixmap.data_mut();
	let blend = |src: f32, dst: u8| -> u8 { (src * sa * 255.0 + dst as f32 * inv).min(255.0) as u8 };
	data[idx] = blend(color.red(), data[idx]);
	data[idx + 1] = blend(color.green(), data[idx + 1]);
	data[idx + 2] = blend(color.blue(), data[idx + 2]);
	data[idx + 3] = (sa * 255.0 + data[idx + 3] as f32 * inv).min(255.0) as u8;
}

/// Draws `text` centered at (center_x, center_y) with the given pixel size.
pub fn draw_text(
	pixmap: &mut PixmapMut<'_>,
	fonts: &Fonts,
	size: f32,
	center_x: f32,
	center_y: f32,
	text: &str,
	color: Color,
) {
	let font = &fonts.font;
	let sf = font.as_scaled(PxScale::from(size));
	let ids: Vec<_> = text.chars().map(|c| sf.glyph_id(c)).collect();
	let mut width = 0.0;
	for (i, id) in ids.iter().enumerate() {
		if i > 0 {
			width += sf.kern(ids[i - 1], *id);
		}
		width += sf.h_advance(*id);
	}
	let baseline = center_y + (sf.ascent() + sf.descent()) / 2.0;
	let mut draw_pass = |dx: f32, dy: f32, color: Color| {
		let mut x = center_x - width / 2.0 + dx;
		for (i, id) in ids.iter().enumerate() {
			if i > 0 {
				x += sf.kern(ids[i - 1], *id);
			}
			let glyph = id.with_scale_and_position(PxScale::from(size), ab_glyph::point(x, baseline + dy));
			if let Some(outlined) = font.outline_glyph(glyph) {
				let bounds = outlined.px_bounds();
				let (ox, oy) = (bounds.min.x as i32, bounds.min.y as i32);
				outlined.draw(|gx, gy, coverage| {
					blend_pixel(pixmap, ox + gx as i32, oy + gy as i32, color, coverage);
				});
			}
			x += sf.h_advance(*id);
		}
	};
	// Drop shadow keeps the text legible over arbitrary background images.
	let offset = (size * 0.045).max(1.0);
	draw_pass(
		offset,
		offset,
		Color::from_rgba(0.0, 0.0, 0.0, 0.55 * color.alpha()).unwrap_or(Color::BLACK),
	);
	draw_pass(0.0, 0.0, color);
}

/// Per-surface rendering inputs.
pub struct SurfaceView<'a> {
	/// Cached background (see `compose_background`), memcpy'd into the frame.
	pub base: &'a Pixmap,
	/// Non-primary surfaces only get the background.
	pub primary: bool,
	/// HiDPI buffer scale; multiplies the fixed-size elements.
	pub scale: f32,
}

/// Renders the whole frame onto `pixmap`.
pub fn render(
	pixmap: &mut PixmapMut<'_>,
	view: &SurfaceView<'_>,
	state: &UiState,
	fonts: &Fonts,
	config: &LockConfig,
	now: Instant,
) {
	pixmap.data_mut().copy_from_slice(view.base.data());
	if !view.primary {
		return;
	}
	let scale = view.scale;

	let lay = layout(pixmap.width(), pixmap.height(), scale);
	let cx = pixmap.width() as f32 / 2.0;
	let time = local_now();
	let text = ts_color(config.text_color, 1.0);

	let time_size = (pixmap.height() as f32 * 0.11).clamp(40.0 * scale, 160.0 * scale);
	draw_text(
		pixmap,
		fonts,
		time_size,
		cx,
		lay.time_center_y,
		&time.clock_text(),
		text,
	);
	draw_text(
		pixmap,
		fonts,
		(time_size * 0.28).max(16.0 * scale),
		cx,
		lay.date_center_y,
		&time.date_text(),
		ts_color(config.text_color, 0.7),
	);

	// Password box, with the shake offset applied on failed auth.
	let (bx, by, bw, bh) = lay.password_box;
	let mut ox = 0.0;
	if let Some(start) = state.shake_start {
		let t = now.duration_since(start).as_secs_f32() * 1000.0 / SHAKE_MS as f32;
		if t < 1.0 {
			ox = 14.0 * scale * (t * std::f32::consts::TAU * 4.0).sin() * (1.0 - t);
		}
	}
	let box_path = rounded_rect_path(bx + ox, by, bw, bh, 12.0 * scale);
	pixmap.fill_path(
		&box_path,
		&solid(ts_color(config.box_color, 0.9)),
		FillRule::Winding,
		Transform::default(),
		None,
	);
	let border = Stroke {
		width: 1.5 * scale,
		..Default::default()
	};
	let border_color = if state.message.is_some() {
		Color::from_rgba(0.9, 0.35, 0.35, 0.9).unwrap()
	} else {
		ts_color(config.text_color, 0.25)
	};
	pixmap.stroke_path(&box_path, &solid(border_color), &border, Transform::default(), None);

	if state.is_empty() {
		draw_text(
			pixmap,
			fonts,
			18.0 * scale,
			cx + ox,
			by + bh / 2.0,
			"Password",
			ts_color(config.text_color, 0.35),
		);
	} else {
		// Dots pop in one by one; the newest is still animating.
		let spacing = 22.0 * scale;
		let total = (state.password_len() as f32 - 1.0) * spacing;
		let start_x = cx + ox - total / 2.0;
		for (i, (_, added)) in state.password.iter().enumerate() {
			let t = ease_out_cubic(now.duration_since(*added).as_secs_f32() * 1000.0 / DOT_POP_MS as f32);
			let r = 5.0 * scale * t;
			if r <= 0.0 {
				continue;
			}
			let dot = circle_path(start_x + i as f32 * spacing, by + bh / 2.0, r);
			pixmap.fill_path(
				&dot,
				&solid(ts_color(config.text_color, t)),
				FillRule::Winding,
				Transform::default(),
				None,
			);
		}
	}

	if let Some(message) = &state.message {
		draw_text(
			pixmap,
			fonts,
			18.0 * scale,
			cx,
			lay.message_center_y,
			message,
			Color::from_rgba(0.9, 0.35, 0.35, 1.0).unwrap(),
		);
	}

	// Face toggle button: crossfade on toggle, pulsing ring while scanning.
	let toggle_t = ease_out_cubic(now.duration_since(state.face_toggled_at).as_secs_f32() * 1000.0 / TOGGLE_MS as f32);
	let active_amount = if state.face_active { toggle_t } else { 1.0 - toggle_t };
	let (fx, fy) = lay.face_center;
	let fr = lay.face_radius;
	let hover_boost = if state.hover_face { 0.08 } else { 0.0 };
	let fill = lerp_color(
		config.box_color,
		config.accent_color,
		active_amount * 0.35,
		0.9 + hover_boost,
	);
	let btn = circle_path(fx, fy, fr);
	pixmap.fill_path(&btn, &solid(fill), FillRule::Winding, Transform::default(), None);

	if state.face_active {
		let pulse = (now.duration_since(state.face_toggled_at).as_secs_f32() * 1000.0 / PULSE_MS as f32
			* std::f32::consts::TAU)
			.sin();
		let ring_alpha = 0.45 + 0.35 * pulse;
		let ring_stroke = Stroke {
			width: 3.0 * scale,
			..Default::default()
		};
		let ring = circle_path(fx, fy, fr + (6.0 + 2.0 * pulse) * scale);
		pixmap.stroke_path(
			&ring,
			&solid(ts_color(config.accent_color, ring_alpha)),
			&ring_stroke,
			Transform::default(),
			None,
		);
	}

	// Face glyph: head circle plus two eyes.
	let glyph_color = lerp_color(config.text_color, config.accent_color, active_amount, 0.9);
	let glyph_stroke = Stroke {
		width: 3.0 * scale,
		..Default::default()
	};
	let head = circle_path(fx, fy - fr * 0.08, fr * 0.42);
	pixmap.stroke_path(&head, &solid(glyph_color), &glyph_stroke, Transform::default(), None);
	for side in [-1.0f32, 1.0] {
		let eye = circle_path(fx + side * fr * 0.18, fy - fr * 0.16, fr * 0.055);
		pixmap.fill_path(&eye, &solid(glyph_color), FillRule::Winding, Transform::default(), None);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::time::Duration;

	// quick check of scale_cover + draw_pixmap blit
	#[test]
	fn background_pixels_survive_scaling_and_blit() {
		use tiny_skia::{Pixmap, PixmapPaint, Transform};
		let mut src = Pixmap::new(8, 8).unwrap();
		src.fill(tiny_skia::Color::from_rgba8(255, 0, 0, 255));
		let scaled = super::scale_cover(&src, 100, 100).unwrap();
		// center pixel of the scaled image must be red
		let px = scaled.pixel(50, 50).expect("pixel");
		assert_eq!((px.red(), px.green(), px.blue(), px.alpha()), (255, 0, 0, 255));

		// and blitting it onto a frame must produce red too
		let mut frame = Pixmap::new(100, 100).unwrap();
		frame.draw_pixmap(
			0,
			0,
			scaled.as_ref(),
			&PixmapPaint::default(),
			Transform::default(),
			None,
		);
		let px = frame.pixel(50, 50).expect("pixel");
		assert_eq!((px.red(), px.green(), px.blue(), px.alpha()), (255, 0, 0, 255));
	}

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
	fn scale_cover_crops_to_target_size() {
		let src = Pixmap::new(400, 200).expect("src");
		let dst = scale_cover(&src, 100, 100).expect("dst");
		assert_eq!((dst.width(), dst.height()), (100, 100));
	}
}
