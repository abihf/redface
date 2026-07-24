//! Font loading, text measurement and the CPU-side glyph atlas shared by all
//! redface apps. Scene building itself lives in each app; this module only
//! provides the font/metrics/atlas plumbing plus the local-clock helpers.

use std::collections::HashMap;
use std::fmt;

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};

use crate::scene::{ATLAS_SIZE, AtlasRect};

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

#[cfg(test)]
mod tests {
	use super::*;

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
}
