mod auth;
mod config;
mod ui;
mod wayland;

use tiny_skia::{IntSize, Pixmap};

use crate::config::LockConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let test = std::env::args().any(|arg| arg == "--test");
	let config = LockConfig::load_default()?;
	let background = config.background_image.as_deref().map(load_background).transpose()?;
	let fonts = ui::Fonts::load()?;
	wayland::run(config, background, fonts, test)
}

/// Decodes the configured background image and premultiplies its alpha for
/// tiny-skia (image decoders return straight alpha).
fn load_background(path: &std::path::Path) -> Result<Pixmap, Box<dyn std::error::Error>> {
	let image = image::open(path)?.to_rgba8();
	let (width, height) = image.dimensions();
	let mut data = image.into_raw();
	for pixel in data.chunks_exact_mut(4) {
		let alpha = pixel[3] as u16;
		pixel[0] = ((pixel[0] as u16 * alpha + 127) / 255) as u8;
		pixel[1] = ((pixel[1] as u16 * alpha + 127) / 255) as u8;
		pixel[2] = ((pixel[2] as u16 * alpha + 127) / 255) as u8;
	}
	let size = IntSize::from_wh(width, height).ok_or("background image has zero dimensions")?;
	Pixmap::from_vec(data, size).ok_or_else(|| "background image size mismatch".into())
}
