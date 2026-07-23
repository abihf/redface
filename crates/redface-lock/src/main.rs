mod auth;
mod config;
mod gpu;
mod scene;
mod ui;
mod wayland;

use crate::config::LockConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let test = std::env::args().any(|arg| arg == "--test");
	let config = LockConfig::load_default()?;
	let background = config.background_image.as_deref().map(load_background).transpose()?;
	let fonts = ui::Fonts::load()?;
	wayland::run(config, background, fonts, test)
}

/// Decodes the configured background image to straight (non-premultiplied)
/// RGBA8 for the Vulkan background texture.
fn load_background(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error>> {
	let image = image::open(path)?.to_rgba8();
	let (width, height) = image.dimensions();
	Ok((image.into_raw(), width, height))
}
