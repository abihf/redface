use std::fmt;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Default config location: `$XDG_CONFIG_HOME/redface/lock.json`, or
/// `~/.config/redface/lock.json` when XDG_CONFIG_HOME is unset.
pub fn default_config_path() -> PathBuf {
	let base = std::env::var_os("XDG_CONFIG_HOME")
		.filter(|v| !v.is_empty())
		.map(PathBuf::from)
		.or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
		.unwrap_or_else(|| PathBuf::from("/etc"));
	base.join("redface/lock.json")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
	pub r: u8,
	pub g: u8,
	pub b: u8,
}

impl Color {
	pub const fn new(r: u8, g: u8, b: u8) -> Self {
		Self { r, g, b }
	}

	/// Parses "#rrggbb" (the leading '#' is optional).
	pub fn parse_hex(value: &str) -> Result<Self, ColorError> {
		let hex = value.strip_prefix('#').unwrap_or(value);
		if hex.len() != 6 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
			return Err(ColorError(value.to_owned()));
		}
		let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digits checked above");
		Ok(Self::new(channel(0), channel(2), channel(4)))
	}
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorError(String);

impl fmt::Display for ColorError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "invalid color '{}': expected #rrggbb", self.0)
	}
}

impl std::error::Error for ColorError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockConfig {
	/// Solid background color, used when no background_image is set.
	pub background: Color,
	/// Optional PNG/JPEG shown instead of the solid background (cover-scaled).
	pub background_image: Option<PathBuf>,
	/// xdg-output name of the monitor that shows the UI (e.g. "eDP-1");
	/// the first connected output is used when unset.
	pub primary_output: Option<String>,
	pub text_color: Color,
	pub box_color: Color,
	pub accent_color: Color,
}

impl Default for LockConfig {
	fn default() -> Self {
		Self {
			background: Color::new(0x10, 0x10, 0x14),
			background_image: None,
			primary_output: None,
			text_color: Color::new(0xe6, 0xe6, 0xe6),
			box_color: Color::new(0x26, 0x26, 0x2e),
			accent_color: Color::new(0x7a, 0xa2, 0xf7),
		}
	}
}

#[derive(Debug)]
pub enum ConfigError {
	Open { path: PathBuf, source: io::Error },
	Parse { path: PathBuf, source: serde_json::Error },
	Color { key: &'static str, source: ColorError },
}

impl fmt::Display for ConfigError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Open { path, source } => write!(f, "failed to open config {}: {source}", path.display()),
			Self::Parse { path, source } => write!(f, "failed to parse config {}: {source}", path.display()),
			Self::Color { key, source } => write!(f, "invalid '{key}': {source}"),
		}
	}
}

impl std::error::Error for ConfigError {}

#[derive(Debug, Default, Deserialize)]
struct RawConfig {
	background: Option<String>,
	background_image: Option<String>,
	primary_output: Option<String>,
	text_color: Option<String>,
	box_color: Option<String>,
	accent_color: Option<String>,
}

impl LockConfig {
	pub fn load_default() -> Result<Self, ConfigError> {
		Self::load_from_path(default_config_path())
	}

	/// A missing config file is not an error: defaults are used.
	pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
		let path = path.as_ref();
		let file = match File::open(path) {
			Ok(file) => file,
			Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
			Err(source) => {
				return Err(ConfigError::Open {
					path: path.to_path_buf(),
					source,
				});
			}
		};
		let raw = serde_json::from_reader(BufReader::new(file)).map_err(|source| ConfigError::Parse {
			path: path.to_path_buf(),
			source,
		})?;
		Self::from_raw(raw)
	}

	fn from_raw(raw: RawConfig) -> Result<Self, ConfigError> {
		let mut config = Self::default();
		let color = |key: &'static str, value: &Option<String>| -> Result<Option<Color>, ConfigError> {
			value
				.as_deref()
				.map(|v| Color::parse_hex(v).map_err(|source| ConfigError::Color { key, source }))
				.transpose()
		};
		if let Some(c) = color("background", &raw.background)? {
			config.background = c;
		}
		if let Some(c) = color("text_color", &raw.text_color)? {
			config.text_color = c;
		}
		if let Some(c) = color("box_color", &raw.box_color)? {
			config.box_color = c;
		}
		if let Some(c) = color("accent_color", &raw.accent_color)? {
			config.accent_color = c;
		}
		config.background_image = raw.background_image.map(PathBuf::from);
		config.primary_output = raw.primary_output;
		Ok(config)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn color_parses_hex_with_and_without_hash() {
		assert_eq!(Color::parse_hex("#7aa2f7"), Ok(Color::new(0x7a, 0xa2, 0xf7)));
		assert_eq!(Color::parse_hex("101014"), Ok(Color::new(0x10, 0x10, 0x14)));
	}

	#[test]
	fn color_rejects_invalid_hex() {
		assert!(Color::parse_hex("#12345").is_err());
		assert!(Color::parse_hex("#1234567").is_err());
		assert!(Color::parse_hex("#zzzzzz").is_err());
		assert!(Color::parse_hex("").is_err());
	}

	#[test]
	fn from_raw_applies_defaults() {
		let config = LockConfig::from_raw(RawConfig::default()).expect("defaults");
		assert_eq!(config, LockConfig::default());
	}

	#[test]
	fn from_raw_overrides_fields() {
		let raw = RawConfig {
			background: Some("#112233".to_owned()),
			background_image: Some("/tmp/bg.png".to_owned()),
			primary_output: Some("eDP-1".to_owned()),
			..RawConfig::default()
		};
		let config = LockConfig::from_raw(raw).expect("config");
		assert_eq!(config.background, Color::new(0x11, 0x22, 0x33));
		assert_eq!(config.background_image, Some(PathBuf::from("/tmp/bg.png")));
		assert_eq!(config.primary_output.as_deref(), Some("eDP-1"));
	}

	#[test]
	fn from_raw_rejects_bad_color() {
		let raw = RawConfig {
			accent_color: Some("red".to_owned()),
			..RawConfig::default()
		};
		assert!(matches!(
			LockConfig::from_raw(raw),
			Err(ConfigError::Color {
				key: "accent_color",
				..
			})
		));
	}

	#[test]
	fn missing_file_yields_defaults() {
		let config = LockConfig::load_from_path("/nonexistent/redface-lock-test.json").expect("defaults");
		assert_eq!(config, LockConfig::default());
	}

	#[test]
	fn default_path_is_under_config_home() {
		assert!(default_config_path().ends_with("redface/lock.json"));
	}
}
