use std::fmt;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;

pub const DEFAULT_CONFIG_PATH: &str = "/etc/redface/config.json";
pub const DEFAULT_SOCKET_PATH: &str = "/var/run/redface.sock";
pub const DEFAULT_PID_PATH: &str = "/var/run/redface.pid";
pub const DEFAULT_DATA_DIR: &str = "/usr/share/redface";
pub const DEFAULT_MODELS_DIR: &str = "/etc/redface/models";

const OSD_SOCKET_SUFFIX: &str = "redface-osd.sock";

pub fn get_osd_socket_path(user_id: u32) -> PathBuf {
	PathBuf::from(format!("/run/user/{user_id}/{OSD_SOCKET_SUFFIX}"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
	pub device: String,
	/// Inference device preference: "NPU", "CPU" or "AUTO" (default "NPU").
	pub inference_device: String,
	/// Cosine-similarity acceptance threshold in [-1, 1]; higher is stricter.
	pub threshold: f64,
	pub timeout: u64,
	pub socket: String,
	pub pid_file: String,
}

#[derive(Debug)]
pub enum ConfigError {
	Open { path: PathBuf, source: io::Error },
	Parse { path: PathBuf, source: serde_json::Error },
	MissingDevice,
	InvalidInferenceDevice { value: String },
}

impl fmt::Display for ConfigError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Open { path, source } => write!(f, "failed to open config {}: {source}", path.display()),
			Self::Parse { path, source } => write!(f, "failed to parse config {}: {source}", path.display()),
			Self::MissingDevice => write!(f, "Device not set"),
			Self::InvalidInferenceDevice { value } => {
				write!(f, "invalid inference_device '{value}': expected NPU, CPU or AUTO")
			}
		}
	}
}

impl std::error::Error for ConfigError {}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawConfig {
	device: Option<String>,
	inference_device: Option<String>,
	threshold: Option<f64>,
	timeout: Option<u64>,
	socket: Option<String>,
	pid_file: Option<String>,
}

impl Config {
	pub fn load_default() -> Result<Self, ConfigError> {
		Self::load_from_path(DEFAULT_CONFIG_PATH)
	}

	pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
		let path = path.as_ref();
		let file = File::open(path).map_err(|source| ConfigError::Open {
			path: path.to_path_buf(),
			source,
		})?;
		let raw = serde_json::from_reader(file).map_err(|source| ConfigError::Parse {
			path: path.to_path_buf(),
			source,
		})?;
		Self::from_raw(raw)
	}

	fn from_raw(raw: RawConfig) -> Result<Self, ConfigError> {
		let device = raw.device.unwrap_or_default();
		if device.is_empty() {
			return Err(ConfigError::MissingDevice);
		}

		let inference_device = raw.inference_device.unwrap_or_else(|| "NPU".to_owned());
		DevicePref::parse(&inference_device).map_err(|_| ConfigError::InvalidInferenceDevice {
			value: inference_device.clone(),
		})?;

		Ok(Self {
			device,
			inference_device,
			threshold: raw.threshold.unwrap_or(0.9),
			timeout: raw.timeout.unwrap_or(10),
			socket: raw.socket.unwrap_or_else(|| DEFAULT_SOCKET_PATH.to_owned()),
			pid_file: raw.pid_file.unwrap_or_else(|| DEFAULT_PID_PATH.to_owned()),
		})
	}
}

/// Preferred inference device, configured per deployment. On the OpenVINO
/// backend (opt-in `openvino` feature) it selects the OpenVINO device; on the
/// default ncnn backend `Npu`/`Auto` request the Vulkan GPU (with automatic
/// CPU fallback when no Vulkan device is present) and `Cpu` forces CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DevicePref {
	/// OpenVINO "NPU" device (Intel NPU, e.g. Arrow Lake); falls back to CPU
	/// when the NPU driver or plugin is unavailable.
	#[default]
	Npu,
	/// OpenVINO "CPU" device.
	Cpu,
	/// Alias for `Npu` (kept for config compatibility). We deliberately avoid
	/// OpenVINO's "AUTO:NPU,CPU" meta-plugin: a broken NPU plugin install
	/// segfaults inside AUTO instead of returning a catchable error, which
	/// would defeat the CPU fallback.
	Auto,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePrefError(String);

impl Display for DevicePrefError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "invalid inference device preference '{}'", self.0)
	}
}

impl std::error::Error for DevicePrefError {}

impl DevicePref {
	pub fn parse(value: &str) -> Result<Self, DevicePrefError> {
		match value.to_ascii_uppercase().as_str() {
			"NPU" => Ok(Self::Npu),
			"CPU" => Ok(Self::Cpu),
			"AUTO" | "" => Ok(Self::Auto),
			other => Err(DevicePrefError(other.to_owned())),
		}
	}
}

impl fmt::Display for DevicePref {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Npu => write!(f, "NPU"),
			Self::Cpu => write!(f, "CPU"),
			Self::Auto => write!(f, "AUTO"),
		}
	}
}
