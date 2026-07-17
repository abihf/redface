use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use redface_capture::{Camera, CaptureError, StreamAction};
use redface_core::{Descriptor, DescriptorError, read_descriptors};
use redface_recognition::{DevicePref, Recognizer, RecognizerError};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CONFIG_PATH: &str = "/etc/redface/config.json";
pub const DEFAULT_SOCKET_PATH: &str = "/var/run/redface.sock";
pub const DEFAULT_PID_PATH: &str = "/var/run/redface.pid";
pub const DEFAULT_DATA_DIR: &str = "/usr/share/redface";
pub const DEFAULT_MODELS_DIR: &str = "/etc/redface/models";

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
        let raw = serde_json::from_reader(BufReader::new(file)).map_err(|source| {
            ConfigError::Parse {
                path: path.to_path_buf(),
                source,
            }
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    #[serde(rename = "AUTH")]
    Authenticate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Req {
    pub action: Action,
    pub params: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthReq {
    pub client: String,
    pub user: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    #[serde(rename = "SUCCESS")]
    Success,
    #[serde(rename = "ERROR")]
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Res {
    pub status: Status,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub extras: std::collections::BTreeMap<String, String>,
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "SUCCESS",
            Self::Error => "ERROR",
        }
    }
}

pub fn read_req(reader: impl Read) -> serde_json::Result<Req> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    Req::deserialize(&mut deserializer)
}

pub fn read_res(reader: impl Read) -> serde_json::Result<Res> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    Res::deserialize(&mut deserializer)
}

pub fn to_auth_req(req: &Req) -> AuthReq {
    AuthReq {
        client: req.params.get("client").cloned().unwrap_or_default(),
        user: req.params.get("user").cloned().unwrap_or_default(),
    }
}

pub fn write_auth_req(mut writer: impl Write, user: &str, client: &str) -> serde_json::Result<()> {
    let req = Req {
        action: Action::Authenticate,
        params: std::collections::BTreeMap::from([
            ("client".to_owned(), client.to_owned()),
            ("user".to_owned(), user.to_owned()),
        ]),
    };
    serde_json::to_writer(&mut writer, &req)?;
    writer.write_all(b"\n").map_err(serde_json::Error::io)
}

pub fn write_success_res(mut writer: impl Write, extras: std::collections::BTreeMap<String, String>) -> serde_json::Result<()> {
    let res = Res {
        status: Status::Success,
        error: String::new(),
        extras,
    };
    serde_json::to_writer(&mut writer, &res)?;
    writer.write_all(b"\n").map_err(serde_json::Error::io)
}

pub fn write_error_res(mut writer: impl Write, err: &dyn std::error::Error) -> serde_json::Result<()> {
    let res = Res {
        status: Status::Error,
        error: err.to_string(),
        extras: std::collections::BTreeMap::new(),
    };
    serde_json::to_writer(&mut writer, &res)?;
    writer.write_all(b"\n").map_err(serde_json::Error::io)
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifyOptions {
    pub device: PathBuf,
    pub model_file: PathBuf,
    pub timeout: Duration,
    pub threshold: f64,
}

#[derive(Debug)]
pub enum VerifyError {
    OpenModel { path: PathBuf, source: io::Error },
    ParseModel(DescriptorError),
    Capture(CaptureError),
    Recognition(RecognizerError),
    Timeout(Duration),
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenModel { path, source } => write!(f, "failed to open model {}: {source}", path.display()),
            Self::ParseModel(source) => write!(f, "failed to parse model file: {source}"),
            Self::Capture(source) => source.fmt(f),
            Self::Recognition(source) => source.fmt(f),
            Self::Timeout(duration) => write!(f, "timeout {duration:?}"),
        }
    }
}

impl std::error::Error for VerifyError {}

pub fn verify(recognizer: &mut Recognizer, options: &VerifyOptions) -> Result<bool, VerifyError> {
    let models = load_models(&options.model_file)?;
    let camera = Camera::new(&options.device);
    let started = Instant::now();
    let mut matched = false;
    let mut timed_out = false;
    let mut fatal_error = None;
    let mut no_face_frames = 0usize;

    camera
        .stream(|frame| {
            if options.timeout > Duration::ZERO && started.elapsed() >= options.timeout {
                timed_out = true;
                return StreamAction::Stop;
            }

            let rec_start = Instant::now();
            let faces = match recognizer.recognize(&frame.buffer, frame.width, frame.height, 0) {
                Ok(faces) => faces,
                Err(err) => {
                    fatal_error = Some(VerifyError::Recognition(err));
                    return StreamAction::Stop;
                }
            };

            if faces.is_empty() {
                no_face_frames += 1;
                return StreamAction::Continue;
            }

            println!("* Found {} faces in {:?}", faces.len(), rec_start.elapsed());
            for (index, face) in faces.iter().enumerate() {
                print!("  - Face [{}]:", index);
                for model in &models {
                    let similarity = model.cosine_similarity(&face.descriptor);
                    print!(" {:.3}", similarity);
                    if similarity > options.threshold {
                        println!(" (found)");
                        matched = true;
                        return StreamAction::Stop;
                    }
                }
                println!();
            }

            StreamAction::Continue
        })
        .map_err(VerifyError::Capture)?;

    if let Some(err) = fatal_error {
        return Err(err);
    }
    if timed_out {
        return Err(VerifyError::Timeout(options.timeout));
    }
    if no_face_frames > 0 {
        println!("> Frames without face found: {}\n", no_face_frames);
    }

    Ok(matched)
}

fn load_models(path: &Path) -> Result<Vec<Descriptor>, VerifyError> {
    let file = File::open(path).map_err(|source| VerifyError::OpenModel {
        path: path.to_path_buf(),
        source,
    })?;
    read_descriptors(BufReader::new(file)).map_err(VerifyError::ParseModel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn config_applies_defaults() {
        let config = Config::from_raw(RawConfig {
            device: Some("/dev/video0".to_owned()),
            ..RawConfig::default()
        })
        .expect("config should parse");

        assert_eq!(config.inference_device, "NPU");
        assert_eq!(config.threshold, 0.9);
        assert_eq!(config.timeout, 10);
        assert_eq!(config.socket, DEFAULT_SOCKET_PATH);
        assert_eq!(config.pid_file, DEFAULT_PID_PATH);
    }

    #[test]
    fn config_rejects_invalid_inference_device() {
        let result = Config::from_raw(RawConfig {
            device: Some("/dev/video0".to_owned()),
            inference_device: Some("TPU".to_owned()),
            ..RawConfig::default()
        });

        assert!(matches!(
            result,
            Err(ConfigError::InvalidInferenceDevice { .. })
        ));
    }

    #[test]
    fn config_accepts_npu_inference_device() {
        let config = Config::from_raw(RawConfig {
            device: Some("/dev/video0".to_owned()),
            inference_device: Some("npu".to_owned()),
            ..RawConfig::default()
        })
        .expect("config should parse");

        assert_eq!(config.inference_device, "npu");
    }

    #[test]
    fn protocol_round_trip() {
        let mut buf = Vec::new();
        write_auth_req(&mut buf, "1000", "check").expect("write request");
        let req = read_req(Cursor::new(&buf)).expect("read request");

        assert_eq!(to_auth_req(&req), AuthReq { client: "check".into(), user: "1000".into() });
    }

    #[test]
    fn success_response_round_trip() {
        let mut buf = Vec::new();
        write_success_res(&mut buf, std::collections::BTreeMap::new()).expect("write response");
        let res = read_res(Cursor::new(&buf)).expect("read response");
        assert_eq!(res.status, Status::Success);
    }
}
