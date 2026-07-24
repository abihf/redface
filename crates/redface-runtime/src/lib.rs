use std::fmt;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{
	Arc,
	atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use redface_capture::{Camera, CaptureError, StreamAction};
use redface_core::{Descriptor, DescriptorError, OSDNotification, read_descriptors};
use redface_recognition::{Recognizer, RecognizerError};

#[derive(Clone, Debug)]
pub struct VerifyOptions {
	pub device: PathBuf,
	pub face_file: PathBuf,
	pub timeout: Option<Duration>,
	pub threshold: f64,
	/// When set and flagged, the capture loop stops at the next frame (e.g.
	/// the daemon sets this when the client disconnects mid-verification).
	pub cancel: Option<Arc<AtomicBool>>,
	/// When set and flagged, the capture loop stops at the next frame (e.g.
	/// the daemon sets this when the OSD sends Cancelling).
	pub osd_cancel: Option<Arc<AtomicBool>>,
}

#[derive(Debug)]
pub enum VerifyError {
	OpenModel { path: PathBuf, source: io::Error },
	ParseModel(DescriptorError),
	Capture(CaptureError),
	Recognition(RecognizerError),
	Timeout(Duration),
	Cancelled,
}

impl fmt::Display for VerifyError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::OpenModel { path, source } => write!(f, "failed to open model {}: {source}", path.display()),
			Self::ParseModel(source) => write!(f, "failed to parse model file: {source}"),
			Self::Capture(source) => source.fmt(f),
			Self::Recognition(source) => source.fmt(f),
			Self::Timeout(duration) => write!(f, "timeout {duration:?}"),
			Self::Cancelled => write!(f, "cancelled"),
		}
	}
}

impl std::error::Error for VerifyError {}

pub fn verify(
	recognizer: &mut Recognizer,
	options: &VerifyOptions,
	mut on_event: impl FnMut(OSDNotification),
) -> Result<bool, VerifyError> {
	let descriptors = load_descriptors(&options.face_file)?;
	let camera = Camera::new(&options.device);
	let started = Instant::now();
	let mut matched = false;
	let mut timed_out = false;
	let mut cancelled = false;
	let mut fatal_error = None;
	let mut no_face_frames = 0usize;
	let mut frame_count = 0u64;
	let mut was_mismatch = false;

	camera
		.stream(|frame| {
			frame_count += 1;
			if options.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
				cancelled = true;
				return StreamAction::Stop;
			}
			if options.osd_cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
				cancelled = true;
				return StreamAction::Stop;
			}
			if let Some(timeout) = options.timeout
				&& timeout > Duration::ZERO
				&& started.elapsed() >= timeout
			{
				timed_out = true;
				return StreamAction::Stop;
			}

			// Throttle Verifying to ~2 Hz (every 30 frames at 60 fps).
			if frame_count % 30 == 0 {
				on_event(OSDNotification::Verifying);
			}

			let rec_start = Instant::now();
			// 1:1 authentication against a known user: encode only the
			// best-scoring detection — the detector routinely reports the
			// same face several times at different scales.
			let faces = match recognizer.recognize(&frame.buffer, frame.width, frame.height, 1) {
				Ok(faces) => faces,
				Err(err) => {
					fatal_error = Some(VerifyError::Recognition(err));
					return StreamAction::Stop;
				}
			};

			if faces.is_empty() {
				no_face_frames += 1;
				if was_mismatch {
					was_mismatch = false;
					on_event(OSDNotification::Verifying);
				}
				return StreamAction::Continue;
			}

			println!("* Found {} faces in {:?}", faces.len(), rec_start.elapsed());
		for (index, face) in faces.iter().enumerate() {
			print!("  - Face [{}]:", index);
			for descriptor in &descriptors {
				let similarity = descriptor.cosine_similarity(&face.descriptor);
				print!(" {:.3}", similarity);
				if similarity > options.threshold {
					println!(" (found)");
					matched = true;
					on_event(OSDNotification::Success);
					return StreamAction::Stop;
				}
			}
			println!();
		}

		// Faces seen but none matched the enrolled descriptors.
		was_mismatch = true;
		on_event(OSDNotification::FaceMismatch);

			StreamAction::Continue
		})
		.map_err(VerifyError::Capture)?;

	if let Some(err) = fatal_error {
		return Err(err);
	}
	if cancelled {
		return Err(VerifyError::Cancelled);
	}
	if timed_out {
		return Err(VerifyError::Timeout(options.timeout.unwrap_or_default()));
	}
	if no_face_frames > 0 {
		println!("> Frames without face found: {}\n", no_face_frames);
	}

	Ok(matched)
}

fn load_descriptors(path: &Path) -> Result<Vec<Descriptor>, VerifyError> {
	let file = File::open(path).map_err(|source| VerifyError::OpenModel {
		path: path.to_path_buf(),
		source,
	})?;
	read_descriptors(BufReader::new(file)).map_err(VerifyError::ParseModel)
}
