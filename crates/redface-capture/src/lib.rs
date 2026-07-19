use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use v4l::Device;
use v4l::buffer::Type;
use v4l::format::{Format, FourCC};
use v4l::framesize::{FrameSize, FrameSizeEnum};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;

const BUFFER_COUNT: u32 = 4;

const GREY_FOURCC: FourCC = FourCC { repr: *b"GREY" };
const RGB3_FOURCC: FourCC = FourCC { repr: *b"RGB3" };
const YUYV_FOURCC: FourCC = FourCC { repr: *b"YUYV" };

const PREFERRED_FORMATS: [FourCC; 3] = [GREY_FOURCC, RGB3_FOURCC, YUYV_FOURCC];

/// A single camera frame, one byte per pixel (grayscale). NIR cameras
/// deliver GREY natively; RGB and YUYV fallbacks are converted to luma.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
	pub buffer: Vec<u8>,
	pub width: u32,
	pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamAction {
	Continue,
	Stop,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureStats {
	pub dropped_frames: u32,
	pub delivered_frames: u32,
}

/// Shared hand-off slot between the capture thread (producer) and the caller
/// thread running `on_frame` (consumer). The producer always overwrites
/// `frame` with the newest buffer, so any frame that arrives while the
/// consumer is still processing the previous one is dropped. `stop` tells the
/// producer to exit; `error` carries a fatal capture failure to the consumer.
#[derive(Default)]
struct FrameSlot {
	frame: Option<Frame>,
	stats: CaptureStats,
	error: Option<CaptureError>,
	stop: bool,
}

#[derive(Debug)]
pub enum CaptureError {
	OpenDevice { path: PathBuf, source: io::Error },
	ListFormats(io::Error),
	NoSupportedFormats { advertised: Vec<String> },
	ListFrameSizes { fourcc: FourCC, source: io::Error },
	NoFrameSizes { fourcc: FourCC },
	SetFormat { requested: Format, source: io::Error },
	UnsupportedSelectedFormat(FourCC),
	CreateStream(io::Error),
	ReadFrame(io::Error),
}

impl fmt::Display for CaptureError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::OpenDevice { path, source } => {
				write!(f, "can not open device {}: {source}", path.display())
			}
			Self::ListFormats(source) => write!(f, "can not enumerate device formats: {source}"),
			Self::NoSupportedFormats { advertised } => {
				if advertised.is_empty() {
					write!(f, "no supported formats found")
				} else {
					write!(f, "no supported color format found: {}", advertised.join(", "))
				}
			}
			Self::ListFrameSizes { fourcc, source } => {
				write!(f, "can not enumerate frame sizes for {fourcc}: {source}")
			}
			Self::NoFrameSizes { fourcc } => write!(f, "no frame sizes available for {fourcc}"),
			Self::SetFormat { requested, source } => write!(
				f,
				"can not set image format {} {}x{}: {source}",
				requested.fourcc, requested.width, requested.height
			),
			Self::UnsupportedSelectedFormat(fourcc) => {
				write!(f, "selected format is not supported for color conversion: {fourcc}")
			}
			Self::CreateStream(source) => write!(f, "can not start streaming: {source}"),
			Self::ReadFrame(source) => write!(f, "read frame failed: {source}"),
		}
	}
}

impl std::error::Error for CaptureError {}

pub struct Camera {
	device: PathBuf,
}

impl Camera {
	pub fn new(device: impl Into<PathBuf>) -> Self {
		Self { device: device.into() }
	}

	pub fn stream(&self, mut on_frame: impl FnMut(Frame) -> StreamAction) -> Result<CaptureStats, CaptureError> {
		let device = Device::with_path(&self.device).map_err(|source| CaptureError::OpenDevice {
			path: self.device.clone(),
			source,
		})?;

		let formats = device.enum_formats().map_err(CaptureError::ListFormats)?;
		let selected_fourcc = select_preferred_format(formats.iter().map(|desc| desc.fourcc)).ok_or_else(|| {
			CaptureError::NoSupportedFormats {
				advertised: formats.iter().map(|desc| desc.fourcc.to_string()).collect(),
			}
		})?;

		let frame_sizes = device
			.enum_framesizes(selected_fourcc)
			.map_err(|source| CaptureError::ListFrameSizes {
				fourcc: selected_fourcc,
				source,
			})?;
		let (width, height) = select_largest_dimensions(&frame_sizes).ok_or(CaptureError::NoFrameSizes {
			fourcc: selected_fourcc,
		})?;

		let requested = Format::new(width, height, selected_fourcc);
		let active = device
			.set_format(&requested)
			.map_err(|source| CaptureError::SetFormat { requested, source })?;
		if !is_supported_color_format(active.fourcc) {
			return Err(CaptureError::UnsupportedSelectedFormat(active.fourcc));
		}

		let stream =
			MmapStream::with_buffers(&device, Type::VideoCapture, BUFFER_COUNT).map_err(CaptureError::CreateStream)?;

		let slot = Arc::new((Mutex::new(FrameSlot::default()), Condvar::new()));
		let producer_slot = Arc::clone(&slot);
		let fourcc = active.fourcc;
		let (width, height) = (active.width, active.height);

		let producer = thread::spawn(move || {
			let (lock, ready) = &*producer_slot;
			let mut stream = stream;
			loop {
				let (raw, _) = match stream.next() {
					Ok(frame) => frame,
					Err(source) => {
						let mut slot = lock.lock().unwrap();
						slot.error = Some(CaptureError::ReadFrame(source));
						ready.notify_one();
						return;
					}
				};

				{
					let mut slot = lock.lock().unwrap();
					if slot.frame.is_some() {
						slot.stats.dropped_frames += 1;
						continue;
					}
				}

				if fourcc == GREY_FOURCC && is_black_frame(raw) {
					lock.lock().unwrap().stats.dropped_frames += 1;
					continue;
				}

				let frame = Frame {
					buffer: convert_to_gray(fourcc, raw),
					width,
					height,
				};

				let mut slot = lock.lock().unwrap();
				if slot.stop {
					return;
				}
				slot.frame = Some(frame);
				ready.notify_one();
			}
		});

		let (lock, ready) = &*slot;
		let outcome: Result<(), CaptureError> = loop {
			let frame = {
				let mut slot = lock.lock().unwrap();
				while slot.frame.is_none() && slot.error.is_none() {
					slot = ready.wait(slot).unwrap();
				}
				if let Some(error) = slot.error.take() {
					slot.stop = true;
					break Err(error);
				}
				slot.stats.delivered_frames += 1;
				slot.frame.take().expect("frame present when error is none")
			};

			if matches!(on_frame(frame), StreamAction::Stop) {
				lock.lock().unwrap().stop = true;
				break Ok(());
			}
		};

		let _ = producer.join();
		let stats = lock.lock().unwrap().stats;
		outcome.map(|()| stats)
	}
}

fn select_preferred_format(formats: impl IntoIterator<Item = FourCC>) -> Option<FourCC> {
	let available = formats.into_iter().collect::<Vec<_>>();
	PREFERRED_FORMATS
		.into_iter()
		.find(|preferred| available.iter().any(|candidate| candidate == preferred))
}

fn select_largest_dimensions(frame_sizes: &[FrameSize]) -> Option<(u32, u32)> {
	let mut width = 0;
	let mut height = 0;

	for frame_size in frame_sizes {
		match &frame_size.size {
			FrameSizeEnum::Discrete(discrete) => {
				width = width.max(discrete.width);
				height = height.max(discrete.height);
			}
			FrameSizeEnum::Stepwise(stepwise) => {
				width = width.max(stepwise.max_width);
				height = height.max(stepwise.max_height);
			}
		}
	}

	if width == 0 || height == 0 {
		None
	} else {
		Some((width, height))
	}
}

fn is_supported_color_format(fourcc: FourCC) -> bool {
	fourcc == GREY_FOURCC || fourcc == RGB3_FOURCC || fourcc == YUYV_FOURCC
}

fn convert_to_gray(fourcc: FourCC, raw: &[u8]) -> Vec<u8> {
	if fourcc == GREY_FOURCC {
		raw.to_vec()
	} else if fourcc == RGB3_FOURCC {
		rgb_to_gray(raw)
	} else if fourcc == YUYV_FOURCC {
		yuyv_to_gray(raw)
	} else {
		raw.to_vec()
	}
}

/// RGB3 → grayscale via BT.601 luma coefficients, integer approximation:
/// (77R + 150G + 29B) >> 8. Rare fallback path (NIR cameras prefer GREY).
fn rgb_to_gray(rgb: &[u8]) -> Vec<u8> {
	rgb.chunks_exact(3)
		.map(|pixel| ((77 * u16::from(pixel[0]) + 150 * u16::from(pixel[1]) + 29 * u16::from(pixel[2])) >> 8) as u8)
		.collect()
}

/// YUYV → grayscale: extract the Y (luma) bytes. YUYV packs two pixels per
/// 4-byte chunk [Y0, Cb, Y1, Cr], so the Y values sit at even indices.
fn yuyv_to_gray(yuyv: &[u8]) -> Vec<u8> {
	yuyv.iter().step_by(2).copied().collect()
}

/// A GREY frame counts as black — IR illuminator off, lens covered, or
/// exposure not yet settled — when more than half of its pixels sit near the
/// bottom of the range. Such frames can never contain a detectable face, so
/// they are dropped before conversion and inference. Unlike a "black level"
/// check, a bright frame with few dark pixels is fine: close-up face frames
/// are exactly that.
const BLACK_PIXEL: u8 = 32;
const BLACK_FRAME_DARK_RATIO: f32 = 0.9;

fn is_black_frame(img: &[u8]) -> bool {
	if img.is_empty() {
		return true;
	}

	let dark = img.iter().filter(|pixel| **pixel < BLACK_PIXEL).count();
	// println!("Black frame check: {} dark pixels out of {} total", dark, img.len());
	dark as f32 > BLACK_FRAME_DARK_RATIO * img.len() as f32
}

#[cfg(test)]
mod tests {
	use super::*;
	use v4l::framesize::{Discrete, FrameSizeEnum};

	fn frame_size(width: u32, height: u32) -> FrameSize {
		FrameSize {
			index: 0,
			fourcc: GREY_FOURCC,
			typ: 0,
			size: FrameSizeEnum::Discrete(Discrete { width, height }),
		}
	}

	#[test]
	fn prefers_grey_over_other_supported_formats() {
		let selected = select_preferred_format([YUYV_FOURCC, GREY_FOURCC, RGB3_FOURCC]).expect("format selected");

		assert_eq!(selected, GREY_FOURCC);
	}

	#[test]
	fn picks_largest_width_and_height() {
		let selected = select_largest_dimensions(&[frame_size(320, 240), frame_size(640, 200), frame_size(300, 480)]);

		assert_eq!(selected, Some((640, 480)));
	}

	#[test]
	fn grey_passes_through_unchanged() {
		assert_eq!(convert_to_gray(GREY_FOURCC, &[10, 20, 30]), vec![10, 20, 30]);
	}

	#[test]
	fn rgb_to_gray_uses_luma_coefficients() {
		// Pure red, green, blue.
		assert_eq!(rgb_to_gray(&[255, 0, 0]), vec![76]); // 77*255>>8 = 76
		assert_eq!(rgb_to_gray(&[0, 255, 0]), vec![149]); // 150*255>>8 = 149
		assert_eq!(rgb_to_gray(&[0, 0, 255]), vec![28]); // 29*255>>8 = 28
		// White: (77+150+29)*255>>8 = 255.
		assert_eq!(rgb_to_gray(&[255, 255, 255]), vec![255]);
	}

	#[test]
	fn yuyv_extracts_luma_bytes() {
		// [Y0=100, Cb, Y1=120, Cr] → [100, 120]
		assert_eq!(yuyv_to_gray(&[100, 128, 120, 128]), vec![100, 120]);
		assert_eq!(yuyv_to_gray(&[]), Vec::<u8>::new());
	}

	#[test]
	fn detects_black_frames() {
		// Empty input counts as black.
		assert!(is_black_frame(&[]));
		// Fully black frame.
		assert!(is_black_frame(&[10; 100]));
		// More than half of the pixels below the threshold.
		let mut frame = vec![200_u8; 100];
		for pixel in frame.iter_mut().take(60) {
			*pixel = 10;
		}
		assert!(is_black_frame(&frame));
	}

	#[test]
	fn keeps_bright_frames() {
		// Uniform mid-gray close-up: no near-black pixels at all.
		assert!(!is_black_frame(&[180; 100]));
		// Up to half of the pixels may be dark.
		let mut frame = vec![200_u8; 100];
		for pixel in frame.iter_mut().take(50) {
			*pixel = 10;
		}
		assert!(!is_black_frame(&frame));
		// Fully saturated is not a *black* frame.
		assert!(!is_black_frame(&[255; 100]));
	}
}
