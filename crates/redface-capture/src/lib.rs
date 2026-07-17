use std::fmt;
use std::io;
use std::path::PathBuf;

#[cfg(target_arch = "x86")]
use std::arch::x86::{
	__m128i, __m256i, __m512i, _mm_loadu_si128, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128,
	_mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm512_extracti32x4_epi32,
	_mm512_loadu_si512,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
	__m128i, __m256i, __m512i, _mm_loadu_si128, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128,
	_mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm512_extracti32x4_epi32,
	_mm512_loadu_si512,
};

use v4l::Device;
use v4l::buffer::Type;
use v4l::format::{Format, FourCC};
use v4l::framesize::{FrameSize, FrameSizeEnum};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;

const BUFFER_COUNT: u32 = 4;

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

		let mut stream =
			MmapStream::with_buffers(&device, Type::VideoCapture, BUFFER_COUNT).map_err(CaptureError::CreateStream)?;

		let mut stats = CaptureStats::default();
		loop {
			let (raw, _) = stream.next().map_err(CaptureError::ReadFrame)?;

			if active.fourcc == grey_fourcc() && is_black_frame(raw) {
				stats.dropped_frames += 1;
				continue;
			}

			let buffer = convert_to_rgb(active.fourcc, raw);
			stats.delivered_frames += 1;

			let action = on_frame(Frame {
				buffer,
				width: active.width,
				height: active.height,
			});
			if matches!(action, StreamAction::Stop) {
				return Ok(stats);
			}
		}
	}
}

fn select_preferred_format(formats: impl IntoIterator<Item = FourCC>) -> Option<FourCC> {
	let available = formats.into_iter().collect::<Vec<_>>();
	preferred_formats()
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
	fourcc == grey_fourcc() || fourcc == rgb3_fourcc() || fourcc == yuyv_fourcc()
}

fn convert_to_rgb(fourcc: FourCC, raw: &[u8]) -> Vec<u8> {
	if fourcc == grey_fourcc() {
		gray_to_rgb(raw)
	} else if fourcc == rgb3_fourcc() {
		raw.to_vec()
	} else if fourcc == yuyv_fourcc() {
		yuyv_to_rgb(raw)
	} else {
		raw.to_vec()
	}
}

fn preferred_formats() -> [FourCC; 3] {
	[grey_fourcc(), rgb3_fourcc(), yuyv_fourcc()]
}

fn grey_fourcc() -> FourCC {
	FourCC::new(b"GREY")
}

fn rgb3_fourcc() -> FourCC {
	FourCC::new(b"RGB3")
}

fn yuyv_fourcc() -> FourCC {
	FourCC::new(b"YUYV")
}

fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
	let mut rgb = vec![0_u8; gray.len() * 3];

	#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
	{
		if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") {
			// Safe because the runtime feature checks guarantee the required AVX-512 and SSSE3 support.
			unsafe { gray_to_rgb_avx512(gray, &mut rgb) };
			return rgb;
		}
		if std::arch::is_x86_feature_detected!("avx2") {
			// Safe because the runtime feature checks guarantee the required AVX2 and SSSE3 support.
			unsafe { gray_to_rgb_avx2(gray, &mut rgb) };
			return rgb;
		}
		if std::arch::is_x86_feature_detected!("ssse3") {
			// Safe because the runtime feature check guarantees SSSE3 support.
			unsafe { gray_to_rgb_ssse3(gray, &mut rgb) };
			return rgb;
		}
	}

	gray_to_rgb_scalar(gray, &mut rgb);
	rgb
}

fn gray_to_rgb_scalar(gray: &[u8], rgb: &mut [u8]) {
	for (index, pixel) in gray.iter().copied().enumerate() {
		let offset = index * 3;
		rgb[offset] = pixel;
		rgb[offset + 1] = pixel;
		rgb[offset + 2] = pixel;
	}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx2,ssse3")]
unsafe fn gray_to_rgb_avx512(gray: &[u8], rgb: &mut [u8]) {
	let masks = unsafe { shuffle_masks() };

	let mut input_offset = 0;
	let mut output_offset = 0;
	while input_offset + 64 <= gray.len() {
		// Safe because the loop bounds guarantee 64 readable input bytes.
		let pixels = unsafe { _mm512_loadu_si512(gray.as_ptr().add(input_offset).cast::<__m512i>()) };

		unsafe { store_rgb_block_16(_mm512_extracti32x4_epi32::<0>(pixels), &masks, rgb, output_offset) };
		unsafe { store_rgb_block_16(_mm512_extracti32x4_epi32::<1>(pixels), &masks, rgb, output_offset + 48) };
		unsafe { store_rgb_block_16(_mm512_extracti32x4_epi32::<2>(pixels), &masks, rgb, output_offset + 96) };
		unsafe { store_rgb_block_16(_mm512_extracti32x4_epi32::<3>(pixels), &masks, rgb, output_offset + 144) };

		input_offset += 64;
		output_offset += 192;
	}

	if input_offset < gray.len() {
		unsafe { gray_to_rgb_avx2(&gray[input_offset..], &mut rgb[output_offset..]) };
	}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,ssse3")]
unsafe fn gray_to_rgb_avx2(gray: &[u8], rgb: &mut [u8]) {
	let masks = unsafe { shuffle_masks() };

	let mut input_offset = 0;
	let mut output_offset = 0;
	while input_offset + 32 <= gray.len() {
		// Safe because the loop bounds guarantee 32 readable input bytes.
		let pixels = unsafe { _mm256_loadu_si256(gray.as_ptr().add(input_offset).cast::<__m256i>()) };
		unsafe { store_rgb_block_16(_mm256_castsi256_si128(pixels), &masks, rgb, output_offset) };
		unsafe { store_rgb_block_16(_mm256_extracti128_si256::<1>(pixels), &masks, rgb, output_offset + 48) };

		input_offset += 32;
		output_offset += 96;
	}

	if input_offset < gray.len() {
		unsafe { gray_to_rgb_ssse3(&gray[input_offset..], &mut rgb[output_offset..]) };
	}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn gray_to_rgb_ssse3(gray: &[u8], rgb: &mut [u8]) {
	let masks = unsafe { shuffle_masks() };

	let mut input_offset = 0;
	let mut output_offset = 0;
	while input_offset + 16 <= gray.len() {
		// Safe because the loop bounds guarantee 16 readable input bytes.
		let pixels = unsafe { _mm_loadu_si128(gray.as_ptr().add(input_offset).cast::<__m128i>()) };
		unsafe { store_rgb_block_16(pixels, &masks, rgb, output_offset) };

		input_offset += 16;
		output_offset += 48;
	}

	gray_to_rgb_tail(gray, input_offset, rgb, output_offset);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn gray_to_rgb_tail(gray: &[u8], input_offset: usize, rgb: &mut [u8], output_offset: usize) {
	for (index, pixel) in gray[input_offset..].iter().copied().enumerate() {
		let offset = output_offset + index * 3;
		rgb[offset] = pixel;
		rgb[offset + 1] = pixel;
		rgb[offset + 2] = pixel;
	}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn shuffle_masks() -> (__m128i, __m128i, __m128i) {
	(
		_mm_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5),
		_mm_setr_epi8(5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10),
		_mm_setr_epi8(10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15),
	)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn store_rgb_block_16(
	pixels: __m128i,
	masks: &(__m128i, __m128i, __m128i),
	rgb: &mut [u8],
	output_offset: usize,
) {
	// Safe because output_offset always points at a full 48-byte block reserved in the RGB buffer.
	let out_ptr = unsafe { rgb.as_mut_ptr().add(output_offset).cast::<__m128i>() };
	let (mask0, mask1, mask2) = *masks;

	// Safe because the destination buffer is pre-sized and each store writes exactly one 16-byte lane.
	unsafe {
		_mm_storeu_si128(out_ptr, _mm_shuffle_epi8(pixels, mask0));
		_mm_storeu_si128(out_ptr.add(1), _mm_shuffle_epi8(pixels, mask1));
		_mm_storeu_si128(out_ptr.add(2), _mm_shuffle_epi8(pixels, mask2));
	}
}

fn yuyv_to_rgb(yuyv: &[u8]) -> Vec<u8> {
	if yuyv.is_empty() {
		return Vec::new();
	}

	let mut rgb = Vec::with_capacity(yuyv.len() * 6 / 4);
	for chunk in yuyv.chunks_exact(4) {
		let y1 = chunk[0];
		let u = chunk[1];
		let y2 = chunk[2];
		let v = chunk[3];

		rgb.push(y1.wrapping_add(v.wrapping_sub(128).wrapping_mul(2) / 3));
		rgb.push(
			y1.wrapping_sub(u.wrapping_sub(128) / 3)
				.wrapping_sub(v.wrapping_sub(128) / 3),
		);
		rgb.push(y1.wrapping_add(u.wrapping_sub(128).wrapping_mul(2) / 3));

		rgb.push(y2.wrapping_add(v.wrapping_sub(128).wrapping_mul(2) / 3));
		rgb.push(
			y2.wrapping_sub(u.wrapping_sub(128) / 3)
				.wrapping_sub(v.wrapping_sub(128) / 3),
		);
		rgb.push(y2.wrapping_add(u.wrapping_sub(128).wrapping_mul(2) / 3));
	}
	rgb
}

/// A GREY frame counts as black — IR illuminator off, lens covered, or
/// exposure not yet settled — when more than half of its pixels sit near the
/// bottom of the range. Such frames can never contain a detectable face, so
/// they are dropped before conversion and inference. Unlike a "black level"
/// check, a bright frame with few dark pixels is fine: close-up face frames
/// are exactly that.
const BLACK_PIXEL: u8 = 32;
const BLACK_FRAME_DARK_RATIO: usize = 2;

fn is_black_frame(img: &[u8]) -> bool {
	if img.is_empty() {
		return true;
	}

	let dark = img.iter().filter(|pixel| **pixel < BLACK_PIXEL).count();
	dark * BLACK_FRAME_DARK_RATIO > img.len()
}

#[cfg(test)]
mod tests {
	use super::*;
	use v4l::framesize::{Discrete, FrameSizeEnum};

	fn frame_size(width: u32, height: u32) -> FrameSize {
		FrameSize {
			index: 0,
			fourcc: grey_fourcc(),
			typ: 0,
			size: FrameSizeEnum::Discrete(Discrete { width, height }),
		}
	}

	#[test]
	fn prefers_grey_over_other_supported_formats() {
		let selected = select_preferred_format([yuyv_fourcc(), grey_fourcc(), rgb3_fourcc()]).expect("format selected");

		assert_eq!(selected, grey_fourcc());
	}

	#[test]
	fn picks_largest_width_and_height() {
		let selected = select_largest_dimensions(&[frame_size(320, 240), frame_size(640, 200), frame_size(300, 480)]);

		assert_eq!(selected, Some((640, 480)));
	}

	#[test]
	fn expands_greyscale_pixels_to_rgb_triplets() {
		assert_eq!(gray_to_rgb(&[10, 20]), vec![10, 10, 10, 20, 20, 20]);
	}

	#[test]
	fn expands_long_greyscale_input_exactly() {
		let gray = (0_u8..32).collect::<Vec<_>>();
		let rgb = gray_to_rgb(&gray);

		assert_eq!(rgb.len(), gray.len() * 3);
		for (index, pixel) in gray.into_iter().enumerate() {
			let offset = index * 3;
			assert_eq!(&rgb[offset..offset + 3], &[pixel, pixel, pixel]);
		}
	}

	#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
	#[test]
	fn avx2_matches_scalar_when_available() {
		if !(std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("ssse3")) {
			return;
		}

		let gray = (0_u8..65).collect::<Vec<_>>();
		let mut expected = vec![0_u8; gray.len() * 3];
		gray_to_rgb_scalar(&gray, &mut expected);
		let mut actual = vec![0_u8; gray.len() * 3];
		unsafe { gray_to_rgb_avx2(&gray, &mut actual) };

		assert_eq!(actual, expected);
	}

	#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
	#[test]
	fn avx512_matches_scalar_when_available() {
		if !(std::arch::is_x86_feature_detected!("avx512f")
			&& std::arch::is_x86_feature_detected!("avx512bw")
			&& std::arch::is_x86_feature_detected!("ssse3"))
		{
			return;
		}

		let gray = (0_u8..97).collect::<Vec<_>>();
		let mut expected = vec![0_u8; gray.len() * 3];
		gray_to_rgb_scalar(&gray, &mut expected);
		let mut actual = vec![0_u8; gray.len() * 3];
		unsafe { gray_to_rgb_avx512(&gray, &mut actual) };

		assert_eq!(actual, expected);
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

	#[test]
	fn converts_yuyv_using_wrapping_math() {
		let rgb = yuyv_to_rgb(&[100, 128, 120, 128]);

		assert_eq!(rgb, vec![100, 100, 100, 120, 120, 120]);
	}
}
