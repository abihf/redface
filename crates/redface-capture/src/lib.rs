use std::fmt;
use std::io;
use std::path::PathBuf;

use v4l::buffer::Type;
use v4l::format::{Format, FourCC};
use v4l::framesize::{FrameSize, FrameSizeEnum};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::Device;

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
                    write!(
                        f,
                        "no supported color format found: {}",
                        advertised.join(", ")
                    )
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
            Self::UnsupportedSelectedFormat(fourcc) => write!(
                f,
                "selected format is not supported for color conversion: {fourcc}"
            ),
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
        Self {
            device: device.into(),
        }
    }

    pub fn stream(
        &self,
        mut on_frame: impl FnMut(Frame) -> StreamAction,
    ) -> Result<CaptureStats, CaptureError> {
        let device = Device::with_path(&self.device).map_err(|source| CaptureError::OpenDevice {
            path: self.device.clone(),
            source,
        })?;

        let formats = device.enum_formats().map_err(CaptureError::ListFormats)?;
        let selected_fourcc = select_preferred_format(formats.iter().map(|desc| desc.fourcc))
            .ok_or_else(|| CaptureError::NoSupportedFormats {
                advertised: formats.iter().map(|desc| desc.fourcc.to_string()).collect(),
            })?;

        let frame_sizes = device
            .enum_framesizes(selected_fourcc)
            .map_err(|source| CaptureError::ListFrameSizes {
                fourcc: selected_fourcc,
                source,
            })?;
        let (width, height) = select_largest_dimensions(&frame_sizes)
            .ok_or(CaptureError::NoFrameSizes { fourcc: selected_fourcc })?;

        let requested = Format::new(width, height, selected_fourcc);
        let active = device
            .set_format(&requested)
            .map_err(|source| CaptureError::SetFormat { requested, source })?;
        if !is_supported_color_format(active.fourcc) {
            return Err(CaptureError::UnsupportedSelectedFormat(active.fourcc));
        }

        let mut stream =
            MmapStream::with_buffers(&device, Type::VideoCapture, BUFFER_COUNT).map_err(
                CaptureError::CreateStream,
            )?;

        let mut stats = CaptureStats::default();
        loop {
            let (raw, _) = stream.next().map_err(CaptureError::ReadFrame)?;

            if active.fourcc == grey_fourcc() && !has_good_black_level(raw) {
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
    let mut rgb = Vec::with_capacity(gray.len() * 3);
    for pixel in gray {
        rgb.extend_from_slice(&[*pixel, *pixel, *pixel]);
    }
    rgb
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
        rgb.push(y1.wrapping_sub(u.wrapping_sub(128) / 3).wrapping_sub(v.wrapping_sub(128) / 3));
        rgb.push(y1.wrapping_add(u.wrapping_sub(128).wrapping_mul(2) / 3));

        rgb.push(y2.wrapping_add(v.wrapping_sub(128).wrapping_mul(2) / 3));
        rgb.push(y2.wrapping_sub(u.wrapping_sub(128) / 3).wrapping_sub(v.wrapping_sub(128) / 3));
        rgb.push(y2.wrapping_add(u.wrapping_sub(128).wrapping_mul(2) / 3));
    }
    rgb
}

fn has_good_black_level(img: &[u8]) -> bool {
    if img.is_empty() {
        return false;
    }

    let dark = img.iter().filter(|pixel| **pixel < 80).count();
    let darkness = 100 * dark / img.len();
    darkness > 5 && darkness < 95
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
        let selected =
            select_preferred_format([yuyv_fourcc(), grey_fourcc(), rgb3_fourcc()])
                .expect("format selected");

        assert_eq!(selected, grey_fourcc());
    }

    #[test]
    fn picks_largest_width_and_height() {
        let selected = select_largest_dimensions(&[
            frame_size(320, 240),
            frame_size(640, 200),
            frame_size(300, 480),
        ]);

        assert_eq!(selected, Some((640, 480)));
    }

    #[test]
    fn expands_greyscale_pixels_to_rgb_triplets() {
        assert_eq!(gray_to_rgb(&[10, 20]), vec![10, 10, 10, 20, 20, 20]);
    }

    #[test]
    fn detects_bad_black_levels_like_go_code() {
        assert!(!has_good_black_level(&[0; 10]));
        assert!(!has_good_black_level(&[255; 10]));
        assert!(has_good_black_level(&[0, 0, 10, 20, 100, 120, 180, 200, 220, 255]));
    }

    #[test]
    fn converts_yuyv_using_wrapping_math() {
        let rgb = yuyv_to_rgb(&[100, 128, 120, 128]);

        assert_eq!(rgb, vec![100, 100, 100, 120, 120, 120]);
    }
}