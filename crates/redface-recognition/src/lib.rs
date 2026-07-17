use std::ffi::c_void;
use std::fmt;
use std::path::{Path, PathBuf};

use opencv::core::{AlgorithmHint, BORDER_REPLICATE, CV_8UC3, CV_32F, Mat, Ptr, Scalar, Size};
use opencv::prelude::*;
use opencv::{dnn, imgproc};
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, Model, PartialShape, Shape, Tensor};
pub use redface_core::{DESCRIPTOR_LEN, Descriptor};

mod simd;

// Model files distributed in the InsightFace `buffalo_l` pack:
// https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
//
// det_10g.onnx (SCRFD-10G detector): input `input.1` [1, 3, H, W] (dynamic H/W,
// we run it at 640x640), 9 outputs in stride-major order — for strides
// 8/16/32: scores [N,1], bbox distances [N,4] (distance2bbox), landmark
// distances [N,10] (distance2kps). SCRFD predicts 2 anchors per feature-map
// point (N = 2 * (640/stride)^2); a point's anchors are adjacent, so entry i
// belongs to feature point i / 2.
const DETECTOR_MODEL: &str = "det_10g.onnx";
// w600k_r50.onnx (ArcFace ResNet-50): input `input.1` [N, 3, 112, 112], output
// `683` [1, 512]. Input is an aligned face crop, BGR, normalized (x-127.5)/127.5.
const ENCODER_MODEL: &str = "w600k_r50.onnx";

const DETECTOR_INPUT_SIZE: usize = 640;
const DETECTOR_CONF_THRESHOLD: f32 = 0.5;
const DETECTOR_NMS_THRESHOLD: f32 = 0.4;
const STRIDES: [usize; 3] = [8, 16, 32];
const SCRFD_NUM_ANCHORS: usize = 2;
const ENCODER_INPUT_SIZE: usize = 112;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rectangle {
	pub left: i64,
	pub top: i64,
	pub right: i64,
	pub bottom: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Face {
	pub rectangle: Rectangle,
	pub descriptor: Descriptor,
}

/// Preferred inference device, configured per deployment.
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

impl DevicePref {
	pub fn parse(value: &str) -> Result<Self, RecognizerError> {
		match value.to_ascii_uppercase().as_str() {
			"NPU" => Ok(Self::Npu),
			"CPU" => Ok(Self::Cpu),
			"AUTO" | "" => Ok(Self::Auto),
			other => Err(RecognizerError::InvalidDevice(other.to_owned())),
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

#[derive(Debug, PartialEq)]
pub enum RecognizerError {
	Setup(String),
	ModelLoad {
		path: PathBuf,
		message: String,
	},
	InvalidImageBuffer {
		width: u32,
		height: u32,
		expected_len: usize,
		actual_len: usize,
	},
	InvalidDevice(String),
	Inference(String),
}

impl fmt::Display for RecognizerError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Setup(message) => write!(f, "failed to initialize OpenVINO: {message}"),
			Self::ModelLoad { path, message } => {
				write!(f, "failed to load model '{}': {message}", path.display())
			}
			Self::InvalidImageBuffer {
				width,
				height,
				expected_len,
				actual_len,
			} => write!(
				f,
				"invalid RGB buffer for {width}x{height} image: expected {expected_len} bytes, got {actual_len}"
			),
			Self::InvalidDevice(device) => {
				write!(f, "invalid device '{device}': expected NPU, CPU or AUTO")
			}
			Self::Inference(message) => write!(f, "inference failed: {message}"),
		}
	}
}

impl std::error::Error for RecognizerError {}

/// A compiled model together with its input port name and infer request.
struct ModelRunner {
	input_name: String,
	_compiled: CompiledModel,
	request: InferRequest,
}

pub struct Recognizer {
	// The core must outlive the compiled models.
	_core: Core,
	detector: ModelRunner,
	encoder: ModelRunner,
	clahe: Ptr<imgproc::CLAHE>,
}

impl Recognizer {
	pub fn new(model_dir: impl AsRef<Path>, device: DevicePref) -> Result<Self, RecognizerError> {
		let model_dir = model_dir.as_ref();
		let detector_path = model_dir.join(DETECTOR_MODEL);
		let encoder_path = model_dir.join(ENCODER_MODEL);

		let mut core = Core::new().map_err(|err| RecognizerError::Setup(err.to_string()))?;
		let detector = build_runner(
			&mut core,
			&detector_path,
			[1, 3, DETECTOR_INPUT_SIZE as i64, DETECTOR_INPUT_SIZE as i64],
			device,
		)?;
		let encoder = build_runner(
			&mut core,
			&encoder_path,
			[1, 3, ENCODER_INPUT_SIZE as i64, ENCODER_INPUT_SIZE as i64],
			device,
		)?;

		Ok(Self {
			_core: core,
			detector,
			encoder,
			clahe: new_clahe()?,
		})
	}

	pub fn recognize(
		&mut self,
		img_data: &[u8],
		width: u32,
		height: u32,
		max_faces: usize,
	) -> Result<Vec<Face>, RecognizerError> {
		validate_rgb_buffer(width, height, img_data.len())?;

		// CLAHE contrast normalization: raw IR frames are often low-contrast,
		// which RGB-trained CNN detectors handle poorly. Applied before both
		// detection and encoding (the recipe Howdy and Visage use on these
		// cameras).
		let img = equalize_frame(&mut self.clahe, img_data, width as i32, height as i32)?;

		let detections = self.detect(&img, width as usize, height as usize)?;

		if detections.is_empty() {
			return Ok(Vec::new());
		}
		if max_faces > 0 && detections.len() > max_faces {
			return Ok(Vec::new());
		}

		let mut faces = Vec::with_capacity(detections.len());
		for detection in detections {
			let crop = align_face(&img, &detection.landmarks)?;
			let descriptor = self.encode(&crop)?;
			faces.push(Face {
				rectangle: detection.rectangle,
				descriptor,
			});
		}

		Ok(faces)
	}

	fn detect(&mut self, img: &Mat, width: usize, height: usize) -> Result<Vec<Detection>, RecognizerError> {
		let blob = detector_input(img)?;
		let input = input_tensor(
			[1, 3, DETECTOR_INPUT_SIZE as i64, DETECTOR_INPUT_SIZE as i64],
			blob.data_typed::<f32>().map_err(inference_error)?,
		)?;

		self.detector
			.request
			.set_tensor(&self.detector.input_name, &input)
			.map_err(inference_error)?;
		self.detector.request.infer().map_err(inference_error)?;

		let mut branch_data: Vec<Vec<f32>> = Vec::with_capacity(9);
		for index in 0..9 {
			let output = self
				.detector
				.request
				.get_output_tensor_by_index(index)
				.map_err(inference_error)?;
			branch_data.push(output.get_data::<f32>().map_err(inference_error)?.to_vec());
		}

		let ratio = width as f32 / DETECTOR_INPUT_SIZE as f32;
		Ok(decode_detections(&branch_data, ratio, width, height))
	}

	fn encode(&mut self, crop: &[f32]) -> Result<Descriptor, RecognizerError> {
		let input = input_tensor([1, 3, ENCODER_INPUT_SIZE as i64, ENCODER_INPUT_SIZE as i64], crop)?;

		self.encoder
			.request
			.set_tensor(&self.encoder.input_name, &input)
			.map_err(inference_error)?;
		self.encoder.request.infer().map_err(inference_error)?;

		let output = self
			.encoder
			.request
			.get_output_tensor_by_index(0)
			.map_err(inference_error)?;
		let embedding = output.get_data::<f32>().map_err(inference_error)?;

		if embedding.len() != DESCRIPTOR_LEN {
			return Err(RecognizerError::Inference(format!(
				"encoder returned {} values, expected {DESCRIPTOR_LEN}",
				embedding.len()
			)));
		}

		let mut descriptor = [0.0_f32; DESCRIPTOR_LEN];
		descriptor.copy_from_slice(embedding);
		Ok(Descriptor(descriptor))
	}
}

/// Reads the model at `path`, reshapes its (dynamic) input to `input_shape` —
/// the NPU plugin requires fully static shapes — and compiles it for `device`,
/// falling back to CPU when the preferred device is unavailable.
fn build_runner(
	core: &mut Core,
	path: &Path,
	input_shape: [i64; 4],
	device: DevicePref,
) -> Result<ModelRunner, RecognizerError> {
	let model_error = |err: openvino::InferenceError| RecognizerError::ModelLoad {
		path: path.to_path_buf(),
		message: err.to_string(),
	};

	let path_str = path.to_str().ok_or_else(|| RecognizerError::ModelLoad {
		path: path.to_path_buf(),
		message: "path is not valid UTF-8".to_owned(),
	})?;

	// The weights path is ignored for ONNX models.
	let mut model = core.read_model_from_file(path_str, "").map_err(model_error)?;
	model
		.reshape_single_input(&PartialShape::new_static(4, &input_shape).map_err(model_error)?)
		.map_err(model_error)?;
	let input_name = model
		.get_input_by_index(0)
		.and_then(|input| input.get_name())
		.map_err(model_error)?;

	let mut compiled = compile_with_fallback(core, &model, device).map_err(model_error)?;
	let request = compiled.create_infer_request().map_err(model_error)?;

	Ok(ModelRunner {
		input_name,
		_compiled: compiled,
		request,
	})
}

fn compile_with_fallback(
	core: &mut Core,
	model: &Model,
	device: DevicePref,
) -> Result<CompiledModel, openvino::InferenceError> {
	let target = match device {
		DevicePref::Cpu => return core.compile_model(model, DeviceType::CPU),
		DevicePref::Npu | DevicePref::Auto => "NPU",
	};

	core.compile_model(model, DeviceType::from(target)).or_else(|err| {
		eprintln!("redface: OpenVINO device '{target}' unavailable ({err}); falling back to CPU");
		core.compile_model(model, DeviceType::CPU)
	})
}

/// Allocates an OpenVINO f32 tensor of `shape` and copies `data` into it.
fn input_tensor(shape: [i64; 4], data: &[f32]) -> Result<Tensor, RecognizerError> {
	let shape = Shape::new(&shape).map_err(inference_error)?;
	let mut tensor = Tensor::new(ElementType::F32, &shape).map_err(inference_error)?;
	tensor
		.get_data_mut::<f32>()
		.map_err(inference_error)?
		.copy_from_slice(data);
	Ok(tensor)
}

fn inference_error(err: impl fmt::Display) -> RecognizerError {
	RecognizerError::Inference(err.to_string())
}

#[derive(Clone, Debug)]
struct Detection {
	rectangle: Rectangle,
	landmarks: [(f32, f32); 5],
}

/// Resizes the frame to the detector's static input size and converts to NCHW
/// f32 normalized as (x - 127.5) / 128, per the SCRFD reference. OpenCV's
/// blob_from_image resizes with INTER_LINEAR (the InsightFace reference
/// preprocessing).
fn detector_input(img: &Mat) -> Result<Mat, RecognizerError> {
	dnn::blob_from_image(
		img,
		1.0 / 128.0,
		Size::new(DETECTOR_INPUT_SIZE as i32, DETECTOR_INPUT_SIZE as i32),
		Scalar::all(127.5),
		false,
		false,
		CV_32F,
	)
	.map_err(inference_error)
}

/// Decodes SCRFD anchor-free outputs. `branches` holds the 9 output tensors in
/// model order: [score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16,
/// kps32]. Each branch has SCRFD_NUM_ANCHORS entries per feature-map point,
/// adjacent per point. `ratio` maps model-input pixels back to original frame
/// pixels. Scores are pre-filtered with an AVX2 scan (scalar fallback), see
/// `simd`.
fn decode_detections(branches: &[Vec<f32>], ratio: f32, width: usize, height: usize) -> Vec<Detection> {
	let mut candidates: Vec<(Detection, f32)> = Vec::new();

	for (level, stride) in STRIDES.iter().enumerate() {
		let scores = &branches[level];
		let bboxes = &branches[3 + level];
		let kps = &branches[6 + level];
		let fmap = DETECTOR_INPUT_SIZE / stride;

		for index in simd::above_threshold(scores, DETECTOR_CONF_THRESHOLD) {
			let index = index as usize;
			let score = scores[index];

			let point = index / SCRFD_NUM_ANCHORS;
			let cx = ((point % fmap) as f32) * *stride as f32;
			let cy = ((point / fmap) as f32) * *stride as f32;

			let distances = &bboxes[index * 4..index * 4 + 4];
			let x1 = (cx - distances[0] * *stride as f32) * ratio;
			let y1 = (cy - distances[1] * *stride as f32) * ratio;
			let x2 = (cx + distances[2] * *stride as f32) * ratio;
			let y2 = (cy + distances[3] * *stride as f32) * ratio;

			let kps_distances = &kps[index * 10..index * 10 + 10];
			let mut landmarks = [(0.0_f32, 0.0_f32); 5];
			for (point, landmark) in landmarks.iter_mut().enumerate() {
				landmark.0 = (cx + kps_distances[point * 2] * *stride as f32) * ratio;
				landmark.1 = (cy + kps_distances[point * 2 + 1] * *stride as f32) * ratio;
			}

			candidates.push((
				Detection {
					rectangle: Rectangle {
						left: clamp_coord(x1, width),
						top: clamp_coord(y1, height),
						right: clamp_coord(x2, width),
						bottom: clamp_coord(y2, height),
					},
					landmarks,
				},
				score,
			));
		}
	}

	nms(candidates, DETECTOR_NMS_THRESHOLD)
}

/// CLAHE tile grid (8x8) and clip limit, the OpenCV `cv::createCLAHE()`
/// defaults that Howdy uses on the same cameras.
const CLAHE_GRID: i32 = 8;
const CLAHE_CLIP_LIMIT: f64 = 2.0;

fn new_clahe() -> Result<Ptr<imgproc::CLAHE>, RecognizerError> {
	imgproc::create_clahe(CLAHE_CLIP_LIMIT, Size::new(CLAHE_GRID, CLAHE_GRID))
		.map_err(|err| RecognizerError::Setup(err.to_string()))
}

/// Contrast-limited adaptive histogram equalization of the grayscale plane of
/// an RGB24 buffer (gray is replicated across the 3 channels; the equalized
/// gray is replicated back into all three). OpenCV's CLAHE does the SIMD-heavy
/// work.
fn equalize_frame(
	clahe: &mut Ptr<imgproc::CLAHE>,
	img_data: &[u8],
	width: i32,
	height: i32,
) -> Result<Mat, RecognizerError> {
	// SAFETY: `img_data` outlives `frame`, OpenCV never frees external data,
	// and `frame` is only used as a const cvt_color source.
	let frame =
		unsafe { Mat::new_rows_cols_with_data_unsafe_def(height, width, CV_8UC3, img_data.as_ptr() as *mut c_void) }
			.map_err(inference_error)?;

	let mut gray = Mat::default();
	imgproc::cvt_color_def(&frame, &mut gray, imgproc::COLOR_RGB2GRAY).map_err(inference_error)?;
	let mut equalized = Mat::default();
	clahe.apply(&gray, &mut equalized).map_err(inference_error)?;
	let mut rgb = Mat::default();
	imgproc::cvt_color_def(&equalized, &mut rgb, imgproc::COLOR_GRAY2RGB).map_err(inference_error)?;
	Ok(rgb)
}

fn clamp_coord(value: f32, limit: usize) -> i64 {
	value.round().clamp(0.0, limit as f32) as i64
}

fn nms(mut candidates: Vec<(Detection, f32)>, threshold: f32) -> Vec<Detection> {
	candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
	let mut kept: Vec<(Detection, f32)> = Vec::with_capacity(candidates.len());
	for candidate in candidates {
		let dominated = kept
			.iter()
			.any(|(existing, _)| iou(&existing.rectangle, &candidate.0.rectangle) > threshold);
		if !dominated {
			kept.push(candidate);
		}
	}
	kept.into_iter().map(|(detection, _)| detection).collect()
}

fn iou(a: &Rectangle, b: &Rectangle) -> f32 {
	let left = a.left.max(b.left);
	let top = a.top.max(b.top);
	let right = a.right.min(b.right);
	let bottom = a.bottom.min(b.bottom);
	if right <= left || bottom <= top {
		return 0.0;
	}
	let intersection = ((right - left) * (bottom - top)) as f32;
	let area = |r: &Rectangle| ((r.right - r.left) * (r.bottom - r.top)) as f32;
	intersection / (area(a) + area(b) - intersection)
}

/// Standard ArcFace 112x112 alignment template (as used by InsightFace's
/// `face_align.norm_crop` for 5-point landmarks).
const ARCFACE_TEMPLATE: [(f32, f32); 5] = [
	(38.2946, 51.6963),
	(73.5318, 51.5014),
	(56.0252, 71.7366),
	(41.5493, 92.3655),
	(70.7299, 92.2041),
];

/// Warps the frame to an aligned ENCODER_INPUT_SIZE² face crop in NCHW f32,
/// BGR order, normalized (x - 127.5) / 127.5.
fn align_face(img: &Mat, landmarks: &[(f32, f32); 5]) -> Result<Vec<f32>, RecognizerError> {
	let m = similarity_transform(landmarks, &ARCFACE_TEMPLATE);
	let matrix = Mat::from_slice_2d(&[[m[0], m[1], m[2]], [m[3], m[4], m[5]]]).map_err(inference_error)?;

	let size = Size::new(ENCODER_INPUT_SIZE as i32, ENCODER_INPUT_SIZE as i32);
	let mut crop = Mat::default();
	imgproc::warp_affine(
		img,
		&mut crop,
		&matrix,
		size,
		imgproc::INTER_LINEAR,
		BORDER_REPLICATE,
		Scalar::all(0.0),
		AlgorithmHint::ALGO_HINT_DEFAULT,
	)
	.map_err(inference_error)?;

	let blob = dnn::blob_from_image(&crop, 1.0 / 127.5, size, Scalar::all(127.5), true, false, CV_32F)
		.map_err(inference_error)?;
	Ok(blob.data_typed::<f32>().map_err(inference_error)?.to_vec())
}

/// Estimates the 2D similarity transform mapping `src` landmarks to `dst`
/// (Umeyama, no reflection), returned as a 2x3 row-major matrix
/// `[a, -b, tx, b, a, ty]`.
fn similarity_transform(src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> [f32; 6] {
	let n = src.len() as f32;
	let (mut sx, mut sy, mut dx, mut dy) = (0.0, 0.0, 0.0, 0.0);
	for i in 0..src.len() {
		sx += src[i].0;
		sy += src[i].1;
		dx += dst[i].0;
		dy += dst[i].1;
	}
	let (sx, sy, dx, dy) = (sx / n, sy / n, dx / n, dy / n);

	let mut var_src = 0.0;
	let mut dot = 0.0;
	let mut cross = 0.0;
	for i in 0..src.len() {
		let (ux, uy) = (src[i].0 - sx, src[i].1 - sy);
		let (vx, vy) = (dst[i].0 - dx, dst[i].1 - dy);
		var_src += ux * ux + uy * uy;
		dot += ux * vx + uy * vy;
		cross += ux * vy - uy * vx;
	}

	if var_src <= f32::EPSILON {
		// Degenerate landmarks: identity mapping.
		return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
	}

	// dst ≈ s·R(θ)·src + t; with a = s·cosθ, b = s·sinθ the least-squares
	// solution is a = Σ(u·v)/Σ|u|², b = Σ(u×v)/Σ|u|².
	let a = dot / var_src;
	let b = cross / var_src;
	let tx = dx - a * sx + b * sy;
	let ty = dy - b * sx - a * sy;
	[a, -b, tx, b, a, ty]
}

fn validate_rgb_buffer(width: u32, height: u32, actual_len: usize) -> Result<(), RecognizerError> {
	let expected_len = (width as usize)
		.checked_mul(height as usize)
		.and_then(|pixels| pixels.checked_mul(3))
		.ok_or(RecognizerError::InvalidImageBuffer {
			width,
			height,
			expected_len: usize::MAX,
			actual_len,
		})?;

	if actual_len != expected_len {
		return Err(RecognizerError::InvalidImageBuffer {
			width,
			height,
			expected_len,
			actual_len,
		});
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	use opencv::core::Vec3b;

	#[test]
	fn rejects_wrong_rgb_buffer_length() {
		let err = validate_rgb_buffer(2, 2, 11).expect_err("invalid buffer should fail");

		assert_eq!(
			err.to_string(),
			"invalid RGB buffer for 2x2 image: expected 12 bytes, got 11"
		);
	}

	#[test]
	fn accepts_exact_rgb_buffer_length() {
		assert!(validate_rgb_buffer(2, 2, 12).is_ok());
	}

	#[test]
	fn parses_device_pref_values() {
		assert_eq!(DevicePref::parse("NPU"), Ok(DevicePref::Npu));
		assert_eq!(DevicePref::parse("cpu"), Ok(DevicePref::Cpu));
		assert_eq!(DevicePref::parse("Auto"), Ok(DevicePref::Auto));
		assert_eq!(DevicePref::parse(""), Ok(DevicePref::Auto));
		assert!(DevicePref::parse("TPU").is_err());
		assert_eq!(DevicePref::default(), DevicePref::Npu);
	}

	#[test]
	fn clahe_keeps_flat_image_flat() {
		let mut clahe = new_clahe().expect("clahe");
		let img = vec![128u8; 32 * 32 * 3];

		let out = equalize_frame(&mut clahe, &img, 32, 32).expect("equalize");
		let pixels = out.data_typed::<Vec3b>().expect("pixels");

		assert!(pixels.iter().all(|pixel| *pixel == pixels[0]));
	}

	#[test]
	fn clahe_expands_low_contrast_range() {
		// Pseudo-random noise squeezed into [100, 140], like a low-contrast
		// IR frame.
		let size = 340;
		let mut img = vec![0u8; size * size * 3];
		for index in 0..size * size {
			let seed = (index as u64).wrapping_mul(2654435761) & 0xffff_ffff;
			let value = 100 + (seed % 41) as u8;
			for channel in 0..3 {
				img[index * 3 + channel] = value;
			}
		}
		let mut clahe = new_clahe().expect("clahe");

		let out = equalize_frame(&mut clahe, &img, size as i32, size as i32).expect("equalize");
		let pixels = out.data_typed::<Vec3b>().expect("pixels");

		let min = pixels.iter().map(|pixel| pixel[0]).min().unwrap();
		let max = pixels.iter().map(|pixel| pixel[0]).max().unwrap();
		assert!(max - min > 90, "expected range expansion, got {min}..{max}");
		// Channels stay replicated.
		assert!(pixels.iter().all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2]));
	}

	#[test]
	fn detector_input_normalizes_and_resizes() {
		// 2x2 image, pixels chosen so channels differ.
		let img = Mat::from_slice_2d(&[
			[Vec3b::from([255, 0, 0]), Vec3b::from([0, 255, 0])],
			[Vec3b::from([0, 0, 255]), Vec3b::from([127, 127, 127])],
		])
		.expect("mat");

		let blob = detector_input(&img).expect("blob");
		let tensor = blob.data_typed::<f32>().expect("blob data");

		let size = DETECTOR_INPUT_SIZE;
		// Top-left of the resized tensor maps to source pixel (0,0): R=255.
		assert!((tensor[0] - (255.0 - 127.5) / 128.0).abs() < 1e-4);
		// Bottom-right maps to source pixel (1,1): R=G=B=127.
		let last = size * size - 1;
		assert!((tensor[last] - (127.0 - 127.5) / 128.0).abs() < 1e-4);
		assert!((tensor[size * size + last] - (127.0 - 127.5) / 128.0).abs() < 1e-4);
	}

	#[test]
	fn decode_produces_box_and_landmarks_in_frame_coordinates() {
		// Single detection at stride 8, feature point (10, 20), anchor 0.
		let fmap = DETECTOR_INPUT_SIZE / 8;
		let entries = fmap * fmap * SCRFD_NUM_ANCHORS;
		let mut scores = vec![0.0_f32; entries];
		let mut bboxes = vec![0.0_f32; entries * 4];
		let mut kps = vec![0.0_f32; entries * 10];

		let index = (20 * fmap + 10) * SCRFD_NUM_ANCHORS;
		scores[index] = 0.9;
		bboxes[index * 4..index * 4 + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
		for i in 0..10 {
			kps[index * 10 + i] = 0.5;
		}

		// Empty branches for strides 16/32 (2 anchors per point as well).
		let empty16_s = vec![0.0_f32; (DETECTOR_INPUT_SIZE / 16).pow(2) * SCRFD_NUM_ANCHORS];
		let empty16_b = vec![0.0_f32; (DETECTOR_INPUT_SIZE / 16).pow(2) * SCRFD_NUM_ANCHORS * 4];
		let empty16_k = vec![0.0_f32; (DETECTOR_INPUT_SIZE / 16).pow(2) * SCRFD_NUM_ANCHORS * 10];
		let empty32_s = vec![0.0_f32; (DETECTOR_INPUT_SIZE / 32).pow(2) * SCRFD_NUM_ANCHORS];
		let empty32_b = vec![0.0_f32; (DETECTOR_INPUT_SIZE / 32).pow(2) * SCRFD_NUM_ANCHORS * 4];
		let empty32_k = vec![0.0_f32; (DETECTOR_INPUT_SIZE / 32).pow(2) * SCRFD_NUM_ANCHORS * 10];

		let branches = vec![
			scores, empty16_s, empty32_s, bboxes, empty16_b, empty32_b, kps, empty16_k, empty32_k,
		];

		let ratio = 0.5;
		let detections = decode_detections(&branches, ratio, 320, 320);

		assert_eq!(detections.len(), 1);
		let det = &detections[0];
		// cx = 10*8 = 80, cy = 20*8 = 160 (model coords)
		// x1 = (80 - 1*8)*0.5 = 36, y1 = (160 - 2*8)*0.5 = 72
		// x2 = (80 + 3*8)*0.5 = 52, y2 = (160 + 4*8)*0.5 = 96
		assert_eq!(det.rectangle.left, 36);
		assert_eq!(det.rectangle.top, 72);
		assert_eq!(det.rectangle.right, 52);
		assert_eq!(det.rectangle.bottom, 96);
		// landmark x = (80 + 0.5*8)*0.5 = 42, y = (160 + 0.5*8)*0.5 = 82
		assert!((det.landmarks[0].0 - 42.0).abs() < 1e-4);
		assert!((det.landmarks[0].1 - 82.0).abs() < 1e-4);
	}

	#[test]
	fn nms_suppresses_overlapping_lower_score_boxes() {
		let rect = |l: i64| Rectangle {
			left: l,
			top: 0,
			right: l + 100,
			bottom: 100,
		};
		let landmarks = [(0.0, 0.0); 5];
		let candidates = vec![
			(
				Detection {
					rectangle: rect(0),
					landmarks,
				},
				0.9,
			),
			(
				Detection {
					rectangle: rect(10), // ~82% IoU with the first
					landmarks,
				},
				0.8,
			),
			(
				Detection {
					rectangle: rect(500), // disjoint
					landmarks,
				},
				0.7,
			),
		];

		let kept = nms(candidates, 0.4);

		assert_eq!(kept.len(), 2);
		assert_eq!(kept[0].rectangle.left, 0);
		assert_eq!(kept[1].rectangle.left, 500);
	}

	/// 2x3 row-major affine application, for transform assertions.
	fn apply(m: [f32; 6], x: f32, y: f32) -> (f32, f32) {
		(m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
	}

	#[test]
	fn similarity_transform_identity_when_src_equals_dst() {
		let points = [(10.0, 10.0), (50.0, 12.0), (30.0, 40.0), (15.0, 60.0), (45.0, 58.0)];
		let m = similarity_transform(&points, &points);

		let (x, y) = apply(m, 25.0, 33.0);
		assert!((x - 25.0).abs() < 1e-3, "x={x}");
		assert!((y - 33.0).abs() < 1e-3, "y={y}");
	}

	#[test]
	fn similarity_transform_maps_src_to_dst() {
		let src = [(10.0, 10.0), (50.0, 12.0), (30.0, 40.0), (15.0, 60.0), (45.0, 58.0)];
		// dst = 2 * src + (5, -7)
		let mut dst = [(0.0, 0.0); 5];
		for (i, p) in src.iter().enumerate() {
			dst[i] = (p.0 * 2.0 + 5.0, p.1 * 2.0 - 7.0);
		}

		let m = similarity_transform(&src, &dst);
		for (s, d) in src.iter().zip(dst.iter()) {
			let (x, y) = apply(m, s.0, s.1);
			assert!((x - d.0).abs() < 1e-3, "x={x} expected {}", d.0);
			assert!((y - d.1).abs() < 1e-3, "y={y} expected {}", d.1);
		}
	}

	#[test]
	fn similarity_transform_handles_rotation() {
		// 90° clockwise rotation about the origin plus scale 1: src x-axis
		// maps to dst -y-axis.
		let src = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 1.0)];
		let rotate = |p: (f32, f32)| (p.1, -p.0);
		let mut dst = [(0.0, 0.0); 5];
		for (i, p) in src.iter().enumerate() {
			dst[i] = rotate(*p);
		}

		let m = similarity_transform(&src, &dst);
		for (s, d) in src.iter().zip(dst.iter()) {
			let (x, y) = apply(m, s.0, s.1);
			assert!((x - d.0).abs() < 1e-3, "x={x} expected {}", d.0);
			assert!((y - d.1).abs() < 1e-3, "y={y} expected {}", d.1);
		}
	}

	#[test]
	fn align_face_output_shape_and_range() {
		let img = Mat::new_rows_cols_with_default(340, 340, CV_8UC3, Scalar::all(128.0)).expect("mat");
		let landmarks = [
			(120.0, 140.0),
			(220.0, 140.0),
			(170.0, 190.0),
			(130.0, 250.0),
			(210.0, 250.0),
		];

		let crop = align_face(&img, &landmarks).expect("align");

		assert_eq!(crop.len(), 3 * ENCODER_INPUT_SIZE * ENCODER_INPUT_SIZE);
		// Uniform gray input -> every normalized value is (128-127.5)/127.5.
		let expected = (128.0_f32 - 127.5) / 127.5;
		assert!(crop.iter().all(|&v| (v - expected).abs() < 0.05));
	}
}
