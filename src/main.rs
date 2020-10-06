extern crate defer;
extern crate opencv;

use opencv::{
	core::{self, Scalar, Size, UMat},
	dnn::{self, blob_from_image, read_net_from_caffe, read_net_from_torch, Net},
	highgui, imgproc,
	prelude::*,
	videoio::{self, VideoCapture},
};
use std::cell::Cell;
use std::error::Error;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

fn main() -> Result<()> {
	let opencl_have = core::have_opencl()?;
	if opencl_have {
		core::set_use_opencl(false)?;
	}
	capture_and_recognize(true)
}

fn capture_and_recognize(show_window: bool) -> Result<()> {
	let mut detector = FaceDetector::create()?;
	let mut extractor = FaceExtractor::create()?;

	let mut capture = VideoCapture::default()?;
	let opened = capture.open(0, videoio::CAP_V4L2)?;
	if !opened {
		return Ok(());
	}
	let vid_width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
	let vid_height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
	let frame_delay = (1000.0 / capture.get(videoio::CAP_PROP_FPS)?) as i32;
	let vid_size = core::Size::new(vid_width, vid_height);

	let mut last_feature = read_file("face.json")
		.or_else(|_| Mat::new_rows_cols_with_default(1, 128, core::CV_32F, Scalar::all(0.0)))?;

	let mut original_frame = UMat::new(core::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY)?;
	let mut frame = UMat::new(core::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY)?;
	let mut bgr_image = UMat::new_rows_cols_with_default(
		vid_width,
		vid_height,
		core::CV_8UC3,
		Scalar::default(),
		core::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY,
	)?;

	if show_window {
		highgui::named_window("redface", highgui::WINDOW_AUTOSIZE)?;
	}

	loop {
		let key_code = highgui::wait_key(frame_delay)?;
		if key_code == 27 {
			// escepe
			break;
		}
		let ok = capture.read(&mut original_frame)?;
		if !ok {
			continue;
		}
		core::flip(&original_frame, &mut frame, 1)?;
		imgproc::cvt_color(&frame, &mut bgr_image, imgproc::COLOR_RGBA2BGR, 0)?;

		let face_recs = match detector.detect(&bgr_image, &vid_size) {
			Ok(rec) => rec,
			Err(err) => {
				println!("Can not recognize face: {}", err);
				vec![]
			}
		};

		for (i, rec) in face_recs.iter().enumerate() {
			let mut face = UMat::roi(&bgr_image, *rec)?;
			// imgproc::get_rect_sub_pix(src, dst, dsize, fx, fy, interpolation)
			let features = match extractor.extract(&face) {
				Ok(val) => val,
				Err(err) => {
					println!("Can not extract face {}", err);
					face.release()?;
					continue;
				}
			};
			face.release()?;

			let distance = core::norm2(&features, &last_feature, core::NORM_L2, &Mat::default()?)?;
			let color = if distance < 0.07 {
				Scalar::new(0.0, 255.0, 0.0, 255.0)
			} else {
				Scalar::new(0.0, 0.0, 255.0, 255.0)
			};

			if i == 0 && key_code == 13 {
				// enter key
				write_file("face.json", &features)?;
				last_feature = features;
			}

			if show_window {
				imgproc::rectangle(&mut frame, *rec, color, 1, imgproc::LINE_8, 0)?;
				imgproc::put_text(
					&mut frame,
					format!("{:.6}", distance).as_str(),
					core::Point::new(rec.x, rec.y - 5),
					imgproc::FONT_HERSHEY_SCRIPT_SIMPLEX,
					0.75,
					color,
					2,
					imgproc::LINE_8,
					false,
				)?;
			}
		}

		if show_window {
			highgui::imshow("redface", &frame)?;
		}
	}
	Ok(())
}

fn read_file(filename: &str) -> Result<Mat> {
	// 24 = READ | FORMAT_JSON
	let mut file = core::FileStorage::new(filename, 24, "")?;
	let mat = file.get("mat")?.mat()?;
	file.release()?;
	Ok(mat)
}

fn write_file(filename: &str, mat: &Mat) -> Result<()> {
	// 25 = WRITE | FORMAT_JSON
	let mut file = core::FileStorage::new(filename, 25, "")?;
	file.write_mat("mat", mat)?;
	file.release()?;
	Ok(())
}

struct FaceDetector {
	net: Cell<Net>,
}

impl FaceDetector {
	fn create() -> Result<Self> {
		let mut net = read_net_from_caffe(
			"data/deploy_lowres.prototxt",
			"data/res10_300x300_ssd_iter_140000_fp16.caffemodel",
		)?;

		if core::use_opencl()? {
			net.set_preferable_target(dnn::DNN_TARGET_OPENCL)?;
		}

		Ok(Self {
			net: Cell::new(net),
		})
	}

	fn detect(&mut self, image: &UMat, size: &Size) -> Result<Vec<core::Rect>> {
		let net = self.net.get_mut();

		let scale = if size.width > size.height {
			300.0 / size.width as f32
		} else {
			300.0 / size.height as f32
		};
		let blob = blob_from_image(
			&image,
			1.0,
			core::Size::new(
				(size.width as f32 * scale) as i32,
				(size.height as f32 * scale) as i32,
			),
			Scalar::new(104.0, 177.0, 123.0, 0.0),
			false,
			false,
			core::CV_32F,
		)?;
		net.set_input(&blob, "", 1.0, Scalar::default())?;
		let output = net.forward_single("")?;

		let sizes = output.mat_size();
		let len = sizes.get(2).unwrap_or(0);
		let mut result = Vec::new();

		for i in 0..len {
			let confidence = *output.at_nd::<f32>(&[0, 0, i, 2])?;
			if confidence == 0.0 {
				break;
			} else if confidence < 0.5 {
				continue;
			}

			let left = output.at_nd::<f32>(&[0, 0, i, 3])?.max(0.0);
			let top = output.at_nd::<f32>(&[0, 0, i, 4])?.max(0.0);
			let right = output.at_nd::<f32>(&[0, 0, i, 5])?.min(1.0);
			let bottom = output.at_nd::<f32>(&[0, 0, i, 6])?.min(1.0);

			let x = (size.width as f32 * left) as i32;
			let y = (size.height as f32 * top) as i32;
			let width = (size.width as f32 * right) as i32 - x;
			let height = (size.height as f32 * bottom) as i32 - y;

			result.push(core::Rect::new(x, y, width, height));
		}

		Ok(result)
	}
}

struct FaceExtractor {
	net: Cell<Net>,
}

impl FaceExtractor {
	fn create() -> Result<Self> {
		// let mut net = read_net_from_torch("data/openface.nn4.small2.v1.t7", true, true)?;
		let mut net = read_net_from_torch("data/nn4.v2.t7", true, true)?;

		if core::use_opencl()? {
			net.set_preferable_target(dnn::DNN_TARGET_OPENCL)?;
		}

		Ok(Self {
			net: Cell::new(net),
		})
	}

	fn extract(&mut self, face: &UMat) -> Result<Mat> {
		let net = self.net.get_mut();
		let blob = blob_from_image(
			&face,
			1.0,
			core::Size::new(96, 96),
			Scalar::default(),
			true,
			false,
			core::CV_32F,
		)?;
		net.set_input(&blob, "", 1.0, Scalar::default())?;
		let result = net.forward_single("")?;
		// for i in 0..128 {
		//     let item: &f32 = result.at(i)?;
		//     print!("{} ", *item)
		// }
		let mut copied = Mat::default()?;
		result.copy_to(&mut copied)?;
		Ok(copied)
		// vec.slice
		// println!("veccc {}", vec.len());
		// Ok(())
	}
}
