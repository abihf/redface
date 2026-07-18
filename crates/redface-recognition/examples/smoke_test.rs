// Smoke test: loads the real ONNX models through the active backend (OpenCV
// DNN CPU by default, OpenVINO when built with --features openvino) and runs
// one inference pass. Run: cargo run -p redface-recognition --example smoke_test
// DEVICE only selects the OpenVINO device on openvino builds (default: NPU,
// falls back to CPU); the default OpenCV DNN CPU backend ignores it.
// FRAME=blank (default) uses a flat gray frame and asserts no detections;
// FRAME=noise uses a deterministic noise frame and only prints the count.
use redface_recognition::{DevicePref, Recognizer};

fn main() {
	let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "data".to_owned());
	let device = std::env::var("DEVICE")
		.ok()
		.map(|value| DevicePref::parse(&value).expect("valid DEVICE"))
		.unwrap_or_default();
	eprintln!("loading recognizer (device {device})...");
	let mut recognizer = Recognizer::new(&model_dir, device).expect("recognizer loads");
	eprintln!("recognizer loaded");

	let frame_kind = std::env::var("FRAME").unwrap_or_else(|_| "blank".to_owned());
	let frame = match frame_kind.as_str() {
		// 340x340 mid-gray frame — no face, but exercises the full pipeline.
		"blank" => vec![128u8; 340 * 340 * 3],
		"noise" => noise_frame(),
		other => panic!("invalid FRAME '{other}': expected blank or noise"),
	};
	eprintln!("running recognize...");
	let faces = recognizer.recognize(&frame, 340, 340, 0).expect("inference works");
	eprintln!("recognize done");

	if frame_kind == "blank" {
		println!("OK: detections on blank frame: {}", faces.len());
		assert!(faces.is_empty(), "blank frame should yield no faces");
	} else {
		println!("OK: detections on noise frame: {}", faces.len());
	}
}

/// Deterministic 340x340 gray noise frame from a tiny seeded xorshift PRNG
/// (gray is replicated across the 3 channels).
fn noise_frame() -> Vec<u8> {
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut frame = vec![0u8; 340 * 340 * 3];
	for pixel in frame.chunks_mut(3) {
		// xorshift64
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		let value = (state >> 56) as u8;
		pixel[0] = value;
		pixel[1] = value;
		pixel[2] = value;
	}
	frame
}
