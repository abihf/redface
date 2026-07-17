// Smoke test: loads the real ONNX models via OpenVINO and runs one inference
// pass. Run: cargo run -p redface-recognition --example smoke_test
// DEVICE=CPU selects the OpenVINO device (default: NPU, falls back to CPU).
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

	// 340x340 mid-gray frame — no face, but exercises the full pipeline.
	let frame = vec![128u8; 340 * 340 * 3];
	eprintln!("running recognize...");
	let faces = recognizer.recognize(&frame, 340, 340, 0).expect("inference works");
	eprintln!("recognize done");

	println!("OK: detections on blank frame: {}", faces.len());
	assert!(faces.is_empty(), "blank frame should yield no faces");
}
