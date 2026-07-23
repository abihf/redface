// Benchmark: times recognize() on a blank 340x340 frame, which exercises the
// full preprocessing path (CLAHE, detector input, decode) plus detector
// inference. Run: cargo run --release -p redface-recognition --example bench_preprocess
// DEVICE selects the inference target (on the default ncnn backend NPU/AUTO
// mean the Vulkan GPU with CPU fallback, CPU forces CPU; on OpenVINO builds
// it picks the OpenVINO device). ITERATIONS overrides the count.
use std::time::Instant;

use redface_recognition::{DevicePref, Recognizer};

fn main() {
	let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "data".to_owned());
	let device = std::env::var("DEVICE")
		.ok()
		.map(|value| DevicePref::parse(&value).expect("valid DEVICE"))
		.unwrap_or_default();
	let iterations: u32 = std::env::var("ITERATIONS")
		.ok()
		.and_then(|value| value.parse().ok())
		.unwrap_or(50);

	eprintln!("loading recognizer (device {device})...");
	let mut recognizer = Recognizer::new(&model_dir, device).expect("recognizer loads");

	let frame = vec![128u8; 340 * 340];
	for _ in 0..3 {
		let _ = recognizer.recognize(&frame, 340, 340, 0).expect("inference works");
	}

	let start = Instant::now();
	for _ in 0..iterations {
		let _ = recognizer.recognize(&frame, 340, 340, 0).expect("inference works");
	}
	let elapsed = start.elapsed();

	println!(
		"{iterations} iterations: {:.3} ms/frame",
		elapsed.as_secs_f64() * 1000.0 / f64::from(iterations)
	);
}
