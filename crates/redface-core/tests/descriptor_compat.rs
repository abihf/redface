use std::io::Cursor;

use redface_core::{DESCRIPTOR_LEN, Descriptor, read_descriptors, write_descriptors};

/// Deterministic synthetic descriptors, so the tests need no fixture file.
fn sample_descriptors() -> Vec<Descriptor> {
	(0..20u32)
		.map(|row| {
			let mut values = [0.0_f32; DESCRIPTOR_LEN];
			for (col, slot) in values.iter_mut().enumerate() {
				let seed = row
					.wrapping_mul(DESCRIPTOR_LEN as u32)
					.wrapping_add(col as u32)
					.wrapping_mul(2654435761);
				let unit = (seed % 1000) as f32 / 1000.0; // [0,1)
				*slot = (unit - 0.5) * 0.04;
			}
			Descriptor(values)
		})
		.collect()
}

#[test]
fn encoded_descriptors_are_parseable_again() {
	let descriptors = sample_descriptors();

	let mut buffer = Vec::new();
	write_descriptors(&mut buffer, &descriptors).expect("writes descriptors");

	let reparsed = read_descriptors(Cursor::new(buffer)).expect("round trip parses");
	assert_eq!(reparsed, descriptors);
}

#[test]
fn parser_rejects_wrong_descriptor_length() {
	let err = Descriptor::parse_line("0x1p+00 0x1p+00").expect_err("short line should fail");

	assert!(err.to_string().contains("expected 512 descriptor values"));
}
