use std::io::Cursor;

use redface_core::{DESCRIPTOR_LEN, Descriptor, read_descriptors, write_descriptors};

const SAMPLE_FACE: &str = include_str!("../../../capture.face");

#[test]
fn reads_existing_face_file() {
    let descriptors = read_descriptors(Cursor::new(SAMPLE_FACE)).expect("sample .face parses");

    assert!(!descriptors.is_empty());
    assert_eq!(descriptors[0].0.len(), DESCRIPTOR_LEN);
}

#[test]
fn encoded_descriptors_are_parseable_again() {
    let descriptors = read_descriptors(Cursor::new(SAMPLE_FACE)).expect("sample .face parses");

    let mut buffer = Vec::new();
    write_descriptors(&mut buffer, &descriptors[..2]).expect("writes descriptors");

    let reparsed = read_descriptors(Cursor::new(buffer)).expect("round trip parses");
    assert_eq!(reparsed, descriptors[..2].to_vec());
}

#[test]
fn parser_rejects_wrong_descriptor_length() {
    let err = Descriptor::parse_line("0x1p+00 0x1p+00").expect_err("short line should fail");

    assert!(err.to_string().contains("expected 128 descriptor values"));
}