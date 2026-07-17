use std::io::{self, Write};

use redface_core::Descriptor;
use redface_recognition::Face;

#[derive(Debug, Clone, PartialEq)]
pub enum RecordEvent {
	NoFaceDetected,
	FaceRecorded { index: usize, similarity: f64 },
}

#[derive(Debug, Default)]
pub struct RecordSession {
	aggregate: Option<Descriptor>,
	no_face_frames: usize,
}

impl RecordSession {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn no_face_frames(&self) -> usize {
		self.no_face_frames
	}

	pub fn aggregate(&self) -> Option<&Descriptor> {
		self.aggregate.as_ref()
	}

	pub fn record_faces(&mut self, faces: &[Face], mut writer: impl Write) -> io::Result<Vec<RecordEvent>> {
		if faces.is_empty() {
			self.no_face_frames += 1;
			return Ok(vec![RecordEvent::NoFaceDetected]);
		}

		let mut events = Vec::with_capacity(faces.len());
		for (index, face) in faces.iter().enumerate() {
			let (similarity, descriptor_to_write) = match self.aggregate {
				Some(current) => {
					let similarity = face.descriptor.cosine_similarity(&current);
					let updated = current.middle(&face.descriptor);
					self.aggregate = Some(updated);
					(similarity, updated)
				}
				None => {
					self.aggregate = Some(face.descriptor);
					(1.0, face.descriptor)
				}
			};

			descriptor_to_write.write_line(&mut writer)?;
			writer.write_all(b"\n")?;
			events.push(RecordEvent::FaceRecorded { index, similarity });
		}

		Ok(events)
	}
}

#[cfg(test)]
mod tests {
	use redface_recognition::Rectangle;

	use super::*;

	fn face_with_scalar(value: f32) -> Face {
		Face {
			rectangle: Rectangle {
				left: 0,
				top: 0,
				right: 10,
				bottom: 10,
			},
			descriptor: Descriptor([value; redface_core::DESCRIPTOR_LEN]),
		}
	}

	#[test]
	fn first_face_is_written_as_initial_aggregate() {
		let mut session = RecordSession::new();
		let mut out = Vec::new();

		let events = session
			.record_faces(&[face_with_scalar(2.0)], &mut out)
			.expect("recording should succeed");

		assert_eq!(
			events,
			vec![RecordEvent::FaceRecorded {
				index: 0,
				similarity: 1.0
			}]
		);
		assert_eq!(
			String::from_utf8(out).expect("valid utf8"),
			format!("{}\n", Descriptor([2.0; redface_core::DESCRIPTOR_LEN]).encode_line())
		);
		assert_eq!(
			session.aggregate(),
			Some(&Descriptor([2.0; redface_core::DESCRIPTOR_LEN]))
		);
	}

	#[test]
	fn later_faces_use_similarity_before_averaging() {
		let mut session = RecordSession::new();
		let mut out = Vec::new();
		session
			.record_faces(&[face_with_scalar(2.0)], io::sink())
			.expect("seed face recorded");

		let events = session
			.record_faces(&[face_with_scalar(4.0)], &mut out)
			.expect("recording should succeed");

		// Same direction, different magnitude -> cosine similarity ~1.0.
		match &events[..] {
			[RecordEvent::FaceRecorded { index, similarity }] => {
				assert_eq!(*index, 0);
				assert!((similarity - 1.0).abs() < 1e-9, "similarity={similarity}");
			}
			other => panic!("unexpected events: {other:?}"),
		}
		assert_eq!(
			String::from_utf8(out).expect("valid utf8"),
			format!("{}\n", Descriptor([3.0; redface_core::DESCRIPTOR_LEN]).encode_line())
		);
		assert_eq!(
			session.aggregate(),
			Some(&Descriptor([3.0; redface_core::DESCRIPTOR_LEN]))
		);
	}

	#[test]
	fn empty_face_list_counts_no_face_frame() {
		let mut session = RecordSession::new();
		let mut out = Vec::new();

		let events = session.record_faces(&[], &mut out).expect("recording should succeed");

		assert_eq!(events, vec![RecordEvent::NoFaceDetected]);
		assert!(out.is_empty());
		assert_eq!(session.no_face_frames(), 1);
	}
}
