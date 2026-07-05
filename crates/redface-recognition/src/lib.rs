use std::fmt;
use std::path::{Path, PathBuf};

use dlib_face_recognition::{
    FaceDetector, FaceDetectorTrait, FaceEncoderNetwork, FaceEncoderTrait, ImageMatrix,
    LandmarkPredictor, LandmarkPredictorTrait,
};
pub use dlib_face_recognition::Rectangle;
pub use redface_core::{DESCRIPTOR_LEN, Descriptor};

const LANDMARK_MODEL: &str = "shape_predictor_5_face_landmarks.dat";
const ENCODER_MODEL: &str = "dlib_face_recognition_resnet_model_v1.dat";

#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    pub rectangle: Rectangle,
    pub descriptor: Descriptor,
}

#[derive(Debug)]
pub enum RecognizerError {
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
}

impl fmt::Display for RecognizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
        }
    }
}

impl std::error::Error for RecognizerError {}

pub struct Recognizer {
    detector: FaceDetector,
    landmark_predictor: LandmarkPredictor,
    face_encoder: FaceEncoderNetwork,
}

impl Recognizer {
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self, RecognizerError> {
        let model_dir = model_dir.as_ref();
        let landmark_path = model_dir.join(LANDMARK_MODEL);
        let encoder_path = model_dir.join(ENCODER_MODEL);

        let landmark_predictor = LandmarkPredictor::open(&landmark_path).map_err(|message| {
            RecognizerError::ModelLoad {
                path: landmark_path.clone(),
                message,
            }
        })?;

        let face_encoder =
            FaceEncoderNetwork::open(&encoder_path).map_err(|message| RecognizerError::ModelLoad {
                path: encoder_path.clone(),
                message,
            })?;

        Ok(Self {
            detector: FaceDetector::new(),
            landmark_predictor,
            face_encoder,
        })
    }

    pub fn recognize(
        &self,
        img_data: &[u8],
        width: u32,
        height: u32,
        max_faces: usize,
    ) -> Result<Vec<Face>, RecognizerError> {
        validate_rgb_buffer(width, height, img_data.len())?;

        let image = unsafe { ImageMatrix::new(width as usize, height as usize, img_data.as_ptr()) };
        let mut rectangles = self.detector.face_locations(&image).to_vec();

        if rectangles.is_empty() {
            return Ok(Vec::new());
        }
        if max_faces > 0 && rectangles.len() > max_faces {
            return Ok(Vec::new());
        }

        let landmarks = rectangles
            .iter()
            .map(|rectangle| self.landmark_predictor.face_landmarks(&image, rectangle))
            .collect::<Vec<_>>();
        let encodings = self.face_encoder.get_face_encodings(&image, &landmarks, 0);

        let faces = rectangles
            .into_iter()
            .zip(encodings.iter())
            .map(|(rectangle, encoding)| Face {
                rectangle,
                descriptor: descriptor_from_encoding(encoding.as_ref()),
            })
            .collect();

        Ok(faces)
    }
}

fn descriptor_from_encoding(values: &[f64]) -> Descriptor {
    let mut descriptor = [0.0_f32; DESCRIPTOR_LEN];
    for (slot, value) in descriptor.iter_mut().zip(values.iter().copied()) {
        *slot = value as f32;
    }
    Descriptor(descriptor)
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
    fn converts_face_encoding_to_descriptor() {
        let input = (0..DESCRIPTOR_LEN).map(|value| value as f64 / 10.0).collect::<Vec<_>>();

        let descriptor = descriptor_from_encoding(&input);

        assert_eq!(descriptor.0[0], 0.0);
        assert_eq!(descriptor.0[10], 1.0);
        assert_eq!(descriptor.0[127], 12.7);
    }
}