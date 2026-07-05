use std::env;
use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use redface_capture::{Camera, StreamAction};
use redface_record::{RecordEvent, RecordSession};
use redface_recognition::Recognizer;

const DEFAULT_MODEL_DIR: &str = "/usr/share/redface";
const DEFAULT_OUTPUT_FILE: &str = "capture.face";

fn main() -> ExitCode {
    match main_impl() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn main_impl() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let recognizer = Recognizer::new(&args.model_dir)?;
    let camera = Camera::new(&args.device);
    let mut output = File::create(&args.output_file)?;
    let mut session = RecordSession::new();

    let stats = camera.stream(|frame| {
        let faces = match recognizer.recognize(&frame.buffer, frame.width, frame.height, 1) {
            Ok(faces) => faces,
            Err(err) => {
                eprintln!("{err}");
                return StreamAction::Stop;
            }
        };

        let events = match session.record_faces(&faces, &mut output) {
            Ok(events) => events,
            Err(err) => {
                eprintln!("{err}");
                return StreamAction::Stop;
            }
        };

        for event in events {
            match event {
                RecordEvent::NoFaceDetected => println!("\t- No face detected"),
                RecordEvent::FaceRecorded { index, distance } => {
                    println!("  - Face [{index}] (distance: {distance:.3})");
                }
            }
        }

        StreamAction::Continue
    })?;

    if stats.dropped_frames > 0 {
        println!("Dropped {} frames", stats.dropped_frames);
    }
    println!("Frames without face found: {}", session.no_face_frames());

    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    device: PathBuf,
    model_dir: PathBuf,
    output_file: PathBuf,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, io::Error> {
        let mut device = None;
        let mut model_dir = PathBuf::from(DEFAULT_MODEL_DIR);
        let mut output_file = PathBuf::from(DEFAULT_OUTPUT_FILE);

        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--device" => device = Some(PathBuf::from(next_value(&mut iter, "--device")?)),
                "--model-dir" => model_dir = PathBuf::from(next_value(&mut iter, "--model-dir")?),
                "--output" => output_file = PathBuf::from(next_value(&mut iter, "--output")?),
                "-h" | "--help" => {
                    print_usage();
                    return Err(io::Error::other("help requested"));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unknown argument: {arg}"),
                    ));
                }
            }
        }

        let device = device.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "missing required --device argument")
        })?;

        Ok(Self {
            device,
            model_dir,
            output_file,
        })
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, io::Error> {
    iter.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing value for {flag}"),
        )
    })
}

fn print_usage() {
    println!(
        "Usage: redface-record --device <path> [--model-dir <dir>] [--output <file>]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_and_optional_arguments() {
        let args = Args::parse(vec![
            "--device".into(),
            "/dev/video0".into(),
            "--model-dir".into(),
            "/models".into(),
            "--output".into(),
            "faces.out".into(),
        ])
        .expect("args should parse");

        assert_eq!(
            args,
            Args {
                device: PathBuf::from("/dev/video0"),
                model_dir: PathBuf::from("/models"),
                output_file: PathBuf::from("faces.out"),
            }
        );
    }

    #[test]
    fn requires_device_argument() {
        let err = Args::parse(Vec::new()).expect_err("missing device should fail");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("missing required --device"));
    }
}