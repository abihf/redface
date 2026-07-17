use std::fs::{self, File};
use std::io::{self, ErrorKind, Write};
use std::net::Shutdown;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{
	Arc,
	atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use redface_recognition::{DevicePref, Recognizer};
use redface_runtime::{
	Config, DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, VerifyOptions, read_req, to_auth_req, verify, write_error_res,
	write_success_res,
};
use signal_hook::consts::signal::{SIGINT, SIGTERM};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let config = Config::load_default()?;
	if is_already_running(&config.pid_file) {
		return Err("already run".into());
	}

	let mut recognizer = Recognizer::new(DEFAULT_DATA_DIR, DevicePref::parse(&config.inference_device)?)?;
	let _pid_guard = PidFileGuard::create(&config.pid_file)?;

	let socket_path = PathBuf::from(&config.socket);
	let _ = fs::remove_file(&socket_path);
	let listener = UnixListener::bind(&socket_path)?;
	fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o666))?;
	listener.set_nonblocking(true)?;

	let stop = Arc::new(AtomicBool::new(false));
	signal_hook::flag::register(SIGINT, stop.clone())?;
	signal_hook::flag::register(SIGTERM, stop.clone())?;
	let _ = sd_notify::notify(&[sd_notify::NotifyState::Ready]);

	while !stop.load(Ordering::Relaxed) {
		match listener.accept() {
			Ok((mut conn, _)) => {
				if let Err(err) = handle_connection(&mut recognizer, &config, &mut conn) {
					eprintln!("Connection error: {err}");
					let _ = conn.shutdown(Shutdown::Both);
				}
			}
			Err(err) if err.kind() == ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(100)),
			Err(err) => return Err(Box::new(err)),
		}
	}

	let _ = fs::remove_file(&socket_path);
	Ok(())
}

fn handle_connection(
	recognizer: &mut Recognizer,
	config: &Config,
	conn: &mut UnixStream,
) -> Result<(), Box<dyn std::error::Error>> {
	loop {
		let req = match read_req(&mut *conn) {
			Ok(req) => req,
			Err(err) if err.is_eof() => return Ok(()),
			Err(err) => return Err(Box::new(err)),
		};

		match req.action {
			redface_runtime::Action::Authenticate => {
				let auth_req = to_auth_req(&req);
				println!("Authorizing {}", auth_req.user);

				let model_file = Path::new(DEFAULT_MODELS_DIR).join(format!("{}.face", auth_req.user));
				let success = verify(
					recognizer,
					&VerifyOptions {
						device: PathBuf::from(&config.device),
						face_file: model_file,
						timeout: Duration::from_secs(config.timeout),
						threshold: config.threshold,
					},
				);

				match success {
					Ok(true) => write_success_res(&mut *conn, std::collections::BTreeMap::new())?,
					Ok(false) => {
						let err = io::Error::other("face not recognized");
						write_error_res(&mut *conn, &err)?;
					}
					Err(err) => write_error_res(&mut *conn, &err)?,
				}
			}
		}
	}
}

fn is_already_running(path: &str) -> bool {
	let pid = match fs::read_to_string(path) {
		Ok(contents) => match contents.trim().parse::<i32>() {
			Ok(pid) => pid,
			Err(_) => return false,
		},
		Err(_) => return false,
	};

	unsafe { libc::kill(pid, 0) == 0 }
}

struct PidFileGuard {
	path: PathBuf,
}

impl PidFileGuard {
	fn create(path: &str) -> io::Result<Self> {
		let mut file = File::create(path)?;
		writeln!(file, "{}", std::process::id())?;
		Ok(Self {
			path: PathBuf::from(path),
		})
	}
}

impl Drop for PidFileGuard {
	fn drop(&mut self) {
		let _ = fs::remove_file(&self.path);
	}
}
