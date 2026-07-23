use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
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
				}
				let _ = conn.shutdown(Shutdown::Both);
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
	let req = match read_req(&mut *conn) {
		Ok(req) => req,
		Err(err) if err.is_eof() => return Ok(()),
		Err(err) => return Err(Box::new(err)),
	};

	match req.action {
		redface_runtime::Action::Authenticate => {
			let auth_req = to_auth_req(&req);
			println!("Authorizing {}", auth_req.user);

			// Watch the socket: the client closing the connection mid-verify
			// (timeout, Ctrl-C) must stop the camera stream immediately.
			let disconnected = watch_disconnect(conn)?;

			let timeout = if let Some(timeout) = auth_req.timeout {
				if timeout <= 0 {
					None
				} else {
					Some(Duration::from_secs(timeout as u64))
				}
			} else {
				Some(Duration::from_secs(config.timeout))
			};

			let model_file = Path::new(DEFAULT_MODELS_DIR).join(format!("{}.face", auth_req.user));
			let success = verify(
				recognizer,
				&VerifyOptions {
					device: PathBuf::from(&config.device),
					face_file: model_file,
					timeout,
					threshold: config.threshold,
					cancel: Some(disconnected),
				},
			);

			match success {
				Ok(true) => write_success_res(&mut *conn, std::collections::BTreeMap::new())?,
				Ok(false) => {
					let err = io::Error::other("face not recognized");
					write_error_res(&mut *conn, &err)?;
				}
				Err(redface_runtime::VerifyError::Cancelled) => println!("Client disconnected"),
				Err(err) => write_error_res(&mut *conn, &err)?,
			}
		}
	};
	Ok(())
}

/// Spawns a thread that flags the returned bool once the client is gone.
/// Bytes in flight (e.g. the trailing newline serde_json leaves after the
/// request) are drained; only EOF (peer closed) or an error flags the watch.
fn watch_disconnect(conn: &UnixStream) -> io::Result<Arc<AtomicBool>> {
	let disconnected = Arc::new(AtomicBool::new(false));
	let mut watcher = conn.try_clone()?;
	let watcher_flag = disconnected.clone();
	thread::spawn(move || {
		let mut buf = [0u8; 64];
		loop {
			match watcher.read(&mut buf) {
				Ok(0) | Err(_) => break,
				Ok(_) => {}
			}
		}
		watcher_flag.store(true, Ordering::Relaxed);
	});
	Ok(disconnected)
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


#[cfg(test)]
mod tests {
	use super::*;
	use redface_runtime::{AuthReq, write_auth_req};

	#[test]
	fn watcher_ignores_trailing_newline_but_flags_peer_close() {
		let (mut client, server) = UnixStream::pair().expect("socket pair");
		write_auth_req(
			&mut client,
			&AuthReq {
				client: "test".into(),
				user: "1000".into(),
				timeout: None,
			},
		)
		.expect("write request");
		read_req(&server).expect("read request");

		let disconnected = watch_disconnect(&server).expect("spawn watcher");
		// The newline write_auth_req appends after the JSON is still buffered
		// in the socket; draining it must not count as a disconnect.
		thread::sleep(Duration::from_millis(100));
		assert!(!disconnected.load(Ordering::Relaxed));

		drop(client);
		for _ in 0..100 {
			if disconnected.load(Ordering::Relaxed) {
				return;
			}
			thread::sleep(Duration::from_millis(10));
		}
		panic!("watcher did not flag the closed connection");
	}
}
