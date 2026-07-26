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

use redface_core::{
	Config, DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, DaemonRequest, DaemonResponse, DevicePref, OSDNotification,
	get_osd_socket_path, prelude::*,
};
use redface_recognition::Recognizer;
use redface_runtime::{VerifyOptions, verify};
use signal_hook::consts::signal::{SIGINT, SIGTERM};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	env_logger::init();
	let mut app = App::new()?;
	app.run()?;
	Ok(())
}

struct App {
	recognizer: Recognizer,
	config: Config,
	_pid_guard: PidFileGuard,
}

impl App {
	fn new() -> Result<Self, Box<dyn std::error::Error>> {
		let config = Config::load_default()?;
		if is_already_running(&config.pid_file) {
			return Err("already run".into());
		}
		let recognizer = Recognizer::new(DEFAULT_DATA_DIR, DevicePref::parse(&config.inference_device)?)?;
		let pid_guard = PidFileGuard::create(&config.pid_file)?;
		Ok(Self {
			recognizer,
			config,
			_pid_guard: pid_guard,
		})
	}

	fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
		let socket_path = PathBuf::from(&self.config.socket);
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
					if let Err(err) = self.handle_connection(&mut conn) {
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

	fn handle_connection(&mut self, conn: &mut UnixStream) -> Result<(), Box<dyn std::error::Error>> {
		let req = match DaemonRequest::read_from(&mut *conn) {
			Ok(req) => req,
			Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
			Err(err) => return Err(Box::new(err)),
		};

		match req {
			DaemonRequest::Authenticate {
				client: _client,
				user,
				timeout,
				show_osd,
			} => {
				self.handle_authentication(conn, user, timeout, show_osd)?;
			}
			_ => {} // Ignore other requests for now.
		}
		Ok(())
	}

	fn handle_authentication(
		&mut self,
		conn: &mut UnixStream,
		user: String,
		timeout: Option<i32>,
		show_osd: bool,
	) -> Result<(), Box<dyn std::error::Error>> {
		println!("Authorizing {user}");

		// Watch the socket: the client closing the connection mid-verify
		// (timeout, Ctrl-C) must stop the camera stream immediately.
		let disconnected = watch_disconnect(conn)?;

		let timeout = if let Some(timeout) = timeout {
			if timeout <= 0 {
				None
			} else {
				Some(Duration::from_secs(timeout as u64))
			}
		} else {
			Some(Duration::from_secs(self.config.timeout))
		};

		// Connect to the per-user OSD socket when the client requested
		// visual feedback (redface-check sets show_osd = true).
		log::debug!("daemon: show_osd={show_osd}");
		let (mut osd_conn, osd_cancelled) = if show_osd { connect_osd(&user) } else { None }.unzip();
		log::debug!(
			"daemon: osd_conn={}",
			if osd_conn.is_some() { "connected" } else { "none" }
		);

		let face_file = Path::new(DEFAULT_MODELS_DIR).join(format!("{user}.face"));
		let mut osd_notify = osd_conn.as_mut().map(|c| c.try_clone()).transpose()?;
		let threshold = self.config.threshold;
		let success = verify(
			&mut self.recognizer,
			&VerifyOptions {
				device: PathBuf::from(&self.config.device),
				face_file,
				timeout,
				threshold,
				cancel: Some(disconnected),
				osd_cancel: osd_cancelled,
			},
			|event| {
				if let Some(ref mut stream) = osd_notify {
					let _ = event.write_to(&mut *stream);
				}
			},
		);

		// Send Stopped so the OSD knows the session is over.
		if let Some(ref mut stream) = osd_conn {
			log::debug!("daemon: sending Stopped to OSD");
			let _ = OSDNotification::Stopped.write_to(&mut *stream);
			let _ = stream.shutdown(Shutdown::Both);
			log::debug!("daemon: OSD connection shut down");
		}

		match success {
			Ok(true) => {
				DaemonResponse::AuthSuccess.write_to(&mut *conn)?;
			}
			Ok(false) => {
				DaemonResponse::AuthError("face not recognized".to_owned()).write_to(&mut *conn)?;
			}
			Err(redface_runtime::VerifyError::Cancelled) => println!("Client disconnected"),
			Err(err) => {
				DaemonResponse::AuthError(err.to_string()).write_to(&mut *conn)?;
			}
		}
		Ok(())
	}
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

/// Tries to connect to the per-user OSD socket. Returns the connection and a
/// cancel flag that the OSD can trip by sending `Cancelling` (or closing the
/// connection). If the socket doesn't exist or `user` isn't a valid uid the
/// function returns `None` — verification proceeds without visual feedback.
fn connect_osd(user: &str) -> Option<(UnixStream, Arc<AtomicBool>)> {
	let uid: u32 = user.parse().ok()?;
	let path = get_osd_socket_path(uid);
	log::debug!("daemon: osd socket path: {}", path.display());
	if !path.exists() {
		log::debug!("daemon: osd socket does not exist, skipping");
		return None;
	}
	let mut conn = UnixStream::connect(&path).ok()?;
	log::debug!("daemon: connected to osd socket");
	// Let the OSD know we're starting.
	let _ = OSDNotification::Verifying.write_to(&mut conn);
	log::debug!("daemon: sent Verifying to osd");

	let cancelled = Arc::new(AtomicBool::new(false));
	let mut watcher = conn.try_clone().ok()?;
	let watcher_flag = cancelled.clone();
	thread::spawn(move || {
		log::debug!("daemon: osd cancel watcher started");
		loop {
			match OSDNotification::read_from(&mut watcher) {
				Ok(OSDNotification::Cancelling) => {
					log::debug!("daemon: osd cancel watcher: received Cancelling");
					watcher_flag.store(true, Ordering::Relaxed);
					break;
				}
				Ok(other) => log::debug!("daemon: osd cancel watcher: ignoring {:?}", other),
				Err(err) => {
					log::debug!("daemon: osd cancel watcher: error/EOF ({err}), flagging cancel");
					watcher_flag.store(true, Ordering::Relaxed);
					break;
				}
			}
		}
		log::debug!("daemon: osd cancel watcher exiting");
	});
	Some((conn, cancelled))
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
