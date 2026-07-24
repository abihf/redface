//! redface-osd: persistent Wayland OSD (wlr-layer-shell, top layer) that
//! listens on a per-user Unix socket and shows face-recognition feedback.
//! Esc or the Cancel button sends `Cancelling` back to the daemon; the OSD
//! hides 3 seconds after `Stopped` and then waits for the next session.

mod ui;

use std::fs;
use std::io::ErrorKind;
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::ExitCode;
use std::time::Instant;

use redface_core::{OSDNotification, get_osd_socket_path, prelude::*};
use redface_toolkit::scene::{Scene, Uniforms};
use redface_toolkit::text::{Fonts, GlyphAtlas};
use redface_toolkit::{
	Anchor, App, KeyEvent, KeyboardInteractivity, Keysym, Layer, LayerConfig, PointerEventKind, Role, RunConfig, run,
};

/// Logical surface size; kept in sync with `LayerConfig::size` for
/// hit-testing (pointer positions are logical coordinates).
const SURFACE_SIZE: (u32, u32) = (380, 210);
const BTN_LEFT: u32 = 0x110;

struct OsdApp {
	/// The accepted daemon connection; non-blocking so reads in `on_tick`
	/// never stall the Wayland event loop.
	osd_conn: UnixStream,
	/// Last notification received from the daemon.
	notification: OSDNotification,
	/// When `Stopped` was received; the UI stays visible for 3 more seconds.
	stopped_at: Option<Instant>,
	/// True after the user clicked Cancel (we sent Cancelling and are waiting
	/// for the daemon to reply with Stopped).
	cancelled_by_user: bool,
	hover_cancel: bool,
	/// Face colour derived from `notification`.
	face_color: [f32; 4],
	/// Epoch of the Wayland connection (set via first `uniforms` call).
	epoch: Instant,
	/// Set by `on_tick` / `apply_notification` when the scene needs a rebuild.
	needs_redraw: bool,
}

impl OsdApp {
	fn new(osd_conn: UnixStream) -> Self {
		Self {
			osd_conn,
			notification: OSDNotification::Verifying,
			stopped_at: None,
			cancelled_by_user: false,
			hover_cancel: false,
			face_color: ui::ACCENT_COLOR, // blue = Verifying
			epoch: Instant::now(),
			needs_redraw: true,
		}
	}

	fn apply_notification(&mut self, notif: OSDNotification) {
		self.face_color = match notif {
			OSDNotification::Verifying => ui::ACCENT_COLOR,
			OSDNotification::Success => ui::SUCCESS_COLOR,
			OSDNotification::FaceMismatch => ui::MISMATCH_COLOR,
			// Keep the last active colour; the 3 s fade-out is driven by
			// stopped_at rather than the colour.
			OSDNotification::Stopped | OSDNotification::Cancelling => self.face_color,
		};
		self.notification = notif;
		self.needs_redraw = true;
		// Start the 3-second hide timer as soon as the daemon tells us
		// the session is over, not only on the subsequent EOF.
		if matches!(self.notification, OSDNotification::Stopped) && self.stopped_at.is_none() {
			log::debug!("osd: received Stopped, starting 3s hide timer");
			self.stopped_at = Some(Instant::now());
		}
	}

	fn active(&self) -> bool {
		self.stopped_at.is_none()
	}

	fn button_label(&self) -> &str {
		if self.stopped_at.is_some() { "Close" } else { "Cancel" }
	}
}

impl App for OsdApp {
	fn build_scene(
		&mut self,
		fonts: &Fonts,
		atlas: &mut GlyphAtlas,
		width: u32,
		height: u32,
		scale: f32,
		_epoch: Instant,
		_primary: bool,
	) -> Scene {
		let scene = ui::build_scene(
			self.hover_cancel,
			self.face_color,
			self.button_label(),
			fonts,
			atlas,
			width,
			height,
			scale,
		);
		self.needs_redraw = false;
		scene
	}

	fn uniforms(&self, _epoch: Instant) -> Uniforms {
		let elapsed = self.epoch.elapsed().as_secs_f32();
		let face_active = if self.active() { 1.0 } else { 0.0 };
		Uniforms {
			surface_size: [0.0, 0.0],
			bg_image_size: [0.0, 0.0],
			bg_color: [0.0, 0.0, 0.0, 0.0],
			text_color: ui::TEXT_COLOR,
			box_color: ui::BOX_COLOR,
			accent_color: self.face_color,
			time: elapsed,
			shake_start: -1.0,
			face_toggled_at: 0.0,
			face_active,
		}
	}

	fn on_key(&mut self, event: &KeyEvent) {
		if event.keysym == Keysym::Escape {
			if self.stopped_at.is_some() {
				// Already stopped: Esc dismisses immediately.
				self.dismiss();
			} else {
				self.send_cancel();
			}
		}
	}

	fn on_pointer(&mut self, kind: PointerEventKind, position: (f64, f64)) {
		let lay = ui::layout(SURFACE_SIZE.0, SURFACE_SIZE.1, 1.0);
		match kind {
			PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
				self.hover_cancel = ui::hit_cancel(&lay, position.0, position.1);
			}
			PointerEventKind::Leave { .. } => self.hover_cancel = false,
			PointerEventKind::Press { button, .. }
				if button == BTN_LEFT && ui::hit_cancel(&lay, position.0, position.1) =>
			{
				if self.stopped_at.is_some() {
					self.dismiss();
				} else {
					self.send_cancel();
				}
			}
			_ => {}
		}
	}

	fn animating(&self) -> bool {
		self.active()
	}

	fn should_exit(&self) -> bool {
		if let Some(at) = self.stopped_at {
			let elapsed = at.elapsed().as_secs();
			let should = elapsed >= 3;
			if should {
				log::debug!("osd: stopped_at elapsed {}s >= 3s, exiting", elapsed);
			}
			should
		} else {
			false
		}
	}

	/// Extra fd added to the poll set; [`App::on_tick`] runs when data
	/// arrives on the daemon connection.
	fn wake_fd(&self) -> Option<std::os::fd::RawFd> {
		Some(self.osd_conn.as_raw_fd())
	}

	fn on_tick(&mut self) {
		// Once the session is over the socket is shut down and will keep
		// waking up poll; don't try to read from it again.
		if self.stopped_at.is_some() {
			return;
		}
		loop {
			match OSDNotification::read_from(&mut self.osd_conn) {
				Ok(notif) => {
					log::debug!("osd: received notification: {:?}", notif);
					self.apply_notification(notif);
				}
				Err(ref err) if err.kind() == ErrorKind::WouldBlock => break,
				Err(err) => {
					log::debug!("osd: socket closed ({err}), treating as Stopped");
					if self.stopped_at.is_none() {
						self.stopped_at = Some(Instant::now());
						self.needs_redraw = true;
					}
					break;
				}
			}
		}
	}

	fn tick_dirty(&self) -> bool {
		self.needs_redraw
	}
}

impl OsdApp {
	fn send_cancel(&mut self) {
		if self.cancelled_by_user || !self.active() {
			return;
		}
		log::debug!("osd: user cancelled, sending Cancelling to daemon");
		self.cancelled_by_user = true;
		let _ = OSDNotification::Cancelling.write_to(&mut self.osd_conn);
	}

	/// Set stopped_at far enough in the past that `should_exit` returns
	/// true on the next check — dismisses the UI immediately.
	fn dismiss(&mut self) {
		log::debug!("osd: user dismissed");
		self.stopped_at = Some(Instant::now() - std::time::Duration::from_secs(4));
	}
}

fn main() -> ExitCode {
	env_logger::init();
	let uid = unsafe { libc::geteuid() };
	let socket_path = get_osd_socket_path(uid);
	eprintln!("redface-osd: starting, socket={}", socket_path.display());

	// Remove a stale socket from a previous run.
	let _ = fs::remove_file(&socket_path);
	if let Some(parent) = socket_path.parent() {
		let _ = fs::create_dir_all(parent);
	}
	let listener = match UnixListener::bind(&socket_path) {
		Ok(l) => l,
		Err(err) => {
			eprintln!("redface-osd: bind {}: {err}", socket_path.display());
			return ExitCode::from(2);
		}
	};
	if let Err(err) = fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600)) {
		eprintln!("redface-osd: chmod {}: {err}", socket_path.display());
	}

	loop {
		log::debug!("osd: waiting for daemon connection...");
		let conn = loop {
			match listener.accept() {
				Ok((conn, peer)) => {
					log::debug!("osd: accepted connection from {:?}", peer);
					break conn;
				}
				Err(ref err) if err.kind() == ErrorKind::Interrupted => continue,
				Err(err) => {
					eprintln!("redface-osd: accept: {err}");
					let _ = fs::remove_file(&socket_path);
					return ExitCode::from(2);
				}
			}
		};
		if let Err(err) = conn.set_nonblocking(true) {
			eprintln!("redface-osd: set_nonblocking: {err}");
			continue;
		}

		let config = RunConfig {
			role: Role::Layer(LayerConfig {
				layer: Layer::Top,
				anchor: Anchor::TOP,
				size: SURFACE_SIZE,
				exclusive_zone: 0,
				interactivity: KeyboardInteractivity::OnDemand,
				margin: (40, 0, 0, 0),
				all_outputs: false,
			}),
			namespace: "redface-osd".to_owned(),
			background: None,
		};

		log::debug!("osd: entering run() (session start)");
		let mut app = OsdApp::new(conn);
		match run(config, &mut app) {
			Ok(()) => log::debug!("osd: run() returned Ok"),
			Err(err) => log::debug!("osd: run() returned Err: {err}"),
		}
		log::debug!("osd: loop iteration done, going back to accept");
		// Loop back to accept the next session.
	}
}
