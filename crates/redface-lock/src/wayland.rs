use std::error::Error;
use std::io::Read;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use redface_toolkit::scene::{Scene, Uniforms};
use redface_toolkit::text::{Fonts, GlyphAtlas};
use redface_toolkit::wayland::{App, LayerConfig, Role, RunConfig};
use redface_toolkit::{Anchor, KeyEvent, KeyboardInteractivity, Keysym, Layer, PointerEventKind};

use crate::auth::{AuthEvent, FaceAuth};
use crate::config::{Color, LockConfig};
use crate::ui::{self, UiState};

const BTN_LEFT: u32 = 0x110;

/// The lock screen as a toolkit app: all Wayland/Vulkan plumbing lives in
/// redface-toolkit; this keeps the auth state machine (PAM + face) and the
/// UI state.
pub struct LockApp {
	test: bool,
	exit: bool,
	ui: UiState,
	config: LockConfig,
	/// Logical size of the primary surface, derived from the last build_scene
	/// (buffer size / scale) for pointer hit-testing.
	logical_size: (u32, u32),
	uid: u32,
	username: String,
	socket_path: String,
	face: Option<FaceAuth>,
	pam_running: Arc<AtomicBool>,
	auth_tx: Sender<AuthEvent>,
	auth_rx: Receiver<AuthEvent>,
	wake_rx: UnixStream,
	wake_tx: UnixStream,
}

impl LockApp {
	fn toggle_face(&mut self) {
		if let Some(face) = self.face.take() {
			face.stop();
			self.ui.set_face_active(false);
		} else {
			let wake = match self.wake_tx.try_clone() {
				Ok(wake) => wake,
				Err(err) => {
					self.ui.fail(format!("wake pipe failed: {err}"));
					return;
				}
			};
			match FaceAuth::start(&self.socket_path, self.uid, self.auth_tx.clone(), wake) {
				Ok(face) => {
					self.face = Some(face);
					self.ui.set_face_active(true);
					self.ui.message = None;
				}
				Err(err) => self.ui.fail(format!("face unavailable: {err}")),
			}
		}
	}

	fn submit(&mut self) {
		if self.ui.is_empty() {
			self.toggle_face();
			return;
		}
		if self.pam_running.load(Ordering::Relaxed) {
			return;
		}
		let password = self.ui.take_password();
		self.pam_running.store(true, Ordering::Relaxed);
		let running = self.pam_running.clone();
		let tx = self.auth_tx.clone();
		let username = self.username.clone();
		if let Ok(mut wake) = self.wake_tx.try_clone() {
			thread::spawn(move || {
				use std::io::Write;
				let result = crate::auth::pam_auth(&username, &password);
				running.store(false, Ordering::Relaxed);
				let _ = tx.send(AuthEvent::Pam(result));
				let _ = wake.write_all(&[1]);
			});
		}
	}

	fn handle_auth(&mut self, event: AuthEvent) {
		match event {
			AuthEvent::Pam(Ok(())) | AuthEvent::Face(Ok(())) => self.exit = true,
			AuthEvent::Pam(Err(msg)) => self.ui.fail(msg),
			AuthEvent::Face(Err(msg)) => {
				self.face = None;
				self.ui.set_face_active(false);
				self.ui.fail(msg);
			}
		}
	}
}

impl App for LockApp {
	fn build_scene(
		&mut self,
		fonts: &Fonts,
		atlas: &mut GlyphAtlas,
		width: u32,
		height: u32,
		scale: f32,
		epoch: Instant,
		primary: bool,
	) -> Scene {
		self.logical_size = (
			(width as f32 / scale).round() as u32,
			(height as f32 / scale).round() as u32,
		);
		if !primary {
			// Secondary monitors only show the background.
			return Scene::default();
		}
		ui::build_scene(&self.ui, &self.config, fonts, atlas, width, height, scale, epoch)
	}

	fn primary_output(&self) -> Option<String> {
		self.config.primary_output.clone()
	}

	/// Colors and animation timestamps; the runner fills in the sizes.
	fn uniforms(&self, epoch: Instant) -> Uniforms {
		Uniforms {
			surface_size: [0.0; 2],
			bg_image_size: [0.0; 2],
			bg_color: color_uniform(self.config.background),
			text_color: color_uniform(self.config.text_color),
			box_color: color_uniform(self.config.box_color),
			accent_color: color_uniform(self.config.accent_color),
			time: epoch.elapsed().as_secs_f32(),
			shake_start: self.ui.shake_start_secs(epoch),
			face_toggled_at: self.ui.face_toggled_at_secs(epoch),
			face_active: if self.ui.face_active { 1.0 } else { 0.0 },
		}
	}

	fn on_key(&mut self, event: &KeyEvent) {
		if event.keysym == Keysym::Escape {
			// In test mode Escape unlocks and exits; locked, it clears the password.
			if self.test {
				self.exit = true;
				return;
			}
			self.ui.clear();
		} else if event.keysym == Keysym::BackSpace {
			self.ui.backspace();
		} else if event.keysym == Keysym::Return || event.keysym == Keysym::KP_Enter {
			self.submit();
		} else if let Some(utf8) = &event.utf8 {
			for c in utf8.chars() {
				if !c.is_control() {
					self.ui.push_char(c);
				}
			}
		}
	}

	fn on_pointer(&mut self, kind: PointerEventKind, position: (f64, f64)) {
		// Pointer positions are logical surface coordinates.
		let lay = ui::layout(self.logical_size.0, self.logical_size.1, 1.0);
		match kind {
			PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
				self.ui.hover_face = ui::hit_face(&lay, position.0, position.1);
			}
			PointerEventKind::Leave { .. } => {
				self.ui.hover_face = false;
			}
			PointerEventKind::Press { button, .. }
				if button == BTN_LEFT && ui::hit_face(&lay, position.0, position.1) =>
			{
				self.toggle_face();
			}
			_ => {}
		}
	}

	fn animating(&self) -> bool {
		self.ui.animating(Instant::now())
	}

	fn should_exit(&self) -> bool {
		self.exit
	}

	fn on_exit(&mut self) {
		if let Some(face) = self.face.take() {
			face.stop();
		}
	}

	/// Drains the auth wake pipe, then handles all queued auth results.
	fn on_tick(&mut self) {
		let mut buf = [0u8; 64];
		loop {
			match self.wake_rx.read(&mut buf) {
				Ok(0) | Err(_) => break,
				Ok(_) => {}
			}
		}
		while let Ok(event) = self.auth_rx.try_recv() {
			self.handle_auth(event);
		}
	}

	fn wake_fd(&self) -> Option<RawFd> {
		Some(self.wake_rx.as_raw_fd())
	}
}

/// Config colors are 8-bit sRGB; shaders work in 0..1 floats (alpha is 1:
/// everything the lock screen draws is opaque over its background).
fn color_uniform(color: Color) -> [f32; 4] {
	[
		color.r as f32 / 255.0,
		color.g as f32 / 255.0,
		color.b as f32 / 255.0,
		1.0,
	]
}

pub fn run(config: LockConfig, background: Option<(Vec<u8>, u32, u32)>, test: bool) -> Result<(), Box<dyn Error>> {
	let (auth_tx, auth_rx) = mpsc::channel();
	let (wake_tx, wake_rx) = UnixStream::pair()?;
	wake_rx.set_nonblocking(true)?;

	let uid = unsafe { libc::geteuid() };
	let socket_path = redface_core::Config::load_default()
		.map(|config| config.socket)
		.unwrap_or_else(|_| redface_core::DEFAULT_SOCKET_PATH.to_owned());

	let mut app = LockApp {
		test,
		exit: false,
		ui: UiState::new(),
		config,
		logical_size: (0, 0),
		uid,
		username: crate::auth::current_username()?,
		socket_path,
		face: None,
		pam_running: Arc::new(AtomicBool::new(false)),
		auth_tx,
		auth_rx,
		wake_rx,
		wake_tx,
	};

	// Test mode covers all outputs with a fullscreen overlay layer surface (no
	// session lock; Esc exits) so the UI can be tried without valid credentials.
	let role = if test {
		Role::Layer(LayerConfig {
			layer: Layer::Overlay,
			anchor: Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT,
			size: (0, 0),
			exclusive_zone: -1,
			interactivity: KeyboardInteractivity::Exclusive,
			margin: (0, 0, 0, 0),
			all_outputs: true,
		})
	} else {
		Role::SessionLock
	};

	redface_toolkit::run(
		RunConfig {
			role,
			namespace: "redface-lock".to_owned(),
			background,
		},
		&mut app,
	)
}
