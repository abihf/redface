use std::error::Error;
use std::io::Read;
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState, FrameCallbackData};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::reexports::protocols::wp::viewporter::client::wp_viewport::WpViewport;
use smithay_client_toolkit::reexports::protocols::wp::viewporter::client::wp_viewporter::WpViewporter;
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::seat::keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers, RawModifiers};
use smithay_client_toolkit::seat::pointer::{PointerEvent, PointerEventKind, PointerHandler};
use smithay_client_toolkit::seat::{Capability, SeatHandler, SeatState};
use smithay_client_toolkit::session_lock::{
	SessionLock, SessionLockHandler, SessionLockState, SessionLockSurface, SessionLockSurfaceConfigure,
};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::wlr_layer::{
	Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface, LayerSurfaceConfigure,
};
use smithay_client_toolkit::{delegate_registry, registry_handlers};
use wayland_client::globals::registry_queue_init;
use wayland_client::protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_surface};
use wayland_client::{Connection, QueueHandle};

use crate::auth::{AuthEvent, FaceAuth};
use crate::config::{Color, LockConfig};
use crate::gpu::{Gpu, GpuError, GpuSurface};
use crate::scene::{Scene, Uniforms};
use crate::ui::{self, Fonts, GlyphAtlas, UiState};

const BTN_LEFT: u32 = 0x110;

pub struct App {
	registry_state: RegistryState,
	seat_state: SeatState,
	output_state: OutputState,
	compositor: CompositorState,
	session_lock_state: Option<SessionLockState>,
	session_lock: Option<SessionLock>,
	/// True between a successful lock request and `finished()`/unlock; gates
	/// lock-surface creation on hotplugged outputs.
	session_live: bool,
	layer_shell: Option<LayerShell>,

	test: bool,
	exit: bool,
	lock_denied: bool,
	gpu: Gpu,
	viewporter: Option<WpViewporter>,
	/// Fatal surface-creation failure (Vulkan is a hard requirement); surfaced
	/// as the process error after the session has been unlocked cleanly.
	gpu_error: Option<GpuError>,
	atlas: GlyphAtlas,
	/// Start of the process; the reference for all animation time uniforms.
	epoch: Instant,
	surfaces: Vec<SurfaceEntry>,
	ui: UiState,
	config: LockConfig,
	fonts: Fonts,
	keyboard: Option<wl_keyboard::WlKeyboard>,
	pointer: Option<wl_pointer::WlPointer>,
	uid: u32,
	username: String,
	socket_path: String,
	face: Option<FaceAuth>,
	pam_running: Arc<AtomicBool>,
	auth_tx: Sender<AuthEvent>,
	auth_rx: Receiver<AuthEvent>,
	wake_rx: UnixStream,
	wake_tx: UnixStream,
	last_minute: i32,
}

struct SurfaceEntry {
	kind: SurfaceKind,
	viewport: Option<WpViewport>,
	output: Option<wl_output::WlOutput>,
	/// Swapchain surface, created lazily at the first configure.
	gpu: Option<GpuSurface>,
	/// Last built scene; rebuilt only when `scene_dirty` (state/size change),
	/// not on pure animation frames.
	scene: Option<Scene>,
	scene_dirty: bool,
	width: u32,
	height: u32,
	scale: i32,
	primary: bool,
	dirty: bool,
	frame_pending: bool,
}

enum SurfaceKind {
	Lock(SessionLockSurface),
	Test(LayerSurface),
}

impl SurfaceKind {
	fn wl_surface(&self) -> &wl_surface::WlSurface {
		match self {
			Self::Lock(surface) => surface.wl_surface(),
			Self::Test(surface) => surface.wl_surface(),
		}
	}

	fn commit(&self) {
		match self {
			Self::Lock(surface) => surface.wl_surface().commit(),
			Self::Test(surface) => surface.commit(),
		}
	}
}

impl App {
	fn surface_index(&self, surface: &wl_surface::WlSurface) -> Option<usize> {
		self.surfaces
			.iter()
			.position(|entry| entry.kind.wl_surface() == surface)
	}

	fn is_primary(&self, output_name: Option<&str>) -> bool {
		match &self.config.primary_output {
			Some(want) => output_name == Some(want.as_str()),
			None => !self.surfaces.iter().any(|entry| entry.primary),
		}
	}

	fn add_lock_surface(&mut self, qh: &QueueHandle<Self>, output: &wl_output::WlOutput) {
		let Some(lock) = self.session_lock.clone() else {
			return;
		};
		let surface = self.compositor.create_surface(qh);
		let viewport = self.viewporter.as_ref().map(|vp| vp.get_viewport(&surface, qh, ()));
		let lock_surface = lock.create_lock_surface(surface, output, qh);
		let name = self.output_state.info(output).and_then(|info| info.name);
		let primary = self.is_primary(name.as_deref());
		// No initial commit: ext-session-lock sends the first configure on
		// bind, and committing before acking it (or with a null buffer) is a
		// protocol error.
		self.surfaces.push(SurfaceEntry {
			kind: SurfaceKind::Lock(lock_surface),
			viewport,
			output: Some(output.clone()),
			gpu: None,
			scene: None,
			scene_dirty: true,
			width: 0,
			height: 0,
			scale: 1,
			primary,
			dirty: false,
			frame_pending: false,
		});
	}

	/// Test mode: a fullscreen overlay layer surface on the given output.
	/// Covers the screen like the real lock, but the session stays unlocked.
	fn add_test_surface(&mut self, qh: &QueueHandle<Self>, output: &wl_output::WlOutput) {
		let Some(layer_shell) = &self.layer_shell else {
			return;
		};
		let surface = self.compositor.create_surface(qh);
		let viewport = self.viewporter.as_ref().map(|vp| vp.get_viewport(&surface, qh, ()));
		let layer = layer_shell.create_layer_surface(qh, surface, Layer::Overlay, Some("redface-lock"), Some(output));
		layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
		layer.set_keyboard_interactivity(KeyboardInteractivity::Exclusive);
		layer.set_exclusive_zone(-1);
		let name = self.output_state.info(output).and_then(|info| info.name);
		let primary = self.is_primary(name.as_deref());
		// Initial empty commit so the compositor sends a configure.
		layer.commit();
		self.surfaces.push(SurfaceEntry {
			kind: SurfaceKind::Test(layer),
			viewport,
			output: Some(output.clone()),
			gpu: None,
			scene: None,
			scene_dirty: true,
			width: 0,
			height: 0,
			scale: 1,
			primary,
			dirty: false,
			frame_pending: false,
		});
	}

	/// Marks the primary surface dirty and repaints immediately unless a frame
	/// callback is already pending (in which case the frame handler repaints).
	/// State changed, so the scene is rebuilt rather than just redrawn.
	fn request_redraw(&mut self, conn: &Connection, qh: &QueueHandle<Self>) {
		if self.exit {
			return;
		}
		for index in 0..self.surfaces.len() {
			if !self.surfaces[index].primary {
				continue;
			}
			self.surfaces[index].dirty = true;
			self.surfaces[index].scene_dirty = true;
			if !self.surfaces[index].frame_pending {
				self.draw(conn, qh, index);
			}
		}
	}

	fn draw(&mut self, conn: &Connection, qh: &QueueHandle<Self>, index: usize) {
		if self.exit {
			return;
		}
		let (width, height, scale) = {
			let entry = &self.surfaces[index];
			let scale = entry.scale.max(1) as u32;
			(entry.width * scale, entry.height * scale, scale)
		};
		if width == 0 || height == 0 {
			return;
		}

		// The swapchain surface is created on the first configure with a real
		// size. Vulkan is a hard requirement: failure unlocks and exits.
		if self.surfaces[index].gpu.is_none() {
			match self
				.gpu
				.create_surface(conn, self.surfaces[index].kind.wl_surface(), width, height)
			{
				Ok(surface) => self.surfaces[index].gpu = Some(surface),
				Err(err) => {
					self.gpu_error = Some(err);
					self.unlock();
					return;
				}
			}
		}
		self.surfaces[index]
			.gpu
			.as_mut()
			.expect("gpu surface created above")
			.resize(&self.gpu, width, height);

		// Rebuild the scene only when state, size, or the clock changed; pure
		// animation frames just redraw with a fresh `time` uniform.
		if self.surfaces[index].scene_dirty || self.surfaces[index].scene.is_none() {
			let scene = if self.surfaces[index].primary {
				ui::build_scene(
					&self.ui,
					&self.config,
					&self.fonts,
					&mut self.atlas,
					width,
					height,
					scale as f32,
					self.epoch,
				)
			} else {
				// Secondary outputs show only the background.
				Scene::default()
			};
			let pending = self.atlas.drain_pending();
			if !pending.is_empty() {
				self.gpu.upload_glyphs(&pending);
			}
			let entry = &mut self.surfaces[index];
			entry.scene = Some(scene);
			entry.scene_dirty = false;
		}

		let uniforms = Uniforms {
			surface_size: [width as f32, height as f32],
			bg_image_size: self.gpu.background_size(),
			bg_color: color_uniform(self.config.background),
			text_color: color_uniform(self.config.text_color),
			box_color: color_uniform(self.config.box_color),
			accent_color: color_uniform(self.config.accent_color),
			time: self.epoch.elapsed().as_secs_f32(),
			shake_start: self.ui.shake_start_secs(self.epoch),
			face_toggled_at: self.ui.face_toggled_at_secs(self.epoch),
			face_active: if self.ui.face_active { 1.0 } else { 0.0 },
		};

		let entry = &mut self.surfaces[index];
		let surface = entry.gpu.as_mut().expect("gpu surface created above");
		surface.render(&self.gpu, entry.scene.as_ref().expect("scene built above"), &uniforms);

		let wl_surface = entry.kind.wl_surface().clone();
		wl_surface.frame(qh, FrameCallbackData(wl_surface.clone()));
		entry.frame_pending = true;
		entry.dirty = false;
		entry.kind.commit();
	}

	fn toggle_face(&mut self, conn: &Connection, qh: &QueueHandle<Self>) {
		if let Some(face) = self.face.take() {
			face.stop();
			self.ui.set_face_active(false);
		} else {
			let wake = match self.wake_tx.try_clone() {
				Ok(wake) => wake,
				Err(err) => {
					self.ui.fail(format!("wake pipe failed: {err}"));
					self.request_redraw(conn, qh);
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
		self.request_redraw(conn, qh);
	}

	fn submit(&mut self, conn: &Connection, qh: &QueueHandle<Self>) {
		if self.ui.is_empty() {
			self.toggle_face(conn, qh);
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
		self.request_redraw(conn, qh);
	}

	fn handle_key(&mut self, conn: &Connection, qh: &QueueHandle<Self>, event: &KeyEvent) {
		if event.keysym == Keysym::Escape {
			// In test mode Escape unlocks and exits; locked, it clears the password.
			if self.test {
				self.unlock();
				return;
			}
			self.ui.clear();
		} else if event.keysym == Keysym::BackSpace {
			self.ui.backspace();
		} else if event.keysym == Keysym::Return || event.keysym == Keysym::KP_Enter {
			self.submit(conn, qh);
			return;
		} else if let Some(utf8) = &event.utf8 {
			for c in utf8.chars() {
				if !c.is_control() {
					self.ui.push_char(c);
				}
			}
		}
		self.request_redraw(conn, qh);
	}

	fn handle_auth(&mut self, conn: &Connection, qh: &QueueHandle<Self>, event: AuthEvent) {
		match event {
			AuthEvent::Pam(Ok(())) | AuthEvent::Face(Ok(())) => self.unlock(),
			AuthEvent::Pam(Err(msg)) => self.ui.fail(msg),
			AuthEvent::Face(Err(msg)) => {
				self.face = None;
				self.ui.set_face_active(false);
				self.ui.fail(msg);
			}
		}
		self.request_redraw(conn, qh);
	}

	fn unlock(&mut self) {
		self.session_live = false;
		if let Some(face) = self.face.take() {
			face.stop();
		}
		if let Some(lock) = &self.session_lock {
			lock.unlock();
		}
		self.exit = true;
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

impl CompositorHandler for App {
	fn scale_factor_changed(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		surface: &wl_surface::WlSurface,
		factor: i32,
	) {
		let Some(index) = self.surface_index(surface) else {
			return;
		};
		let entry = &mut self.surfaces[index];
		// A scale>1 swapchain only displays correctly when the viewport
		// protocol downscales it back to the logical size.
		let factor = if entry.viewport.is_some() { factor } else { 1 };
		if entry.scale != factor {
			entry.scale = factor;
			entry.scene_dirty = true;
			self.draw(conn, qh, index);
		}
	}

	fn transform_changed(
		&mut self,
		_: &Connection,
		_: &QueueHandle<Self>,
		_: &wl_surface::WlSurface,
		_: wl_output::Transform,
	) {
	}

	fn frame(&mut self, conn: &Connection, qh: &QueueHandle<Self>, surface: &wl_surface::WlSurface, _: u32) {
		let Some(index) = self.surface_index(surface) else {
			return;
		};
		self.surfaces[index].frame_pending = false;
		let redraw = self.surfaces[index].dirty || (self.surfaces[index].primary && self.ui.animating(Instant::now()));
		if redraw {
			self.draw(conn, qh, index);
		}
	}

	fn surface_enter(
		&mut self,
		_: &Connection,
		_: &QueueHandle<Self>,
		_: &wl_surface::WlSurface,
		_: &wl_output::WlOutput,
	) {
	}

	fn surface_leave(
		&mut self,
		_: &Connection,
		_: &QueueHandle<Self>,
		_: &wl_surface::WlSurface,
		_: &wl_output::WlOutput,
	) {
	}
}

impl OutputHandler for App {
	fn output_state(&mut self) -> &mut OutputState {
		&mut self.output_state
	}

	fn new_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
		// A monitor plugged in while running still needs a surface.
		if self.test {
			self.add_test_surface(qh, &output);
		} else if self.session_live {
			self.add_lock_surface(qh, &output);
		}
	}

	fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}

	fn output_destroyed(&mut self, conn: &Connection, qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
		if let Some(index) = self
			.surfaces
			.iter()
			.position(|entry| entry.output.as_ref() == Some(&output))
		{
			let was_primary = self.surfaces[index].primary;
			self.surfaces.remove(index);
			if was_primary && let Some(first) = self.surfaces.first_mut() {
				first.primary = true;
			}
			if was_primary {
				self.request_redraw(conn, qh);
			}
		}
	}
}

impl SessionLockHandler for App {
	fn locked(&mut self, _: &Connection, _: &QueueHandle<Self>, _: SessionLock) {
		// The configured primary_output may not be connected; never leave the
		// lock screen without a UI surface.
		if !self.surfaces.iter().any(|entry| entry.primary)
			&& let Some(first) = self.surfaces.first_mut()
		{
			first.primary = true;
		}
	}

	fn finished(&mut self, _: &Connection, _: &QueueHandle<Self>, _: SessionLock) {
		self.session_live = false;
		self.lock_denied = true;
		self.exit = true;
	}

	fn configure(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		surface: SessionLockSurface,
		configure: SessionLockSurfaceConfigure,
		_: u32,
	) {
		let Some(index) = self.surface_index(surface.wl_surface()) else {
			return;
		};
		let (width, height) = configure.new_size;
		let entry = &mut self.surfaces[index];
		if entry.width != width || entry.height != height {
			entry.scene_dirty = true;
		}
		entry.width = width;
		entry.height = height;
		if let Some(viewport) = &entry.viewport {
			viewport.set_destination(width as i32, height as i32);
		}
		self.draw(conn, qh, index);
	}
}

impl LayerShellHandler for App {
	fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
		self.exit = true;
	}

	fn configure(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		layer: &LayerSurface,
		configure: LayerSurfaceConfigure,
		_: u32,
	) {
		let Some(index) = self.surface_index(layer.wl_surface()) else {
			return;
		};
		// Anchored to all edges, so the compositor suggests the output size.
		let (width, height) = configure.new_size;
		if width == 0 || height == 0 {
			return;
		}
		let entry = &mut self.surfaces[index];
		if entry.width != width || entry.height != height {
			entry.scene_dirty = true;
		}
		entry.width = width;
		entry.height = height;
		if let Some(viewport) = &entry.viewport {
			viewport.set_destination(width as i32, height as i32);
		}
		self.draw(conn, qh, index);
	}
}

impl SeatHandler for App {
	fn seat_state(&mut self) -> &mut SeatState {
		&mut self.seat_state
	}

	fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

	fn new_capability(
		&mut self,
		_: &Connection,
		qh: &QueueHandle<Self>,
		seat: wl_seat::WlSeat,
		capability: Capability,
	) {
		if capability == Capability::Keyboard && self.keyboard.is_none() {
			self.keyboard = self.seat_state.get_keyboard(qh, &seat, None).ok();
		}
		if capability == Capability::Pointer && self.pointer.is_none() {
			self.pointer = self.seat_state.get_pointer(qh, &seat).ok();
		}
	}

	fn remove_capability(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat, capability: Capability) {
		if capability == Capability::Keyboard
			&& let Some(keyboard) = self.keyboard.take()
		{
			keyboard.release();
		}
		if capability == Capability::Pointer
			&& let Some(pointer) = self.pointer.take()
		{
			pointer.release();
		}
	}

	fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for App {
	fn enter(
		&mut self,
		_: &Connection,
		_: &QueueHandle<Self>,
		_: &wl_keyboard::WlKeyboard,
		_: &wl_surface::WlSurface,
		_: u32,
		_: &[u32],
		_: &[Keysym],
	) {
	}

	fn leave(
		&mut self,
		_: &Connection,
		_: &QueueHandle<Self>,
		_: &wl_keyboard::WlKeyboard,
		_: &wl_surface::WlSurface,
		_: u32,
	) {
	}

	fn press_key(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		_: &wl_keyboard::WlKeyboard,
		_: u32,
		event: KeyEvent,
	) {
		self.handle_key(conn, qh, &event);
	}

	fn repeat_key(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		_: &wl_keyboard::WlKeyboard,
		_: u32,
		event: KeyEvent,
	) {
		// Only editing keys repeat; Enter and Escape act on press only.
		if event.keysym == Keysym::BackSpace || event.utf8.is_some() {
			self.handle_key(conn, qh, &event);
		}
	}

	fn release_key(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_keyboard::WlKeyboard, _: u32, _: KeyEvent) {
	}

	fn update_modifiers(
		&mut self,
		_: &Connection,
		_: &QueueHandle<Self>,
		_: &wl_keyboard::WlKeyboard,
		_: u32,
		_: Modifiers,
		_: RawModifiers,
		_: u32,
	) {
	}
}

impl PointerHandler for App {
	fn pointer_frame(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		_: &wl_pointer::WlPointer,
		events: &[PointerEvent],
	) {
		for event in events {
			let Some(index) = self.surface_index(&event.surface) else {
				continue;
			};
			if !self.surfaces[index].primary {
				continue;
			}
			let lay = ui::layout(self.surfaces[index].width, self.surfaces[index].height, 1.0);
			match event.kind {
				PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
					let hover = ui::hit_face(&lay, event.position.0, event.position.1);
					if hover != self.ui.hover_face {
						self.ui.hover_face = hover;
						self.request_redraw(conn, qh);
					}
				}
				PointerEventKind::Leave { .. } => {
					if self.ui.hover_face {
						self.ui.hover_face = false;
						self.request_redraw(conn, qh);
					}
				}
				PointerEventKind::Press { button, .. }
					if button == BTN_LEFT && ui::hit_face(&lay, event.position.0, event.position.1) =>
				{
					self.toggle_face(conn, qh);
				}
				_ => {}
			}
		}
	}
}

delegate_registry!(App);

impl ProvidesRegistryState for App {
	fn registry(&mut self) -> &mut RegistryState {
		&mut self.registry_state
	}
	registry_handlers![OutputState, SeatState];
}

smithay_client_toolkit::delegate_dispatch2!(App);
wayland_client::delegate_noop!(App: ignore WpViewporter);
wayland_client::delegate_noop!(App: ignore WpViewport);

pub fn run(
	config: LockConfig,
	background: Option<(Vec<u8>, u32, u32)>,
	fonts: Fonts,
	test: bool,
) -> Result<(), Box<dyn Error>> {
	let conn = Connection::connect_to_env()?;
	let (globals, mut event_queue) = registry_queue_init(&conn)?;
	let qh = event_queue.handle();

	// Vulkan is a hard requirement: no adapter/device, no lock screen.
	let mut gpu = Gpu::new()?;
	gpu.set_background(background.as_ref().map(|(data, w, h)| (data.as_slice(), *w, *h)));

	let compositor = CompositorState::bind(&globals, &qh)?;
	let viewporter: Option<WpViewporter> = globals.bind(&qh, 1..=1, ()).ok();
	let session_lock_state = if test {
		None
	} else {
		Some(SessionLockState::new(&globals, &qh))
	};
	let layer_shell = if test {
		Some(LayerShell::bind(&globals, &qh)?)
	} else {
		None
	};

	let (auth_tx, auth_rx) = mpsc::channel();
	let (wake_tx, wake_rx) = UnixStream::pair()?;
	wake_rx.set_nonblocking(true)?;

	let uid = unsafe { libc::geteuid() };
	let socket_path = redface_runtime::Config::load_default()
		.map(|config| config.socket)
		.unwrap_or_else(|_| redface_runtime::DEFAULT_SOCKET_PATH.to_owned());

	let mut app = App {
		registry_state: RegistryState::new(&globals),
		seat_state: SeatState::new(&globals, &qh),
		output_state: OutputState::new(&globals, &qh),
		compositor,
		session_lock_state,
		session_lock: None,
		session_live: false,
		layer_shell,
		test,
		exit: false,
		lock_denied: false,
		gpu,
		viewporter,
		gpu_error: None,
		atlas: GlyphAtlas::new(),
		epoch: Instant::now(),
		surfaces: Vec::new(),
		ui: UiState::new(),
		config,
		fonts,
		keyboard: None,
		pointer: None,
		uid,
		username: crate::auth::current_username()?,
		socket_path,
		face: None,
		pam_running: Arc::new(AtomicBool::new(false)),
		auth_tx,
		auth_rx,
		wake_rx,
		wake_tx,
		last_minute: -1,
	};

	// Populate the output registry before creating any surfaces (test-mode
	// layer surfaces are created from new_output during this roundtrip).
	event_queue.roundtrip(&mut app)?;
	if !test {
		// The lock object must be kept alive: dropping it destroys the lock
		// while the session stays locked (Hyprland's lockdead screen).
		let lock = app.session_lock_state.as_ref().expect("session lock state").lock(&qh)?;
		app.session_lock = Some(lock);
		app.session_live = true;
		// The spec expects lock surfaces immediately; Hyprland only sends the
		// locked event once a frame rendered on every output.
		let outputs: Vec<wl_output::WlOutput> = app.output_state.outputs().collect();
		for output in &outputs {
			app.add_lock_surface(&qh, output);
		}
		if !app.surfaces.iter().any(|entry| entry.primary)
			&& let Some(first) = app.surfaces.first_mut()
		{
			first.primary = true;
		}
	}

	let mut fds = [
		libc::pollfd {
			fd: conn.as_fd().as_raw_fd(),
			events: libc::POLLIN,
			revents: 0,
		},
		libc::pollfd {
			fd: app.wake_rx.as_raw_fd(),
			events: libc::POLLIN,
			revents: 0,
		},
	];

	while !app.exit {
		conn.flush()?;
		for fd in &mut fds {
			fd.revents = 0;
		}
		if let Some(guard) = event_queue.prepare_read() {
			let n = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, 500) };
			if n > 0 && fds[0].revents & libc::POLLIN != 0 {
				// WouldBlock is documented and benign (the read races poll()).
				match guard.read() {
					Ok(_) => {}
					Err(wayland_client::backend::WaylandError::Io(err))
						if err.kind() == std::io::ErrorKind::WouldBlock => {}
					Err(err) => return Err(err.into()),
				}
			}
			// Otherwise the guard is dropped, cancelling the read.
		}
		event_queue.dispatch_pending(&mut app)?;

		// Drain the auth wake pipe, then handle all queued auth results.
		let mut buf = [0u8; 64];
		loop {
			match app.wake_rx.read(&mut buf) {
				Ok(0) | Err(_) => break,
				Ok(_) => {}
			}
		}
		while let Ok(event) = app.auth_rx.try_recv() {
			app.handle_auth(&conn, &qh, event);
		}

		// Clock text changes once a minute; that is a scene rebuild, not just
		// a redraw.
		let minute = ui::local_now().min;
		if minute != app.last_minute {
			app.last_minute = minute;
			app.request_redraw(&conn, &qh);
		}
	}

	if !test {
		// The spec requires wl_display.sync after unlock_and_destroy so the
		// compositor has actually unlocked before we exit.
		event_queue.roundtrip(&mut app)?;
	}
	conn.flush()?;
	if let Some(err) = app.gpu_error {
		return Err(err.into());
	}
	if app.lock_denied {
		return Err("the compositor denied the session lock (ext-session-lock-v1 unsupported?)".into());
	}
	Ok(())
}
