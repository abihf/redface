//! Generic smithay-client-toolkit event loop driving an [`App`]: session-lock
//! or layer-shell surfaces on one or all outputs, frame-callback paced
//! rendering through [`crate::gpu`], scenes rebuilt only when dirty.

use std::error::Error;
use std::os::fd::{AsFd, AsRawFd, RawFd};
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

use crate::gpu::{Gpu, GpuError, GpuSurface};
use crate::scene::{Scene, Uniforms};
use crate::text::{self, Fonts, GlyphAtlas};

/// What kind of surfaces the runner creates.
pub enum Role {
	/// ext-session-lock-v1 lock surfaces on every output.
	SessionLock,
	/// wlr-layer-shell surface(s).
	Layer(LayerConfig),
}

pub struct LayerConfig {
	pub layer: Layer,
	pub anchor: Anchor,
	/// (0, 0) fills the anchor zone (fullscreen when anchored to all edges).
	pub size: (u32, u32),
	pub exclusive_zone: i32,
	pub interactivity: KeyboardInteractivity,
	/// (top, right, bottom, left).
	pub margin: (i32, i32, i32, i32),
	/// Create the surface on every output; otherwise only the first.
	pub all_outputs: bool,
}

pub struct RunConfig {
	pub role: Role,
	pub namespace: String,
	/// Decoded straight-alpha RGBA background image; solid `Uniforms.bg_color`
	/// (alpha 0 = transparent) when `None`.
	pub background: Option<(Vec<u8>, u32, u32)>,
}

/// Application callbacks for the toolkit event loop.
///
/// The runner calls [`App::build_scene`] whenever the scene is dirty (input
/// events, clock minute change, resize) and re-renders at the monitor's
/// native refresh while [`App::animating`] returns true.
pub trait App {
	#[allow(clippy::too_many_arguments)]
	fn build_scene(
		&mut self,
		fonts: &Fonts,
		atlas: &mut GlyphAtlas,
		width: u32,
		height: u32,
		scale: f32,
		epoch: Instant,
		primary: bool,
	) -> Scene;

	/// xdg-output name (e.g. "eDP-1") of the monitor that gets the full UI;
	/// other outputs get `primary = false` scenes. Defaults to the first
	/// connected output when `None`.
	fn primary_output(&self) -> Option<String> {
		None
	}

	/// Colors and animation timestamps; the runner overwrites `surface_size`
	/// and `bg_image_size`.
	fn uniforms(&self, epoch: Instant) -> Uniforms;

	/// After every key/pointer event the runner marks the scene dirty.
	fn on_key(&mut self, event: &KeyEvent);
	fn on_pointer(&mut self, kind: PointerEventKind, position: (f64, f64));

	fn animating(&self) -> bool;
	fn should_exit(&self) -> bool;

	/// Called once before the runner tears down (session unlock etc.).
	fn on_exit(&mut self) {}

	/// Called after every event-loop dispatch (poll auth channels etc.).
	fn on_tick(&mut self) {}

	/// Extra fd added to the poll set; [`App::on_tick`] runs when it wakes.
	fn wake_fd(&self) -> Option<RawFd> {
		None
	}
}

struct Runner {
	app: &'static mut dyn App,
	registry_state: RegistryState,
	seat_state: SeatState,
	output_state: OutputState,
	compositor: CompositorState,
	session_lock: Option<SessionLock>,
	/// True between a successful lock request and `finished()`/unlock; gates
	/// lock-surface creation on hotplugged outputs.
	session_live: bool,
	layer_shell: Option<LayerShell>,
	/// Present for `Role::Layer`; drives hotplug surface creation.
	layer_config: Option<LayerConfig>,
	namespace: String,

	exit: bool,
	/// Exit came from [`App::should_exit`] (not from the compositor tearing
	/// the surfaces down); triggers [`App::on_exit`].
	requested_exit: bool,
	lock_denied: bool,
	gpu: Gpu,
	viewporter: Option<WpViewporter>,
	/// Fatal surface-creation failure (GLES is a hard requirement); surfaced
	/// as the process error after the session has been unlocked cleanly.
	gpu_error: Option<GpuError>,
	fonts: Fonts,
	atlas: GlyphAtlas,
	/// Start of the process; the reference for all animation time uniforms.
	epoch: Instant,
	surfaces: Vec<SurfaceEntry>,
	keyboard: Option<wl_keyboard::WlKeyboard>,
	pointer: Option<wl_pointer::WlPointer>,
	last_minute: i32,
}

struct SurfaceEntry {
	kind: SurfaceKind,
	viewport: Option<WpViewport>,
	output: Option<wl_output::WlOutput>,
	/// EGL surface, created lazily at the first configure.
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
	Layer(LayerSurface),
}

impl SurfaceKind {
	fn wl_surface(&self) -> &wl_surface::WlSurface {
		match self {
			Self::Lock(surface) => surface.wl_surface(),
			Self::Layer(surface) => surface.wl_surface(),
		}
	}

	fn commit(&self) {
		match self {
			Self::Lock(surface) => surface.wl_surface().commit(),
			Self::Layer(surface) => surface.commit(),
		}
	}
}

impl Runner {
	fn surface_index(&self, surface: &wl_surface::WlSurface) -> Option<usize> {
		self.surfaces
			.iter()
			.position(|entry| entry.kind.wl_surface() == surface)
	}

	fn is_primary(&self, output: &wl_output::WlOutput) -> bool {
		let name = self.output_state.info(output).and_then(|info| info.name);
		match self.app.primary_output() {
			Some(want) => name.as_deref() == Some(want.as_str()),
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
		// No initial commit: ext-session-lock sends the first configure on
		// bind, and committing before acking it (or with a null buffer) is a
		// protocol error.
		let primary = self.is_primary(output);
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

	fn add_layer_surface(&mut self, qh: &QueueHandle<Self>, output: &wl_output::WlOutput) {
		let (Some(layer_shell), Some(config)) = (&self.layer_shell, &self.layer_config) else {
			return;
		};
		let surface = self.compositor.create_surface(qh);
		let viewport = self.viewporter.as_ref().map(|vp| vp.get_viewport(&surface, qh, ()));
		let layer = layer_shell.create_layer_surface(qh, surface, config.layer, Some(&self.namespace), Some(output));
		layer.set_anchor(config.anchor);
		layer.set_size(config.size.0, config.size.1);
		layer.set_exclusive_zone(config.exclusive_zone);
		layer.set_keyboard_interactivity(config.interactivity);
		let (top, right, bottom, left) = config.margin;
		layer.set_margin(top, right, bottom, left);
		// Initial empty commit so the compositor sends a configure.
		layer.commit();
		let primary = self.is_primary(output);
		self.surfaces.push(SurfaceEntry {
			kind: SurfaceKind::Layer(layer),
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

	/// Marks every surface dirty and repaints immediately unless a frame
	/// callback is already pending (in which case the frame handler repaints).
	/// State changed, so the scene is rebuilt rather than just redrawn.
	fn request_redraw(&mut self, conn: &Connection, qh: &QueueHandle<Self>) {
		if self.exit {
			return;
		}
		for index in 0..self.surfaces.len() {
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

		// The EGL surface is created on the first configure with a real
		// size. GLES is a hard requirement: failure exits after teardown.
		if self.surfaces[index].gpu.is_none() {
			match self
				.gpu
				.create_surface(conn, self.surfaces[index].kind.wl_surface(), width, height)
			{
				Ok(surface) => self.surfaces[index].gpu = Some(surface),
				Err(err) => {
					self.gpu_error = Some(err);
					self.exit = true;
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
			let scene = self.app.build_scene(
				&self.fonts,
				&mut self.atlas,
				width,
				height,
				scale as f32,
				self.epoch,
				self.surfaces[index].primary,
			);
			let pending = self.atlas.drain_pending();
			if !pending.is_empty() {
				self.gpu.upload_glyphs(&pending);
			}
			let entry = &mut self.surfaces[index];
			entry.scene = Some(scene);
			entry.scene_dirty = false;
		}

		let mut uniforms = self.app.uniforms(self.epoch);
		uniforms.surface_size = [width as f32, height as f32];
		uniforms.bg_image_size = self.gpu.background_size();

		let entry = &mut self.surfaces[index];
		let surface = entry.gpu.as_mut().expect("gpu surface created above");
		surface.render(&self.gpu, entry.scene.as_ref().expect("scene built above"), &uniforms);

		let wl_surface = entry.kind.wl_surface().clone();
		wl_surface.frame(qh, FrameCallbackData(wl_surface.clone()));
		entry.frame_pending = true;
		entry.dirty = false;
		entry.kind.commit();
	}
}

impl CompositorHandler for Runner {
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
		// A scale>1 viewport only displays correctly when the viewport
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
		if self.surfaces[index].dirty || (self.surfaces[index].primary && self.app.animating()) {
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

impl OutputHandler for Runner {
	fn output_state(&mut self) -> &mut OutputState {
		&mut self.output_state
	}

	fn new_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
		// A monitor plugged in while running still needs a surface.
		if self.session_live {
			self.add_lock_surface(qh, &output);
		} else if self.layer_config.as_ref().is_some_and(|config| config.all_outputs) {
			self.add_layer_surface(qh, &output);
		}
	}

	fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}

	fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, output: wl_output::WlOutput) {
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
		}
	}
}

impl SessionLockHandler for Runner {
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

impl LayerShellHandler for Runner {
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

impl SeatHandler for Runner {
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

impl KeyboardHandler for Runner {
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
		self.app.on_key(&event);
		self.request_redraw(conn, qh);
	}

	fn repeat_key(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		_: &wl_keyboard::WlKeyboard,
		_: u32,
		event: KeyEvent,
	) {
		self.app.on_key(&event);
		self.request_redraw(conn, qh);
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

impl PointerHandler for Runner {
	fn pointer_frame(
		&mut self,
		conn: &Connection,
		qh: &QueueHandle<Self>,
		_: &wl_pointer::WlPointer,
		events: &[PointerEvent],
	) {
		let mut handled = false;
		for event in events {
			if self.surface_index(&event.surface).is_none() {
				continue;
			}
			self.app.on_pointer(event.kind.clone(), event.position);
			handled = true;
		}
		if handled {
			self.request_redraw(conn, qh);
		}
	}
}

delegate_registry!(Runner);

impl ProvidesRegistryState for Runner {
	fn registry(&mut self) -> &mut RegistryState {
		&mut self.registry_state
	}
	registry_handlers![OutputState, SeatState];
}

smithay_client_toolkit::delegate_dispatch2!(Runner);
wayland_client::delegate_noop!(Runner: ignore WpViewporter);
wayland_client::delegate_noop!(Runner: ignore WpViewport);

/// Runs the event loop until the app exits. GLES is a hard requirement.
pub fn run(config: RunConfig, app: &mut dyn App) -> Result<(), Box<dyn Error>> {
	let conn = Connection::connect_to_env()?;
	let (globals, mut event_queue) = registry_queue_init(&conn)?;
	let qh = event_queue.handle();

	// GLES is a hard requirement: no EGL display, no UI.
	let mut gpu = Gpu::new(&conn)?;
	gpu.set_background(config.background.as_ref().map(|(data, w, h)| (data.as_slice(), *w, *h)));

	let compositor = CompositorState::bind(&globals, &qh)?;
	let viewporter: Option<WpViewporter> = globals.bind(&qh, 1..=1, ()).ok();
	let (session_lock_state, layer_shell, layer_config) = match config.role {
		Role::SessionLock => (Some(SessionLockState::new(&globals, &qh)), None, None),
		Role::Layer(layer_config) => (None, Some(LayerShell::bind(&globals, &qh)?), Some(layer_config)),
	};

	// The wayland event queue state must be 'static, so the app borrow gets a
	// fake lifetime. SAFETY: the runner is created and dropped entirely within
	// this function (it never escapes into globals or threads), so the
	// 'static reference never outlives the caller's `app` borrow.
	let app: &'static mut dyn App = unsafe { std::mem::transmute::<&mut dyn App, &'static mut dyn App>(app) };
	let mut runner = Runner {
		app,
		registry_state: RegistryState::new(&globals),
		seat_state: SeatState::new(&globals, &qh),
		output_state: OutputState::new(&globals, &qh),
		compositor,
		session_lock: None,
		session_live: false,
		layer_shell,
		layer_config,
		namespace: config.namespace,
		exit: false,
		requested_exit: false,
		lock_denied: false,
		gpu,
		viewporter,
		gpu_error: None,
		fonts: Fonts::load()?,
		atlas: GlyphAtlas::new(),
		epoch: Instant::now(),
		surfaces: Vec::new(),
		keyboard: None,
		pointer: None,
		last_minute: -1,
	};

	// Populate the output registry before creating any surfaces.
	event_queue.roundtrip(&mut runner)?;
	let outputs: Vec<wl_output::WlOutput> = runner.output_state.outputs().collect();
	if let Some(lock_state) = &session_lock_state {
		// The lock object must be kept alive: dropping it destroys the lock
		// while the session stays locked (Hyprland's lockdead screen).
		let lock = lock_state.lock(&qh)?;
		runner.session_lock = Some(lock);
		runner.session_live = true;
		// The spec expects lock surfaces immediately; Hyprland only sends the
		// locked event once a frame rendered on every output.
		for output in &outputs {
			runner.add_lock_surface(&qh, output);
		}
	} else {
		let all_outputs = runner.layer_config.as_ref().is_some_and(|config| config.all_outputs);
		for output in &outputs {
			runner.add_layer_surface(&qh, output);
			if !all_outputs {
				break;
			}
		}
	}

	let wake_fd = runner.app.wake_fd();
	while !runner.exit {
		conn.flush()?;
		let mut fds = vec![libc::pollfd {
			fd: conn.as_fd().as_raw_fd(),
			events: libc::POLLIN,
			revents: 0,
		}];
		if let Some(fd) = wake_fd {
			fds.push(libc::pollfd {
				fd,
				events: libc::POLLIN,
				revents: 0,
			});
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
		event_queue.dispatch_pending(&mut runner)?;

		runner.app.on_tick();
		if runner.app.should_exit() {
			runner.exit = true;
			runner.requested_exit = true;
		}

		// Clock text changes once a minute; that is a scene rebuild, not just
		// a redraw.
		let minute = text::local_now().min;
		if minute != runner.last_minute {
			runner.last_minute = minute;
			runner.request_redraw(&conn, &qh);
		}
	}

	if runner.requested_exit {
		runner.app.on_exit();
	}
	if runner.session_live {
		// The spec requires wl_display.sync after unlock_and_destroy so the
		// compositor has actually unlocked before we exit.
		if let Some(lock) = &runner.session_lock {
			lock.unlock();
		}
		runner.session_live = false;
		event_queue.roundtrip(&mut runner)?;
	}
	conn.flush()?;
	if let Some(err) = runner.gpu_error {
		return Err(err.into());
	}
	if runner.lock_denied {
		return Err("the compositor denied the session lock (ext-session-lock-v1 unsupported?)".into());
	}
	Ok(())
}
