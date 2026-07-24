//! redface-osd: small Wayland OSD (wlr-layer-shell, top layer) shown while
//! face recognition runs. Esc or the Cancel button aborts; the process exits
//! 1 when cancelled, 0 when the OSD was closed any other way.

mod ui;

use std::process::ExitCode;
use std::time::Instant;

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
	cancelled: bool,
	hover_cancel: bool,
}

impl OsdApp {
	fn new() -> Self {
		Self {
			cancelled: false,
			hover_cancel: false,
		}
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
		ui::build_scene(self.hover_cancel, fonts, atlas, width, height, scale)
	}

	fn uniforms(&self, epoch: Instant) -> Uniforms {
		Uniforms {
			// Overwritten by the runner.
			surface_size: [0.0, 0.0],
			bg_image_size: [0.0, 0.0],
			// Fully transparent: the rounded panel is the only backdrop.
			bg_color: [0.0, 0.0, 0.0, 0.0],
			text_color: ui::TEXT_COLOR,
			box_color: ui::BOX_COLOR,
			accent_color: ui::ACCENT_COLOR,
			time: epoch.elapsed().as_secs_f32(),
			shake_start: -1.0,
			// Active since epoch: the pulse ring animates continuously.
			face_toggled_at: 0.0,
			face_active: 1.0,
		}
	}

	fn on_key(&mut self, event: &KeyEvent) {
		if event.keysym == Keysym::Escape {
			self.cancelled = true;
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
				self.cancelled = true;
			}
			_ => {}
		}
	}

	// The pulse ring animates continuously.
	fn animating(&self) -> bool {
		true
	}

	fn should_exit(&self) -> bool {
		self.cancelled
	}
}

fn main() -> ExitCode {
	let mut app = OsdApp::new();
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
	match run(config, &mut app) {
		Ok(()) if app.cancelled => ExitCode::from(1),
		Ok(()) => ExitCode::SUCCESS,
		Err(err) => {
			eprintln!("redface-osd: {err}");
			ExitCode::from(2)
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn key_event(keysym: Keysym) -> KeyEvent {
		KeyEvent {
			time: 0,
			raw_code: 0,
			keysym,
			utf8: None,
		}
	}

	#[test]
	fn escape_cancels() {
		let mut app = OsdApp::new();
		assert!(!app.should_exit());
		app.on_key(&key_event(Keysym::Escape));
		assert!(app.should_exit());
	}

	#[test]
	fn other_keys_do_not_cancel() {
		let mut app = OsdApp::new();
		app.on_key(&key_event(Keysym::Return));
		app.on_key(&key_event(Keysym::a));
		assert!(!app.should_exit());
	}

	#[test]
	fn left_click_on_button_cancels() {
		let mut app = OsdApp::new();
		let lay = ui::layout(SURFACE_SIZE.0, SURFACE_SIZE.1, 1.0);
		let (bx, by, bw, bh) = lay.cancel_button;
		let center = ((bx + bw / 2.0) as f64, (by + bh / 2.0) as f64);
		app.on_pointer(
			PointerEventKind::Press {
				time: 0,
				button: BTN_LEFT,
				serial: 0,
			},
			center,
		);
		assert!(app.should_exit());
	}

	#[test]
	fn clicks_elsewhere_do_not_cancel() {
		let mut app = OsdApp::new();
		let lay = ui::layout(SURFACE_SIZE.0, SURFACE_SIZE.1, 1.0);
		let (bx, by, _, _) = lay.cancel_button;
		// Left click outside the button.
		app.on_pointer(
			PointerEventKind::Press {
				time: 0,
				button: BTN_LEFT,
				serial: 0,
			},
			((bx - 4.0) as f64, (by - 4.0) as f64),
		);
		// Right click inside the button.
		let (bx, by, bw, bh) = lay.cancel_button;
		app.on_pointer(
			PointerEventKind::Press {
				time: 0,
				button: 0x111,
				serial: 0,
			},
			((bx + bw / 2.0) as f64, (by + bh / 2.0) as f64),
		);
		assert!(!app.should_exit());
	}

	#[test]
	fn pointer_motion_tracks_button_hover() {
		let mut app = OsdApp::new();
		let lay = ui::layout(SURFACE_SIZE.0, SURFACE_SIZE.1, 1.0);
		let (bx, by, bw, bh) = lay.cancel_button;
		assert!(!app.hover_cancel);
		app.on_pointer(
			PointerEventKind::Motion { time: 0 },
			((bx + bw / 2.0) as f64, (by + bh / 2.0) as f64),
		);
		assert!(app.hover_cancel);
		app.on_pointer(PointerEventKind::Motion { time: 0 }, (1.0, 1.0));
		assert!(!app.hover_cancel);
		app.on_pointer(
			PointerEventKind::Enter { serial: 0 },
			((bx + 1.0) as f64, (by + 1.0) as f64),
		);
		assert!(app.hover_cancel);
		app.on_pointer(PointerEventKind::Leave { serial: 0 }, (0.0, 0.0));
		assert!(!app.hover_cancel);
	}

	#[test]
	fn uniforms_are_transparent_and_pulsing() {
		let app = OsdApp::new();
		let u = app.uniforms(Instant::now());
		assert_eq!(u.bg_color, [0.0, 0.0, 0.0, 0.0]);
		assert_eq!(u.text_color, ui::TEXT_COLOR);
		assert_eq!(u.box_color, ui::BOX_COLOR);
		assert_eq!(u.accent_color, ui::ACCENT_COLOR);
		assert!(u.time >= 0.0);
		assert_eq!(u.shake_start, -1.0);
		assert_eq!(u.face_toggled_at, 0.0);
		assert_eq!(u.face_active, 1.0);
	}

	#[test]
	fn osd_animates_continuously() {
		let app = OsdApp::new();
		assert!(app.animating());
	}
}
