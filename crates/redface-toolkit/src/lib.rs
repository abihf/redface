//! Shared Wayland/GLES UI toolkit for redface binaries (dylib).
//!
//! Wraps EGL + glow (OpenGL ES 3.0 renderer, see [`gpu`]) and
//! smithay-client-toolkit (session-lock and layer-shell event loop, see
//! [`wayland`]) behind a small app-facing API: implement [`App`], call [`run`].

pub mod gpu;
pub mod scene;
pub mod text;
pub mod wayland;

pub use wayland::{App, LayerConfig, Role, RunConfig, run};

// Re-exports so consumers don't need direct sctk/wayland-client deps for the
// common event types.
pub use smithay_client_toolkit::seat::keyboard::{KeyEvent, Keysym};
pub use smithay_client_toolkit::seat::pointer::PointerEventKind;
pub use smithay_client_toolkit::shell::wlr_layer::{Anchor, KeyboardInteractivity, Layer};
