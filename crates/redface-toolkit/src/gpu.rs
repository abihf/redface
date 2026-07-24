//! GLES renderer shared by redface apps, driven by EGL + glow. Layout inputs
//! come in as a `Scene` plus `Uniforms`; all animations run in the shape
//! vertex shader off `Uniforms::time`.
//!
//! EGL surfaces are created from pre-existing Wayland wl_surface objects (the
//! caller owns those); there is no windowing library dependency.

use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;

use glow::HasContext;
use wayland_client::Proxy;

use crate::scene::{ATLAS_SIZE, AtlasRect, Scene, ShapeInstance, TextInstance, Uniforms};

// ---------------------------------------------------------------------------
// EGL thin FFI — the API is tiny, no crate needed

type EGLBoolean = u32;
type EGLint = i32;
type EGLAttrib = libc::intptr_t;
type EGLDisplay = *mut std::ffi::c_void;
type EGLConfig = *mut std::ffi::c_void;
type EGLContext = *mut std::ffi::c_void;
type EGLSurface = *mut std::ffi::c_void;

const EGL_TRUE: EGLBoolean = 1;
const EGL_NO_CONTEXT: EGLContext = std::ptr::null_mut();
const EGL_NO_SURFACE: EGLSurface = std::ptr::null_mut();

const EGL_RED_SIZE: EGLint = 0x3024;
const EGL_GREEN_SIZE: EGLint = 0x3023;
const EGL_BLUE_SIZE: EGLint = 0x3022;
const EGL_ALPHA_SIZE: EGLint = 0x3021;
const EGL_SURFACE_TYPE: EGLint = 0x3033;
const EGL_WINDOW_BIT: EGLint = 0x0004;
const EGL_RENDERABLE_TYPE: EGLint = 0x3040;
const EGL_OPENGL_ES3_BIT: EGLint = 0x0040;
const EGL_CONTEXT_MAJOR_VERSION: EGLint = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: EGLint = 0x30FB;
const EGL_NONE: EGLint = 0x3038;
const EGL_PLATFORM_WAYLAND_KHR: EGLint = 0x31D8;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;
#[allow(dead_code)]
const EGL_EXTENSIONS: EGLint = 0x3055;

#[link(name = "EGL")]
unsafe extern "C" {
	fn eglGetDisplay(native_display: EGLAttrib) -> EGLDisplay;
	fn eglInitialize(dpy: EGLDisplay, major: *mut EGLint, minor: *mut EGLint) -> EGLBoolean;
	fn eglChooseConfig(
		dpy: EGLDisplay,
		attrib_list: *const EGLint,
		configs: *mut EGLConfig,
		config_size: EGLint,
		num_config: *mut EGLint,
	) -> EGLBoolean;
	fn eglCreateContext(
		dpy: EGLDisplay,
		config: EGLConfig,
		share_context: EGLContext,
		attrib_list: *const EGLint,
	) -> EGLContext;
	fn eglMakeCurrent(
		dpy: EGLDisplay,
		draw: EGLSurface,
		read: EGLSurface,
		ctx: EGLContext,
	) -> EGLBoolean;
	fn eglSwapBuffers(dpy: EGLDisplay, surface: EGLSurface) -> EGLBoolean;
	#[allow(dead_code)]
	fn eglDestroySurface(dpy: EGLDisplay, surface: EGLSurface) -> EGLBoolean;
	fn eglDestroyContext(dpy: EGLDisplay, ctx: EGLContext) -> EGLBoolean;
	fn eglTerminate(dpy: EGLDisplay) -> EGLBoolean;
	fn eglCreatePbufferSurface(
		dpy: EGLDisplay,
		config: EGLConfig,
		attrib_list: *const EGLint,
	) -> EGLSurface;

	fn eglCreatePlatformWindowSurface(
		dpy: EGLDisplay,
		config: EGLConfig,
		native_window: *mut std::ffi::c_void,
		attrib_list: *const EGLAttrib,
	) -> EGLSurface;
	fn eglGetProcAddress(procname: *const c_char) -> *mut std::ffi::c_void;
	#[allow(dead_code)]
	fn eglQueryString(dpy: EGLDisplay, name: EGLint) -> *const c_char;
	fn eglSwapInterval(dpy: EGLDisplay, interval: EGLint) -> EGLBoolean;
	fn eglGetError() -> EGLint;
	fn eglGetPlatformDisplay(
		platform: EGLint,
		native_display: *mut std::ffi::c_void,
		attrib_list: *const EGLAttrib,
	) -> EGLDisplay;
}

// libwayland-egl: wraps wl_surface into an EGL-compatible native window.
#[link(name = "wayland-egl")]
unsafe extern "C" {
	fn wl_egl_window_create(
		surface: *mut std::ffi::c_void,
		width: i32,
		height: i32,
	) -> *mut std::ffi::c_void;
	fn wl_egl_window_destroy(window: *mut std::ffi::c_void);
	fn wl_egl_window_resize(
		window: *mut std::ffi::c_void,
		width: i32,
		height: i32,
		dx: i32,
		dy: i32,
	);
}

// ---------------------------------------------------------------------------
// GLSL shader compilation helpers

fn compile_shader(gl: &glow::Context, src: &str, kind: u32) -> Result<glow::Shader, String> {
	unsafe {
		let shader = gl.create_shader(kind).map_err(|e| format!("create shader: {e}"))?;
		gl.shader_source(shader, src);
		gl.compile_shader(shader);
		if !gl.get_shader_compile_status(shader) {
			let log = gl.get_shader_info_log(shader);
			gl.delete_shader(shader);
			return Err(log);
		}
		Ok(shader)
	}
}

fn link_program(
	gl: &glow::Context,
	vert_src: &str,
	frag_src: &str,
) -> Result<glow::Program, String> {
	unsafe {
		let vs = compile_shader(gl, vert_src, glow::VERTEX_SHADER)?;
		let fs = compile_shader(gl, frag_src, glow::FRAGMENT_SHADER)?;
		let program = gl.create_program().map_err(|e| format!("create program: {e}"))?;
		gl.attach_shader(program, vs);
		gl.attach_shader(program, fs);
		gl.link_program(program);
		if !gl.get_program_link_status(program) {
			let log = gl.get_program_info_log(program);
			gl.delete_program(program);
			gl.delete_shader(vs);
			gl.delete_shader(fs);
			return Err(log);
		}
		// Shaders are linked into the program; we can detach and delete them.
		gl.detach_shader(program, vs);
		gl.detach_shader(program, fs);
		gl.delete_shader(vs);
		gl.delete_shader(fs);
		Ok(program)
	}
}

// ---------------------------------------------------------------------------
// GpuError

/// Anything that prevents the GL renderer from coming up. There is no
/// fallback; the caller is expected to abort.
#[derive(Debug)]
pub struct GpuError(pub String);

impl fmt::Display for GpuError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "gpu error: {}", self.0)
	}
}

impl Error for GpuError {}

// ---------------------------------------------------------------------------
// Gpu — shared GL state

pub struct Gpu {
	egl_display: EGLDisplay,
	egl_config: EGLConfig,
	egl_context: EGLContext,
	pbuffer: EGLSurface,
	gl: glow::Context,

	// Programs
	bg_program: glow::Program,
	shape_program: glow::Program,
	text_program: glow::Program,

	// Shape uniform locations
	shape_u_surface_size: glow::UniformLocation,
	shape_u_time: glow::UniformLocation,
	shape_u_shake_start: glow::UniformLocation,
	shape_u_face_toggled_at: glow::UniformLocation,
	shape_u_face_active: glow::UniformLocation,
	shape_u_box_color: glow::UniformLocation,
	shape_u_text_color: glow::UniformLocation,
	shape_u_accent_color: glow::UniformLocation,

	// Textures (shared across surfaces)
	atlas: glow::Texture,
	bg_texture: Option<glow::Texture>,
	bg_size: [f32; 2],
	placeholder: glow::Texture,
}

impl Gpu {
	pub fn new(conn: &wayland_client::Connection) -> Result<Gpu, GpuError> {
		// --- EGL init ---
		let wl_display_ptr = conn.backend().display_id().as_ptr() as *mut std::ffi::c_void;

		// Try EGL 1.5 platform-aware first, then eglGetDisplay, then default.
		let egl_display = unsafe {
			let d = eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, wl_display_ptr, [EGL_NONE as EGLAttrib].as_ptr());
			if !d.is_null() { d }
			else {
				let d = eglGetDisplay(wl_display_ptr as EGLAttrib);
				if !d.is_null() { d }
				else { eglGetDisplay(0) } // EGL_DEFAULT_DISPLAY
			}
		};
		let (mut major, mut minor) = (0i32, 0i32);
		if unsafe { eglInitialize(egl_display, &mut major, &mut minor) } != EGL_TRUE {
			return Err(GpuError("eglInitialize failed".to_owned()));
		}

		let config_attribs = [
			EGL_SURFACE_TYPE,
			EGL_WINDOW_BIT,
			EGL_RED_SIZE,
			8,
			EGL_GREEN_SIZE,
			8,
			EGL_BLUE_SIZE,
			8,
			EGL_ALPHA_SIZE,
			8,
			EGL_RENDERABLE_TYPE,
			EGL_OPENGL_ES3_BIT,
			EGL_NONE,
		];
		let mut config: EGLConfig = std::ptr::null_mut();
		let mut num_config: EGLint = 0;
		if unsafe {
			eglChooseConfig(
				egl_display,
				config_attribs.as_ptr(),
				&mut config,
				1,
				&mut num_config,
			)
		} != EGL_TRUE || num_config == 0
		{
			return Err(GpuError("eglChooseConfig failed".to_owned()));
		}

		let context_attribs = [EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE];
		let egl_context = unsafe {
			eglCreateContext(egl_display, config, EGL_NO_CONTEXT, context_attribs.as_ptr())
		};
		if egl_context.is_null() {
			return Err(GpuError("eglCreateContext failed".to_owned()));
		}

		// Bootstrap: create a 1×1 pbuffer surface so we can make the context
		// current for GL resource creation. EGL_NO_SURFACE doesn't work on
		// all drivers (notably NVIDIA).
		let pbuffer_attribs = [
			EGL_WIDTH as EGLint,
			1,
			EGL_HEIGHT as EGLint,
			1,
			EGL_NONE,
		];
		let bootstrap_surface = unsafe {
			eglCreatePbufferSurface(egl_display, config, pbuffer_attribs.as_ptr())
		};
		if bootstrap_surface.is_null() {
			return Err(GpuError("eglCreatePbufferSurface failed".to_owned()));
		}
		unsafe {
			eglMakeCurrent(egl_display, bootstrap_surface, bootstrap_surface, egl_context);
		}

		// --- GL via glow (needs a current context) ---
		let gl = unsafe {
			glow::Context::from_loader_function(|name| {
				let cname = CString::new(name).unwrap();
				eglGetProcAddress(cname.as_ptr())
			})
		};

		// --- Compile shaders ---
		let bg_program = link_program(
			&gl,
			include_str!("../shaders/bg.vert"),
			include_str!("../shaders/bg.frag"),
		)
		.map_err(GpuError)?;
		let shape_program = link_program(
			&gl,
			include_str!("../shaders/shape.vert"),
			include_str!("../shaders/shape.frag"),
		)
		.map_err(GpuError)?;
		let text_program = link_program(
			&gl,
			include_str!("../shaders/text.vert"),
			include_str!("../shaders/text.frag"),
		)
		.map_err(GpuError)?;

		// --- Uniform locations for shape (most used) ---
		unsafe {
			let shape_u_surface_size = gl
				.get_uniform_location(shape_program, "u_surface_size")
				.ok_or_else(|| GpuError("shape u_surface_size not found".to_owned()))?;
			let shape_u_time = gl
				.get_uniform_location(shape_program, "u_time")
				.ok_or_else(|| GpuError("shape u_time not found".to_owned()))?;
			let shape_u_shake_start = gl
				.get_uniform_location(shape_program, "u_shake_start")
				.ok_or_else(|| GpuError("shape u_shake_start not found".to_owned()))?;
			let shape_u_face_toggled_at = gl
				.get_uniform_location(shape_program, "u_face_toggled_at")
				.ok_or_else(|| GpuError("shape u_face_toggled_at not found".to_owned()))?;
			let shape_u_face_active = gl
				.get_uniform_location(shape_program, "u_face_active")
				.ok_or_else(|| GpuError("shape u_face_active not found".to_owned()))?;
			let shape_u_box_color = gl
				.get_uniform_location(shape_program, "u_box_color")
				.ok_or_else(|| GpuError("shape u_box_color not found".to_owned()))?;
			let shape_u_text_color = gl
				.get_uniform_location(shape_program, "u_text_color")
				.ok_or_else(|| GpuError("shape u_text_color not found".to_owned()))?;
			let shape_u_accent_color = gl
				.get_uniform_location(shape_program, "u_accent_color")
				.ok_or_else(|| GpuError("shape u_accent_color not found".to_owned()))?;

			// --- Shared textures ---
			let atlas = gl.create_texture().map_err(|e| GpuError(format!("atlas texture: {e}")))?;
			gl.bind_texture(glow::TEXTURE_2D, Some(atlas));
			gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
			gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
			gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
			gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
			gl.tex_image_2d(
				glow::TEXTURE_2D,
				0,
				glow::R8 as i32,
				ATLAS_SIZE as i32,
				ATLAS_SIZE as i32,
				0,
				glow::RED,
				glow::UNSIGNED_BYTE,
				glow::PixelUnpackData::Slice(None),
			);

			let placeholder = gl.create_texture().map_err(|e| GpuError(format!("placeholder: {e}")))?;
			gl.bind_texture(glow::TEXTURE_2D, Some(placeholder));
			gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
			gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
			let white: [u8; 4] = [255, 255, 255, 255];
			gl.tex_image_2d(
				glow::TEXTURE_2D,
				0,
				glow::RGBA8 as i32,
				1,
				1,
				0,
				glow::RGBA,
				glow::UNSIGNED_BYTE,
				glow::PixelUnpackData::Slice(Some(&white)),
			);
			gl.bind_texture(glow::TEXTURE_2D, None);

			// Disable vsync; Wayland frame callbacks drive the refresh.
			eglSwapInterval(egl_display, 0);

			// Keep the pbuffer alive for background uploads etc. when no real
			// surface is current yet. Unbind but don't destroy.
			eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

			Ok(Gpu {
				egl_display,
				egl_config: config,
				egl_context,
				pbuffer: bootstrap_surface,
				gl,
				bg_program,
				shape_program,
				text_program,
				shape_u_surface_size,
				shape_u_time,
				shape_u_shake_start,
				shape_u_face_toggled_at,
				shape_u_face_active,
				shape_u_box_color,
				shape_u_text_color,
				shape_u_accent_color,
				atlas,
				bg_texture: None,
				bg_size: [0.0, 0.0],
				placeholder,
			})
		}
	}

	pub fn set_background(&mut self, rgba: Option<(&[u8], u32, u32)>) {
		self.ensure_context();
		unsafe {
			let gl = &self.gl;
			match rgba {
				Some((data, w, h)) if w > 0 && h > 0 => {
					let tex = match gl.create_texture() {
						Ok(t) => t,
						Err(_) => return,
					};
					gl.bind_texture(glow::TEXTURE_2D, Some(tex));
					gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
					gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
					gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
					gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
					gl.tex_image_2d(
						glow::TEXTURE_2D,
						0,
						glow::RGBA8 as i32,
						w as i32,
						h as i32,
						0,
						glow::RGBA,
						glow::UNSIGNED_BYTE,
						glow::PixelUnpackData::Slice(Some(data)),
					);
					if let Some(old) = self.bg_texture.take() {
						gl.delete_texture(old);
					}
					self.bg_texture = Some(tex);
					self.bg_size = [w as f32, h as f32];
				}
				_ => {
					if let Some(old) = self.bg_texture.take() {
						gl.delete_texture(old);
					}
					self.bg_size = [0.0, 0.0];
				}
			}
			gl.bind_texture(glow::TEXTURE_2D, None);
		}
	}

	pub fn background_size(&self) -> [f32; 2] {
		self.bg_size
	}

	pub fn upload_glyphs(&self, writes: &[(AtlasRect, Vec<u8>)]) {
		self.ensure_context();
		unsafe {
			let gl = &self.gl;
			gl.bind_texture(glow::TEXTURE_2D, Some(self.atlas));
			gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
			for (rect, data) in writes {
				if rect.w == 0 || rect.h == 0 {
					continue;
				}
				gl.tex_sub_image_2d(
					glow::TEXTURE_2D,
					0,
					rect.x as i32,
					rect.y as i32,
					rect.w as i32,
					rect.h as i32,
					glow::RED,
					glow::UNSIGNED_BYTE,
					glow::PixelUnpackData::Slice(Some(data)),
				);
			}
			gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
			gl.bind_texture(glow::TEXTURE_2D, None);
		}
	}

	pub fn create_surface(
		&self,
		_conn: &wayland_client::Connection,
		surface: &wayland_client::protocol::wl_surface::WlSurface,
		width: u32,
		height: u32,
	) -> Result<GpuSurface, GpuError> {
		let wl_ptr = surface.id().as_ptr() as *mut std::ffi::c_void;

		// Wayland requires wrapping the wl_surface in a wl_egl_window.
		let egl_window = unsafe { wl_egl_window_create(wl_ptr, width as i32, height as i32) };
		if egl_window.is_null() {
			return Err(GpuError("wl_egl_window_create returned null".to_owned()));
		}

		let egl_surface = unsafe {
			eglCreatePlatformWindowSurface(
				self.egl_display,
				self.egl_config,
				egl_window,
				[EGL_NONE as EGLAttrib].as_ptr(),
			)
		};
		if egl_surface.is_null() {
			let err = unsafe { eglGetError() };
			unsafe { wl_egl_window_destroy(egl_window) };
			return Err(GpuError(format!("eglCreatePlatformWindowSurface failed (EGL error 0x{err:x})")));
		}

		Ok(GpuSurface {
			egl_surface,
			egl_window,
			shape_vbo: None,
			shape_cap: 0,
			text_vbo: None,
			text_cap: 0,
			width,
			height,
		})
	}

	/// Activate the GL context on the pbuffer for off-screen operations
	/// (texture uploads, glyph uploads) when no real surface is current.
	fn ensure_context(&self) {
		unsafe {
			eglMakeCurrent(self.egl_display, self.pbuffer, self.pbuffer, self.egl_context);
		}
	}

	/// Bind the GL context to a surface (called before rendering).
	unsafe fn make_current(&self, surface: EGLSurface) -> bool {
		unsafe { eglMakeCurrent(self.egl_display, surface, surface, self.egl_context) == EGL_TRUE }
	}
}

impl Drop for Gpu {
	fn drop(&mut self) {
		unsafe {
			let gl = &self.gl;
			if let Some(tex) = self.bg_texture {
				gl.delete_texture(tex);
			}
			gl.delete_texture(self.atlas);
			gl.delete_texture(self.placeholder);
			gl.delete_program(self.bg_program);
			gl.delete_program(self.shape_program);
			gl.delete_program(self.text_program);
			eglDestroySurface(self.egl_display, self.pbuffer);
			eglDestroyContext(self.egl_display, self.egl_context);
			eglTerminate(self.egl_display);
		}
	}
}

// ---------------------------------------------------------------------------
// GpuSurface — per-output state

pub struct GpuSurface {
	egl_surface: EGLSurface,
	egl_window: *mut std::ffi::c_void, // wl_egl_window *
	shape_vbo: Option<glow::Buffer>,
	shape_cap: usize,
	text_vbo: Option<glow::Buffer>,
	text_cap: usize,
	width: u32,
	height: u32,
}

impl Drop for GpuSurface {
	fn drop(&mut self) {
		unsafe { wl_egl_window_destroy(self.egl_window) };
	}
}

impl GpuSurface {
	pub fn resize(&mut self, _gpu: &Gpu, width: u32, height: u32) {
		self.width = width;
		self.height = height;
		unsafe {
			wl_egl_window_resize(self.egl_window, width as i32, height as i32, 0, 0);
		}
	}

	pub fn render(&mut self, gpu: &Gpu, scene: &Scene, uniforms: &Uniforms) {
		unsafe {
			let gl = &gpu.gl;
			if !gpu.make_current(self.egl_surface) {
				return;
			}

			let w = self.width as i32;
			let h = self.height as i32;

			gl.viewport(0, 0, w, h);
			gl.enable(glow::BLEND);
			gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
			gl.clear_color(0.0, 0.0, 0.0, 0.0);
			gl.clear(glow::COLOR_BUFFER_BIT);

			// --- Background pass ---
			gl.use_program(Some(gpu.bg_program));
			let bg_tex = gpu.bg_texture.unwrap_or(gpu.placeholder);
			gl.active_texture(glow::TEXTURE0);
			gl.bind_texture(glow::TEXTURE_2D, Some(bg_tex));
			gl.uniform_1_i32(
				gl.get_uniform_location(gpu.bg_program, "u_tex").as_ref(),
				0,
			);
			gl.uniform_4_f32(
				gl.get_uniform_location(gpu.bg_program, "u_bg_color").as_ref(),
				uniforms.bg_color[0],
				uniforms.bg_color[1],
				uniforms.bg_color[2],
				uniforms.bg_color[3],
			);
			gl.uniform_2_f32(
				gl.get_uniform_location(gpu.bg_program, "u_surface_size").as_ref(),
				uniforms.surface_size[0],
				uniforms.surface_size[1],
			);
			gl.uniform_2_f32(
				gl.get_uniform_location(gpu.bg_program, "u_bg_image_size").as_ref(),
				gpu.bg_size[0],
				gpu.bg_size[1],
			);
			gl.draw_arrays(glow::TRIANGLES, 0, 3);

			// --- Shape pass ---
			if !scene.shapes.is_empty() {
				gl.use_program(Some(gpu.shape_program));

				gl.uniform_2_f32(
					Some(&gpu.shape_u_surface_size),
					uniforms.surface_size[0],
					uniforms.surface_size[1],
				);
				gl.uniform_1_f32(Some(&gpu.shape_u_time), uniforms.time);
				gl.uniform_1_f32(Some(&gpu.shape_u_shake_start), uniforms.shake_start);
				gl.uniform_1_f32(Some(&gpu.shape_u_face_toggled_at), uniforms.face_toggled_at);
				gl.uniform_1_f32(Some(&gpu.shape_u_face_active), uniforms.face_active);
				gl.uniform_4_f32(
					Some(&gpu.shape_u_box_color),
					uniforms.box_color[0],
					uniforms.box_color[1],
					uniforms.box_color[2],
					uniforms.box_color[3],
				);
				gl.uniform_4_f32(
					Some(&gpu.shape_u_text_color),
					uniforms.text_color[0],
					uniforms.text_color[1],
					uniforms.text_color[2],
					uniforms.text_color[3],
				);
				gl.uniform_4_f32(
					Some(&gpu.shape_u_accent_color),
					uniforms.accent_color[0],
					uniforms.accent_color[1],
					uniforms.accent_color[2],
					uniforms.accent_color[3],
				);

				ensure_vbo(gl, &mut self.shape_vbo, &mut self.shape_cap, scene.shapes.len());
				let vbo = self.shape_vbo.unwrap();
				gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
				gl.buffer_sub_data_u8_slice(
					glow::ARRAY_BUFFER,
					0,
					bytemuck::cast_slice(&scene.shapes),
				);

				setup_shape_attribs(gl);
				gl.draw_arrays_instanced(glow::TRIANGLES, 0, 6, scene.shapes.len() as i32);
			}

			// --- Text pass ---
			if !scene.texts.is_empty() {
				gl.use_program(Some(gpu.text_program));

				gl.uniform_2_f32(
					gl.get_uniform_location(gpu.text_program, "u_surface_size").as_ref(),
					uniforms.surface_size[0],
					uniforms.surface_size[1],
				);

				gl.active_texture(glow::TEXTURE0);
				gl.bind_texture(glow::TEXTURE_2D, Some(gpu.atlas));
				gl.uniform_1_i32(
					gl.get_uniform_location(gpu.text_program, "u_tex").as_ref(),
					0,
				);

				ensure_vbo(gl, &mut self.text_vbo, &mut self.text_cap, scene.texts.len());
				let vbo = self.text_vbo.unwrap();
				gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
				gl.buffer_sub_data_u8_slice(
					glow::ARRAY_BUFFER,
					0,
					bytemuck::cast_slice(&scene.texts),
				);

				setup_text_attribs(gl);
				gl.draw_arrays_instanced(glow::TRIANGLES, 0, 6, scene.texts.len() as i32);
			}

			gl.bind_buffer(glow::ARRAY_BUFFER, None);

			eglSwapBuffers(gpu.egl_display, self.egl_surface);
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers

unsafe fn ensure_vbo(
	gl: &glow::Context,
	buf: &mut Option<glow::Buffer>,
	cap: &mut usize,
	needed: usize,
) { unsafe {
	let _bytes = needed * std::mem::size_of::<ShapeInstance>()
		.max(needed * std::mem::size_of::<TextInstance>())
		.max(64 * std::mem::size_of::<ShapeInstance>());
	if buf.is_some() && needed <= *cap {
		return;
	}
	let new_cap = needed.max(64).next_power_of_two();
	if let Some(old) = buf.take() {
		gl.delete_buffer(old);
	}
	let vbo = gl.create_buffer().unwrap();
	gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
	let size_bytes = (new_cap * std::mem::size_of::<ShapeInstance>()) as i32;
	gl.buffer_data_size(glow::ARRAY_BUFFER, size_bytes, glow::DYNAMIC_DRAW);
	*buf = Some(vbo);
	*cap = new_cap;
}}

unsafe fn setup_shape_attribs(gl: &glow::Context) { unsafe {
	let stride = std::mem::size_of::<ShapeInstance>() as i32;
	// a_center: 2×f32 at offset 0
	gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
	gl.enable_vertex_attrib_array(0);
	gl.vertex_attrib_divisor(0, 1);
	// a_half_size: 2×f32 at offset 8
	gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 8);
	gl.enable_vertex_attrib_array(1);
	gl.vertex_attrib_divisor(1, 1);
	// a_color: 4×f32 at offset 16
	gl.vertex_attrib_pointer_f32(2, 4, glow::FLOAT, false, stride, 16);
	gl.enable_vertex_attrib_array(2);
	gl.vertex_attrib_divisor(2, 1);
	// a_radius: 1×f32 at offset 32
	gl.vertex_attrib_pointer_f32(3, 1, glow::FLOAT, false, stride, 32);
	gl.enable_vertex_attrib_array(3);
	gl.vertex_attrib_divisor(3, 1);
	// a_inner_radius: 1×f32 at offset 36
	gl.vertex_attrib_pointer_f32(4, 1, glow::FLOAT, false, stride, 36);
	gl.enable_vertex_attrib_array(4);
	gl.vertex_attrib_divisor(4, 1);
	// a_birth_time: 1×f32 at offset 40
	gl.vertex_attrib_pointer_f32(5, 1, glow::FLOAT, false, stride, 40);
	gl.enable_vertex_attrib_array(5);
	gl.vertex_attrib_divisor(5, 1);
	// a_kind: 1×u32 at offset 44 (use INTEGER + UNSIGNED_INT for uint attributes)
	gl.vertex_attrib_pointer_i32(6, 1, glow::UNSIGNED_INT, stride, 44);
	gl.enable_vertex_attrib_array(6);
	gl.vertex_attrib_divisor(6, 1);
}}

unsafe fn setup_text_attribs(gl: &glow::Context) { unsafe {
	let stride = std::mem::size_of::<TextInstance>() as i32;
	gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
	gl.enable_vertex_attrib_array(0);
	gl.vertex_attrib_divisor(0, 1);
	gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 8);
	gl.enable_vertex_attrib_array(1);
	gl.vertex_attrib_divisor(1, 1);
	gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, stride, 16);
	gl.enable_vertex_attrib_array(2);
	gl.vertex_attrib_divisor(2, 1);
	gl.vertex_attrib_pointer_f32(3, 2, glow::FLOAT, false, stride, 24);
	gl.enable_vertex_attrib_array(3);
	gl.vertex_attrib_divisor(3, 1);
	gl.vertex_attrib_pointer_f32(4, 4, glow::FLOAT, false, stride, 32);
	gl.enable_vertex_attrib_array(4);
	gl.vertex_attrib_divisor(4, 1);
}}
