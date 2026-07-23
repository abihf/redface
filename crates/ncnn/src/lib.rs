//! Safe Rust bindings for the ncnn inference engine, wrapping the raw
//! `ncnn-bind` FFI. All `unsafe` lives here; dependents get an RAII API whose
//! handles destroy their ncnn objects on drop.
//!
//! A [`Net`] is usable from one thread at a time: it is `Send` (the handle may
//! be moved) but not `Sync` (it may not be shared).

use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;

/// An error from the ncnn engine.
#[derive(Debug)]
pub enum Error {
	/// A model path or blob name contained an interior NUL byte.
	NulName(String),
	/// `ncnn_net_load_param` failed.
	LoadParam(String),
	/// `ncnn_net_load_model` failed.
	LoadModel(String),
	/// `ncnn_extractor_input` failed.
	Input(String),
	/// `ncnn_extractor_extract` failed.
	Extract(String),
	/// A data buffer had an unexpected length.
	BufferLen { expected: usize, actual: usize },
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::NulName(name) => write!(f, "name '{name}' contains an interior NUL byte"),
			Self::LoadParam(path) => write!(f, "failed to load ncnn param '{path}'"),
			Self::LoadModel(path) => write!(f, "failed to load ncnn model '{path}'"),
			Self::Input(name) => write!(f, "failed to set ncnn input '{name}'"),
			Self::Extract(name) => write!(f, "failed to extract ncnn output '{name}'"),
			Self::BufferLen { expected, actual } => {
				write!(f, "ncnn buffer length mismatch: expected {expected}, got {actual}")
			}
		}
	}
}

impl std::error::Error for Error {}

fn cstring(name: &str) -> Result<CString, Error> {
	CString::new(name).map_err(|_| Error::NulName(name.to_owned()))
}

/// An owned ncnn network: the model graph plus weights. Create an [`Extractor`]
/// to run inference.
pub struct Net {
	ptr: ncnn_bind::ncnn_net_t,
}

// ncnn documents a net as usable from one thread at a time, so moving the
// handle between threads is sound while sharing it across threads is not.
unsafe impl Send for Net {}

impl Net {
	/// Creates an empty network.
	pub fn new() -> Self {
		Self {
			// SAFETY: ncnn_net_create returns a fresh, valid handle.
			ptr: unsafe { ncnn_bind::ncnn_net_create() },
		}
	}

	/// Enables or disables Vulkan GPU compute. Call before [`Net::load_param`]
	/// so ncnn builds its Vulkan pipelines while loading the graph. ncnn falls
	/// back to CPU automatically when no Vulkan device is present.
	pub fn set_use_vulkan_compute(&mut self, enable: bool) {
		// SAFETY: `ptr` is valid; ncnn_net_set_option copies the option into
		// the net, so destroying the temporary option afterwards is fine.
		unsafe {
			let opt = ncnn_bind::ncnn_option_create();
			ncnn_bind::ncnn_option_set_use_vulkan_compute(opt, i32::from(enable));
			ncnn_bind::ncnn_net_set_option(self.ptr, opt);
			ncnn_bind::ncnn_option_destroy(opt);
		}
	}

	/// Loads the graph definition (a `.param` file).
	pub fn load_param(&mut self, path: &str) -> Result<(), Error> {
		let path_c = cstring(path)?;
		// SAFETY: `ptr` is valid; `path_c` is NUL-terminated and outlives the
		// call.
		let status = unsafe { ncnn_bind::ncnn_net_load_param(self.ptr, path_c.as_ptr()) };
		if status != 0 {
			return Err(Error::LoadParam(path.to_owned()));
		}
		Ok(())
	}

	/// Loads the weights (a `.bin` file).
	pub fn load_model(&mut self, path: &str) -> Result<(), Error> {
		let path_c = cstring(path)?;
		// SAFETY: `ptr` is valid; `path_c` is NUL-terminated and outlives the
		// call.
		let status = unsafe { ncnn_bind::ncnn_net_load_model(self.ptr, path_c.as_ptr()) };
		if status != 0 {
			return Err(Error::LoadModel(path.to_owned()));
		}
		Ok(())
	}

	/// Creates an extractor that inherits the net's options (including the
	/// Vulkan compute setting). The extractor borrows the net and must not
	/// outlive it.
	pub fn create_extractor(&self) -> Extractor<'_> {
		Extractor {
			// SAFETY: `ptr` is valid for the lifetime of `self`.
			ptr: unsafe { ncnn_bind::ncnn_extractor_create(self.ptr) },
			_net: PhantomData,
		}
	}
}

impl Default for Net {
	fn default() -> Self {
		Self::new()
	}
}

impl Drop for Net {
	fn drop(&mut self) {
		// SAFETY: `ptr` came from ncnn_net_create and has not been destroyed.
		unsafe { ncnn_bind::ncnn_net_destroy(self.ptr) };
	}
}

/// Runs inference for a single forward pass over a borrowed [`Net`]. Several
/// output blobs can be extracted from one extractor: the graph is evaluated
/// once and the results cached.
pub struct Extractor<'a> {
	ptr: ncnn_bind::ncnn_extractor_t,
	_net: PhantomData<&'a Net>,
}

impl Extractor<'_> {
	/// Sets an input blob by name. ncnn shares the mat's refcounted data, so
	/// `mat` need not outlive this call.
	pub fn input(&mut self, name: &str, mat: &Mat) -> Result<(), Error> {
		let name_c = cstring(name)?;
		// SAFETY: `ptr` and `mat.ptr` are valid; `name_c` is NUL-terminated.
		let status = unsafe { ncnn_bind::ncnn_extractor_input(self.ptr, name_c.as_ptr(), mat.ptr) };
		if status != 0 {
			return Err(Error::Input(name.to_owned()));
		}
		Ok(())
	}

	/// Runs the forward pass (if needed) and returns the output blob named
	/// `name`. Takes `&mut self`, so several blobs can be read from one pass.
	pub fn extract(&mut self, name: &str) -> Result<Mat, Error> {
		let name_c = cstring(name)?;
		let mut ptr = std::ptr::null_mut();
		// SAFETY: `ptr` (the extractor) is valid; extract assigns a freshly
		// allocated Mat to `ptr` on success (null before).
		let status = unsafe { ncnn_bind::ncnn_extractor_extract(self.ptr, name_c.as_ptr(), &mut ptr) };
		if status != 0 {
			return Err(Error::Extract(name.to_owned()));
		}
		Ok(Mat { ptr })
	}
}

impl Drop for Extractor<'_> {
	fn drop(&mut self) {
		// SAFETY: `ptr` came from ncnn_extractor_create and has not been
		// destroyed.
		unsafe { ncnn_bind::ncnn_extractor_destroy(self.ptr) };
	}
}

/// An owned ncnn matrix (a tensor).
pub struct Mat {
	ptr: ncnn_bind::ncnn_mat_t,
}

impl Mat {
	/// Creates a packed 3D matrix (`channels` x `height` x `width`,
	/// channel-major, as an NCHW blob) and copies `data` into it. `data.len()`
	/// must equal `width * height * channels`.
	pub fn from_float_3d(width: i32, height: i32, channels: i32, data: &[f32]) -> Result<Self, Error> {
		let expected = (width * height * channels) as usize;
		if data.len() != expected {
			return Err(Error::BufferLen {
				expected,
				actual: data.len(),
			});
		}
		// SAFETY: a null allocator selects ncnn's default allocator; the
		// returned mat owns a freshly allocated packed buffer of exactly
		// `expected` f32, so the copy below stays in bounds.
		let ptr = unsafe { ncnn_bind::ncnn_mat_create_3d(width, height, channels, std::ptr::null_mut()) };
		unsafe {
			std::slice::from_raw_parts_mut(ncnn_bind::ncnn_mat_get_data(ptr) as *mut f32, expected)
				.copy_from_slice(data);
		}
		Ok(Self { ptr })
	}

	/// Number of dimensions (1, 2 or 3).
	pub fn dims(&self) -> i32 {
		// SAFETY: `ptr` is a valid Mat handle for the lifetime of `self`.
		unsafe { ncnn_bind::ncnn_mat_get_dims(self.ptr) }
	}

	/// Width (the innermost dimension).
	pub fn w(&self) -> i32 {
		unsafe { ncnn_bind::ncnn_mat_get_w(self.ptr) }
	}

	/// Height.
	pub fn h(&self) -> i32 {
		unsafe { ncnn_bind::ncnn_mat_get_h(self.ptr) }
	}

	/// Channels.
	pub fn c(&self) -> i32 {
		unsafe { ncnn_bind::ncnn_mat_get_c(self.ptr) }
	}

	/// The matrix elements as f32, length `w * h * c`. Valid for packed mats
	/// (elempack 1, `cstep == w * h`) — which is how [`Mat::from_float_3d`]
	/// creates mats and how ncnn returns extracted output blobs.
	pub fn as_f32(&self) -> &[f32] {
		let len = (self.w() * self.h() * self.c()) as usize;
		// SAFETY: a packed mat owns `len` contiguous f32 from its data pointer.
		unsafe { std::slice::from_raw_parts(ncnn_bind::ncnn_mat_get_data(self.ptr) as *const f32, len) }
	}
}

impl Drop for Mat {
	fn drop(&mut self) {
		// SAFETY: destroying a null handle is a no-op; otherwise `ptr` was
		// created or assigned by ncnn and has not been destroyed.
		unsafe { ncnn_bind::ncnn_mat_destroy(self.ptr) };
	}
}
