//! Vulkan renderer for the lock screen. Mirrors the CPU renderer in `ui.rs`
//! visually: same layout inputs come in as a `Scene` plus `Uniforms`, and all
//! animations (dot pop, shake, face crossfade, pulse ring) run in the shape
//! vertex shader off `Uniforms::time`.

use std::borrow::Cow;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;

use raw_window_handle::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};
use wayland_client::Proxy;
use wgpu::util::DeviceExt;

use crate::scene::{ATLAS_SIZE, AtlasRect, Scene, ShapeInstance, TextInstance, Uniforms};

const SHAPE_ATTRS: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
	0 => Float32x2,
	1 => Float32x2,
	2 => Float32x4,
	3 => Float32,
	4 => Float32,
	5 => Float32,
	6 => Uint32
];

const TEXT_ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
	0 => Float32x2,
	1 => Float32x2,
	2 => Float32x2,
	3 => Float32x2,
	4 => Float32x4
];

/// Anything that prevents the Vulkan renderer from coming up. There is no
/// fallback: the caller is expected to abort.
#[derive(Debug)]
pub struct GpuError(pub String);

impl fmt::Display for GpuError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "gpu error: {}", self.0)
	}
}

impl std::error::Error for GpuError {}

/// Shared GPU state: device, pipelines' layouts and the textures/bind groups
/// used by every surface (background image, glyph atlas, uniforms).
pub struct Gpu {
	instance: wgpu::Instance,
	adapter: wgpu::Adapter,
	device: wgpu::Device,
	queue: wgpu::Queue,
	uniform_buffer: wgpu::Buffer,
	sampler: wgpu::Sampler,
	atlas: wgpu::Texture,
	bg_texture: Option<wgpu::Texture>,
	bg_view: Option<wgpu::TextureView>,
	bg_size: [f32; 2],
	placeholder_view: wgpu::TextureView,
	bg_layout: wgpu::BindGroupLayout,
	shape_layout: wgpu::BindGroupLayout,
	text_layout: wgpu::BindGroupLayout,
	bg_bind_group: wgpu::BindGroup,
	shape_bind_group: wgpu::BindGroup,
	text_bind_group: wgpu::BindGroup,
	pipeline_cache: Option<(wgpu::PipelineCache, std::path::PathBuf)>,
}

impl Gpu {
	/// Creates a Vulkan-only instance and requests an adapter and device. Any
	/// failure is fatal to the caller (`GpuError`), there is no fallback.
	pub fn new() -> Result<Gpu, GpuError> {
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::VULKAN,
			flags: wgpu::InstanceFlags::default(),
			memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
			backend_options: wgpu::BackendOptions::default(),
			display: None,
		});
		let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::default(),
			compatible_surface: None,
			force_fallback_adapter: false,
			apply_limit_buckets: false,
		}))
		.map_err(|err| GpuError(format!("no Vulkan adapter available: {err}")))?;
		let required_features = wgpu::Features::default()
			| if adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
				wgpu::Features::PIPELINE_CACHE
			} else {
				wgpu::Features::empty()
			};
		let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
			required_features,
			..Default::default()
		}))
		.map_err(|err| GpuError(format!("failed to request device: {err}")))?;

		// Pipeline cache persisted between runs; shortens pipeline creation
		// when the driver's own cache is cold.
		let pipeline_cache = wgpu::util::pipeline_cache_key(&adapter.get_info()).and_then(|key| {
			let dir = std::env::var_os("XDG_CACHE_HOME")
				.map(std::path::PathBuf::from)
				.or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".cache")))?
				.join("redface-lock");
			let path = dir.join(key);
			let data = std::fs::read(&path).ok();
			// SAFETY: the data comes from our own cache file; wgpu validates it
			// and falls back to an empty cache on mismatch.
			let cache = unsafe {
				device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
					label: Some("redface-lock"),
					data: data.as_deref(),
					fallback: true,
				})
			};
			Some((cache, path))
		});

		let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("uniforms"),
			size: std::mem::size_of::<Uniforms>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: Some("linear"),
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Linear,
			..Default::default()
		});

		let atlas = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("glyph atlas"),
			size: wgpu::Extent3d {
				width: ATLAS_SIZE,
				height: ATLAS_SIZE,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::R8Unorm,
			usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
			view_formats: &[],
		});
		let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());

		// Stand-in bound while there is no background image; never sampled
		// because the shader takes the solid-color branch then.
		let placeholder = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("bg placeholder"),
			size: wgpu::Extent3d {
				width: 1,
				height: 1,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba8Unorm,
			usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
			view_formats: &[],
		});
		let placeholder_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());

		let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
			binding,
			visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
			ty: wgpu::BindingType::Buffer {
				ty: wgpu::BufferBindingType::Uniform,
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		};
		let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
			binding,
			visibility: wgpu::ShaderStages::FRAGMENT,
			ty: wgpu::BindingType::Texture {
				sample_type: wgpu::TextureSampleType::Float { filterable: true },
				view_dimension: wgpu::TextureViewDimension::D2,
				multisampled: false,
			},
			count: None,
		};
		let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
			binding,
			visibility: wgpu::ShaderStages::FRAGMENT,
			ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
			count: None,
		};

		let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("bg layout"),
			entries: &[uniform_entry(0), texture_entry(1), sampler_entry(2)],
		});
		let shape_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("shape layout"),
			entries: &[uniform_entry(0)],
		});
		let text_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("text layout"),
			entries: &[uniform_entry(0), texture_entry(1), sampler_entry(2)],
		});

		let bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("bg bind group"),
			layout: &bg_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: uniform_buffer.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(&placeholder_view),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::Sampler(&sampler),
				},
			],
		});
		let shape_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("shape bind group"),
			layout: &shape_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: uniform_buffer.as_entire_binding(),
			}],
		});
		let text_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("text bind group"),
			layout: &text_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: uniform_buffer.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(&atlas_view),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::Sampler(&sampler),
				},
			],
		});

		Ok(Gpu {
			instance,
			adapter,
			device,
			queue,
			uniform_buffer,
			sampler,
			atlas,
			bg_texture: None,
			bg_view: None,
			bg_size: [0.0, 0.0],
			placeholder_view,
			bg_layout,
			shape_layout,
			text_layout,
			bg_bind_group,
			shape_bind_group,
			text_bind_group,
			pipeline_cache,
		})
	}

	/// Uploads a straight-alpha RGBA8 background image, or switches back to
	/// the solid `bg_color` when called with `None`.
	pub fn set_background(&mut self, rgba: Option<(&[u8], u32, u32)>) {
		match rgba {
			Some((data, width, height)) if width > 0 && height > 0 => {
				let texture = self.device.create_texture(&wgpu::TextureDescriptor {
					label: Some("background"),
					size: wgpu::Extent3d {
						width,
						height,
						depth_or_array_layers: 1,
					},
					mip_level_count: 1,
					sample_count: 1,
					dimension: wgpu::TextureDimension::D2,
					format: wgpu::TextureFormat::Rgba8Unorm,
					usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
					view_formats: &[],
				});
				self.queue.write_texture(
					wgpu::TexelCopyTextureInfo {
						texture: &texture,
						mip_level: 0,
						origin: wgpu::Origin3d::ZERO,
						aspect: wgpu::TextureAspect::All,
					},
					data,
					wgpu::TexelCopyBufferLayout {
						offset: 0,
						bytes_per_row: Some(width * 4),
						rows_per_image: Some(height),
					},
					wgpu::Extent3d {
						width,
						height,
						depth_or_array_layers: 1,
					},
				);
				self.bg_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
				self.bg_texture = Some(texture);
				self.bg_size = [width as f32, height as f32];
			}
			_ => {
				self.bg_texture = None;
				self.bg_view = None;
				self.bg_size = [0.0, 0.0];
			}
		}
		let view = self.bg_view.as_ref().unwrap_or(&self.placeholder_view);
		self.bg_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("bg bind group"),
			layout: &self.bg_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: self.uniform_buffer.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(view),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::Sampler(&self.sampler),
				},
			],
		});
	}

	/// Background image size in px, or [0, 0] when rendering a solid color.
	pub fn background_size(&self) -> [f32; 2] {
		self.bg_size
	}

	/// Writes glyph coverage bitmaps into the shared R8 atlas texture.
	pub fn upload_glyphs(&self, writes: &[(AtlasRect, Vec<u8>)]) {
		for (rect, data) in writes {
			if rect.w == 0 || rect.h == 0 {
				continue;
			}
			debug_assert_eq!(data.len(), (rect.w * rect.h) as usize);
			self.queue.write_texture(
				wgpu::TexelCopyTextureInfo {
					texture: &self.atlas,
					mip_level: 0,
					origin: wgpu::Origin3d {
						x: rect.x,
						y: rect.y,
						z: 0,
					},
					aspect: wgpu::TextureAspect::All,
				},
				data,
				wgpu::TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(rect.w),
					rows_per_image: Some(rect.h),
				},
				wgpu::Extent3d {
					width: rect.w,
					height: rect.h,
					depth_or_array_layers: 1,
				},
			);
		}
	}

	/// Creates a swapchain surface for a Wayland surface.
	pub fn create_surface(
		&self,
		conn: &wayland_client::Connection,
		surface: &wayland_client::protocol::wl_surface::WlSurface,
		width: u32,
		height: u32,
	) -> Result<GpuSurface, GpuError> {
		let display_ptr = conn.backend().display_id().as_ptr() as *mut c_void;
		let window_ptr = surface.id().as_ptr() as *mut c_void;
		let display = WaylandDisplayHandle::new(
			NonNull::new(display_ptr).ok_or_else(|| GpuError("null wl_display pointer".to_owned()))?,
		);
		let window = WaylandWindowHandle::new(
			NonNull::new(window_ptr).ok_or_else(|| GpuError("null wl_surface pointer".to_owned()))?,
		);
		// SAFETY: both pointers come from a live wayland-client connection and
		// proxy. The caller keeps the connection and the wl_surface alive for
		// at least as long as the returned GpuSurface, and the instance was
		// restricted to the Vulkan backend, matching the raw handle types.
		let gpu_surface = unsafe {
			self.instance
				.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
					raw_display_handle: Some(RawDisplayHandle::Wayland(display)),
					raw_window_handle: RawWindowHandle::Wayland(window),
				})
		}
		.map_err(|err| GpuError(format!("failed to create surface: {err}")))?;

		let caps = gpu_surface.get_capabilities(&self.adapter);
		// Prefer a non-sRGB format: shader outputs (uniform colors, unorm
		// texture samples) then map 1:1 to the framebuffer, matching the old
		// CPU rendering exactly. Fall back to whatever the surface offers.
		let preferred = [
			wgpu::TextureFormat::Bgra8Unorm,
			wgpu::TextureFormat::Rgba8Unorm,
			wgpu::TextureFormat::Bgra8UnormSrgb,
			wgpu::TextureFormat::Rgba8UnormSrgb,
		];
		let format = preferred
			.iter()
			.copied()
			.find(|f| caps.formats.contains(f))
			.unwrap_or(caps.formats[0]);
		let config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format,
			color_space: wgpu::SurfaceColorSpace::Auto,
			width: width.max(1),
			height: height.max(1),
			present_mode: wgpu::PresentMode::Fifo,
			desired_maximum_frame_latency: 2,
			alpha_mode: caps.alpha_modes[0],
			view_formats: vec![],
		};
		gpu_surface.configure(&self.device, &config);

		// Pipelines are compiled against the surface format, so they are
		// per-surface; layouts, textures and bind groups stay shared.
		let (bg_pipeline, shape_pipeline, text_pipeline) = create_pipelines(&self.device, format, self);
		if let Some((cache, path)) = &self.pipeline_cache
			&& let Some(data) = cache.get_data()
			&& let Some(parent) = path.parent()
		{
			// Write atomically: tmp file + rename.
			let tmp = path.with_extension("tmp");
			if std::fs::create_dir_all(parent).is_ok() && std::fs::write(&tmp, &data).is_ok() {
				let _ = std::fs::rename(&tmp, path);
			}
		}

		Ok(GpuSurface {
			surface: gpu_surface,
			config,
			bg_pipeline,
			shape_pipeline,
			text_pipeline,
			shape_buf: None,
			shape_cap: 0,
			text_buf: None,
			text_cap: 0,
		})
	}
}

fn create_pipelines(
	device: &wgpu::Device,
	format: wgpu::TextureFormat,
	gpu: &Gpu,
) -> (wgpu::RenderPipeline, wgpu::RenderPipeline, wgpu::RenderPipeline) {
	let module = |label: &str, src: &'static str| {
		device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some(label),
			source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
		})
	};
	let bg_module = module("bg shader", include_str!("../shaders/bg.wgsl"));
	let shape_module = module("shape shader", include_str!("../shaders/shape.wgsl"));
	let text_module = module("text shader", include_str!("../shaders/text.wgsl"));

	let pipeline = |layout: &wgpu::BindGroupLayout,
	                module: &wgpu::ShaderModule,
	                buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
	                blend: wgpu::BlendState| {
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: None,
			bind_group_layouts: &[Some(layout)],
			immediate_size: 0,
		});
		device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: None,
			layout: Some(&pipeline_layout),
			vertex: wgpu::VertexState {
				module,
				entry_point: Some("vs_main"),
				compilation_options: wgpu::PipelineCompilationOptions::default(),
				buffers,
			},
			fragment: Some(wgpu::FragmentState {
				module,
				entry_point: Some("fs_main"),
				compilation_options: wgpu::PipelineCompilationOptions::default(),
				targets: &[Some(wgpu::ColorTargetState {
					format,
					blend: Some(blend),
					write_mask: wgpu::ColorWrites::ALL,
				})],
			}),
			primitive: wgpu::PrimitiveState::default(),
			depth_stencil: None,
			multisample: wgpu::MultisampleState::default(),
			multiview_mask: None,
			cache: gpu.pipeline_cache.as_ref().map(|(cache, _)| cache),
		})
	};

	let shape_vertex = wgpu::VertexBufferLayout {
		array_stride: std::mem::size_of::<ShapeInstance>() as u64,
		step_mode: wgpu::VertexStepMode::Instance,
		attributes: &SHAPE_ATTRS,
	};
	let text_vertex = wgpu::VertexBufferLayout {
		array_stride: std::mem::size_of::<TextInstance>() as u64,
		step_mode: wgpu::VertexStepMode::Instance,
		attributes: &TEXT_ATTRS,
	};

	// All shaders output premultiplied alpha (like the CPU renderer's
	// blend_pixel), so every translucent pass uses premultiplied blending.
	let bg_pipeline = pipeline(&gpu.bg_layout, &bg_module, &[], wgpu::BlendState::REPLACE);
	let shape_pipeline = pipeline(
		&gpu.shape_layout,
		&shape_module,
		&[Some(shape_vertex)],
		wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
	);
	let text_pipeline = pipeline(
		&gpu.text_layout,
		&text_module,
		&[Some(text_vertex)],
		wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
	);
	(bg_pipeline, shape_pipeline, text_pipeline)
}

/// A per-output swapchain surface plus its pipelines and instance buffers.
pub struct GpuSurface {
	surface: wgpu::Surface<'static>,
	config: wgpu::SurfaceConfiguration,
	bg_pipeline: wgpu::RenderPipeline,
	shape_pipeline: wgpu::RenderPipeline,
	text_pipeline: wgpu::RenderPipeline,
	shape_buf: Option<wgpu::Buffer>,
	shape_cap: usize,
	text_buf: Option<wgpu::Buffer>,
	text_cap: usize,
}

/// Grows `*buf` to hold at least `needed` instances, allocating on first use.
fn ensure_instance_buffer(
	device: &wgpu::Device,
	buf: &mut Option<wgpu::Buffer>,
	cap: &mut usize,
	needed: usize,
	elem_size: usize,
	label: &str,
) {
	if buf.is_some() && needed <= *cap {
		return;
	}
	let new_cap = needed.max(64).next_power_of_two();
	*buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some(label),
		contents: &vec![0u8; new_cap * elem_size],
		usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
	}));
	*cap = new_cap;
}

impl GpuSurface {
	/// Reconfigures the swapchain after an output resize.
	pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
		if width == 0 || height == 0 {
			return;
		}
		self.config.width = width;
		self.config.height = height;
		self.surface.configure(&gpu.device, &self.config);
	}

	/// Renders one frame: background, then shapes, then text, in a single
	/// render pass. Swapchain errors are recovered from by reconfiguring once;
	/// anything else drops the frame.
	pub fn render(&mut self, gpu: &Gpu, scene: &Scene, uniforms: &Uniforms) {
		ensure_instance_buffer(
			&gpu.device,
			&mut self.shape_buf,
			&mut self.shape_cap,
			scene.shapes.len(),
			std::mem::size_of::<ShapeInstance>(),
			"shape instances",
		);
		ensure_instance_buffer(
			&gpu.device,
			&mut self.text_buf,
			&mut self.text_cap,
			scene.texts.len(),
			std::mem::size_of::<TextInstance>(),
			"text instances",
		);
		if !scene.shapes.is_empty() {
			gpu.queue
				.write_buffer(self.shape_buf.as_ref().unwrap(), 0, bytemuck::cast_slice(&scene.shapes));
		}
		if !scene.texts.is_empty() {
			gpu.queue
				.write_buffer(self.text_buf.as_ref().unwrap(), 0, bytemuck::cast_slice(&scene.texts));
		}
		gpu.queue
			.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

		use wgpu::CurrentSurfaceTexture as Cst;
		let frame = match self.surface.get_current_texture() {
			Cst::Success(frame) | Cst::Suboptimal(frame) => frame,
			Cst::Lost | Cst::Outdated => {
				self.surface.configure(&gpu.device, &self.config);
				match self.surface.get_current_texture() {
					Cst::Success(frame) | Cst::Suboptimal(frame) => frame,
					_ => return,
				}
			}
			_ => return,
		};
		let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
		let mut encoder = gpu
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("lock screen"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
						store: wgpu::StoreOp::Store,
					},
					depth_slice: None,
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
				multiview_mask: None,
			});
			pass.set_pipeline(&self.bg_pipeline);
			pass.set_bind_group(0, &gpu.bg_bind_group, &[]);
			pass.draw(0..3, 0..1);
			if !scene.shapes.is_empty() {
				pass.set_pipeline(&self.shape_pipeline);
				pass.set_bind_group(0, &gpu.shape_bind_group, &[]);
				pass.set_vertex_buffer(0, self.shape_buf.as_ref().unwrap().slice(..));
				pass.draw(0..6, 0..scene.shapes.len() as u32);
			}
			if !scene.texts.is_empty() {
				pass.set_pipeline(&self.text_pipeline);
				pass.set_bind_group(0, &gpu.text_bind_group, &[]);
				pass.set_vertex_buffer(0, self.text_buf.as_ref().unwrap().slice(..));
				pass.draw(0..6, 0..scene.texts.len() as u32);
			}
		}
		gpu.queue.submit([encoder.finish()]);
		gpu.queue.present(frame);
	}
}
