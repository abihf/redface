# AGENTS.md

Face-recognition PAM authentication for Linux (Howdy-style), using an embedded IR
camera (developed on a Dell XPS 9370: 340x340 8-bit grayscale ~60fps). Rust
workspace; the old Go implementation under `cmd/` is archived (`*.go.bak`, not
built — do not treat it as live code).

## Layout

- `crates/redface-core` — descriptor type: 512-dim `f32`, text format is Go-style
  hex floats, one descriptor per line (`*.face` files). Parsing/formatting lives
  here; keep it dependency-free. `cosine_similarity` has an AVX2+FMA fast path
  (`src/simd.rs`, runtime dispatch, scalar fallback).
- `crates/redface-capture` — V4L2 camera capture (GREY preferred, YUYV/RGB3
  fallback), frames delivered as grayscale (1 byte per pixel). GREY is
  passed through; YUYV extracts luma; RGB3 converts via BT.601.
- `crates/redface-recognition` — the inference stack: CLAHE preprocessing,
  SCRFD detection, ArcFace alignment/encoding. Accepts grayscale frames,
  replicates to 3 channels internally for the models. Two inference backends,
  selected by cargo feature: ncnn on the Vulkan GPU (default, with automatic
  CPU fallback; via the safe `ncnn` crate) or OpenVINO (opt-in
  `openvino` feature); pixel processing (CLAHE, resize, warp) goes through
  OpenCV. No ncnn FFI `unsafe` lives here.
- `crates/ncnn` — safe wrapper over `ncnn-bind` (RAII `Net`/`Extractor`/`Mat`
  handles that destroy their ncnn objects on drop); all the ncnn FFI `unsafe`
  lives here.
- `crates/ncnn-bind` — vendored FFI bindings to ncnn's C API from
  `tpoisonooo/rust-ncnn` (Apache-2.0), with the build script rewritten to
  bindgen against the system ncnn (upstream pins bindgen 0.59, which cannot
  parse current glibc headers). Linked via `NCNN_DIR` (default `/usr`).
- `crates/redface-runtime` — config (`/etc/redface/config.json`), the unix-socket
  protocol, and the `verify()` loop shared by daemon and tools.
- `crates/redface-toolkit` — shared Wayland/GLES UI toolkit, built as a dylib
  (`libredface_toolkit.so`, `crate-type = ["dylib", "rlib"]`): wraps EGL +
  glow (OpenGL ES 3.0 renderer; GLSL ES shaders in `shaders/`, compiled at
  runtime by the driver) and smithay-client-toolkit (ext-session-lock-v1 and
  wlr-layer-shell event loop) behind a small app-facing API (implement the
  `App` trait, call `run()` with a `Role`). Also owns the scene data types
  (`scene.rs`) and text rasterization with ab_glyph/fontdb into a glyph atlas
  (`text.rs`). EGL surfaces are created from pre-existing `wl_surface` objects
  via `libwayland-egl` (`wl_egl_window_create`); the GL context is created once
  at startup on a 1×1 pbuffer and shared across all output surfaces. No Vulkan
  dependency — the renderer needs only `libEGL` and `libwayland-egl` (both
  provided by Mesa).
- `crates/redface-osd` — layer-shell feedback window (via redface-toolkit);
  for now an animated face indicator with a cancel action.
- `crates/redface-record` — enrollment CLI (`redface-record`), writes `.face` files.
- `crates/redface-check` — CLI client that asks the daemon to authenticate.
- `crates/redface-lock` — Wayland session locker (`ext-session-lock-v1`, works on
  Hyprland). Only the lock-specific parts live here: the PAM/face auth state
  machine (`src/auth.rs`, `src/wayland.rs` as a thin redface-toolkit `App`) and
  the UI layout/scene builder (`src/ui.rs`); rendering and the Wayland event
  loop come from redface-toolkit, which the binary links as
  `libredface_toolkit.so` (`make install-lock` installs it to `$(LIBDIR)`,
  `/usr/lib` by default). Animations are evaluated in shaders from
  time uniforms; text is rasterized with ab_glyph into a glyph atlas. Passwords
  go through the `redface-lock` PAM service
  (minimal client-side FFI in `src/auth.rs`; pam-client/pam-sys bindgen against
  libclang, which conflicts with this workspace's clang `runtime` feature). Face
  unlock talks to redfaced over the socket (toggling it off drops the connection,
  which the daemon treats as cancel). `--test` covers all outputs with
  wlr-layer-shell overlay surfaces (no session lock; Esc exits) so the UI can
  be tried without valid credentials. Config: `~/.config/redface/lock.json`
  (`background`, `background_image`, `primary_output`, colors; all optional).
  Only the primary output shows the UI; other outputs draw the background.
  Animations run off `wl_surface.frame` callbacks at native refresh.
- `crates/redfaced` — daemon: owns the camera, serves auth requests on
  `/var/run/redface.sock`.
- `crates/pam-redface` — PAM module (`libpam_redface.so`), talks to the daemon.
- `data/` — models + systemd unit. ONNX models are untracked; fetch with
  `make fetch-data` (InsightFace `buffalo_l` pack, non-commercial license),
  then `make convert-models` (pnnx via pipx) produces the ncnn `.param`/`.bin`
  the default backend reads.

## Build & test

```sh
cargo build --workspace          # debug (ncnn Vulkan GPU backend, no OpenVINO)
make build                       # release binaries (pam, daemon, check, record, lock, osd) + libredface_toolkit.so
cargo test --workspace           # full test suite
make fetch-data                  # download ONNX models into data/
make convert-models              # convert to ncnn .param/.bin via pnnx (smoke test)
cargo build --workspace --no-default-features --features openvino   # opt into OpenVINO (workspace-wide)
cargo test --workspace --no-default-features --features openvino    # full test suite with OpenVINO
```

`make build` produces ncnn-backend binaries; OpenVINO release binaries need
per-package cargo builds, e.g. `cargo build --release -p redfaced
--no-default-features --features openvino` (`make build ENABLE_OPENVINO=1`
passes the same flags). The `--no-default-features` matters: it drops the
default `ncnn` feature, so an OpenVINO build neither compiles the `ncnn` crate
nor links `libncnn`.

System dependencies: `opencv` + `clang` (the `opencv` crate generates
bindings with libclang at build time; `.cargo/config.toml` sets
`OPENCV_PKGCONFIG_NAME=opencv5` because the system pkg-config file is
`opencv5.pc`), `ncnn` library + headers (the vendored `ncnn-bind` crate
bindgen-generates against `$NCNN_DIR/include/ncnn` and links `libncnn`;
`.cargo/config.toml` sets `NCNN_DIR=/usr`). GPU inference needs `libncnn`
built with Vulkan plus a Vulkan driver/ICD for the GPU (ncnn falls back to
CPU if none is present). Also v4l2 and PAM headers. The locker additionally
needs `wayland-client`, `libEGL`, `libwayland-egl`, and a compositor with
`ext-session-lock-v1` (Hyprland has it). `openvino`
is only needed when opting into the `openvino` cargo feature (the `openvino`
crate links `libopenvino_c`; a default build has zero OpenVINO dependency),
plus `openvino-intel-npu-plugin` + `intel-npu-driver` for NPU on that build.
Install/packaging goes through the `Makefile` (`make install DESTDIR=...`);
there is no distro package in the repo anymore.

The release profile uses fat LTO (`lto = true`, `codegen-units = 1`) and
`.cargo/config.toml` sets `-C target-cpu=native`, so builds are tuned for the
local machine and are not portable across CPUs.

Inference smoke test (loads real models, runs one pass):

```sh
cargo run -p redface-recognition --example smoke_test   # ncnn Vulkan GPU backend (DEVICE=CPU forces CPU)
cargo run -p redface-recognition --no-default-features --features openvino --example smoke_test   # OpenVINO, NPU default
DEVICE=CPU cargo run -p redface-recognition --no-default-features --features openvino --example smoke_test   # OpenVINO, CPU forced
```

Run it (plus `cargo test --workspace`) after any change to
`crates/redface-recognition`.

## Inference conventions (redface-recognition)

- Runtime: two backends, selected by cargo feature (not at runtime).
  Default: ncnn on the Vulkan GPU, driven through the safe `crates/ncnn`
  wrapper over the vendored `ncnn-bind` FFI (the raw ncnn C API, not the
  `ncnn-rs` wrappers — the wrapper's `Extractor::extract` consumes the
  extractor, so one forward pass couldn't yield all nine SCRFD outputs). `DevicePref` selects the ncnn target:
  `Npu`/`Auto` set `use_vulkan_compute` (set on the net's `Option` *before*
  `load_param` so ncnn builds its Vulkan pipelines during load); ncnn falls
  back to CPU automatically when no Vulkan device is present — the C API
  exposes no GPU-count query, so there is no explicit fallback warning, but
  ncnn prints its Vulkan device on load. `Cpu` forces CPU. One stderr line at
  startup notes the choice. ncnn reads `.param`/`.bin` models converted from
  the ONNX (`make convert-models`). Opt-in `openvino` feature: `openvino`
  crate 0.11 (openvino-rs); device selection via `DevicePref`: `Npu` (default)
  compiles for OpenVINO `"NPU"` and falls back to `"CPU"` with a stderr
  warning; `Cpu` forces CPU; `Auto` is a config-compat alias of `Npu` — do NOT
  reintroduce OpenVINO's `AUTO:NPU,CPU` meta-plugin (a broken NPU plugin
  install segfaults inside it, defeating the fallback). Enroll and verify with
  the same backend so descriptors stay consistent.
- Detector: `det_10g.onnx` (SCRFD-10G). Outputs are 9 tensors in stride-major
  order (scores/bboxes/kps for strides 8/16/32), **2 anchors per feature point,
  adjacent per point** — entry `i` belongs to feature point `i / 2`. The ONNX
  output names are numeric graph ids (`448`, `471`, ...), not `score_8`-style
  names, so neither backend relies on them: the ncnn path reads the output
  blob names from the `.param` graph (blobs produced but never consumed) and —
  like the old DNN path before it — maps outputs into the decode order
  [score8,score16,score32,bbox8,bbox16,bbox32,kps8,kps16,kps32] by output
  shape: entries = 2·(640/stride)² identifies the stride, width ∈ {1,4,10}
  identifies score/bbox/kps. Getting this wrong was a real bug once; do not
  regress it.
  The score pre-filter in `decode_detections` has an AVX2 fast path
  (`src/simd.rs`, runtime dispatch, scalar fallback).
- Encoder: `w600k_r50.onnx` (ArcFace R50), 112x112 aligned crop via Umeyama
  similarity transform on the 5 landmarks, BGR, `(x-127.5)/127.5`.
- On OpenVINO both models are reshaped to static input shapes at load (an
  NPU-plugin requirement); the ncnn path runs the converted `.param`/`.bin`,
  which are fixed at the conversion input size. SCRFD input is
  `(x-127.5)/128` at 640x640.
- Pixel processing uses the `opencv` crate (0.99, features `imgproc` + `dnn` +
  `clang-runtime`): CLAHE via `createCLAHE`, detector input via
  `dnn::blob_from_image` (INTER_LINEAR resize — the InsightFace reference
  preprocessing), alignment via `warpAffine` (BORDER_REPLICATE) +
  `blob_from_image`. The `dnn` feature is now used only for `blob_from_image`
  preprocessing — inference itself moved to ncnn. `clang-runtime` is required:
  v4l2-sys-mit's bindgen enables clang-sys's `runtime` feature workspace-wide,
  which breaks the opencv build script without it.
- `recognize()` applies CLAHE (8x8 tiles, clip 2.0) to the grayscale plane
  before detection *and* encoding — required on these IR cameras
  (Howdy/Visage recipe). It shifts descriptors: enrollment and verification
  must both run the same preprocessing.
- Auth: cosine similarity against enrolled descriptors, threshold from config
  (default 0.9, higher = stricter).

## Code conventions

- Rust edition 2024. Keep changes minimal and match surrounding style; the
  codebase favors small free functions, explicit error enums with
  `fmt::Display` + `std::error::Error`, and `#[cfg(test)] mod tests` per file.
- Tests must be self-contained: no fixture files, no camera, no models, no
  network. (Descriptor tests generate their data deterministically in code.)
  Tests that need the real models live in `examples/`, not `tests/`.
- No new crates without checking the manifest first; the workspace is
  deliberately lean. OpenCV is the sanctioned image-processing dependency —
  route pixel work through it instead of hand-rolling loops. `crates/ncnn` is
  the only place ncnn FFI `unsafe` may live (dependents use its RAII handles).
  `crates/ncnn-bind` is a vendored copy of rust-ncnn's FFI (Apache-2.0), not a
  new invention — upstream's bindgen 0.59 cannot parse current glibc headers,
  so its build script was rewritten; keep it a thin bindings crate.
- No NIR/IR-trained face models exist publicly; don't add model downloads
  without documenting license and source in `README.md` + `Makefile fetch-data`.
- The UI toolkit uses GLES (EGL + glow), not Vulkan. GLSL ES 3.00 shaders are
  compiled by the GPU driver at runtime. `crates/ncnn` is the only place
  ncnn FFI `unsafe` may live; the EGL FFI in `redface-toolkit` is also `unsafe`
  but self-contained (~10 functions).

## Performance & memory

The two hot paths are the **recognition pipeline** (60 fps camera frames →
detection → encoding) and the **UI render loop** (frame-callback-paced GL
draws at native refresh). Both have been audited end-to-end; regressions in
allocation volume or copy bandwidth must be avoided.

### Recognition (redface-recognition, redface-runtime, redface-capture, ncnn)

**Reuse, don't re-allocate.** The `Recognizer` struct holds reusable `Mat`
buffers for every intermediate pixel stage in the hot path:

- `equalized`, `rgb`, `crop` — reused `CV_8UC*` Mats for CLAHE output,
  gray→RGB expansion, and warp-affine output.
- `resized`, `float_hwc` — reused `CV_8UC3` and `CV_32FC3` intermediates for
  the detector resize+normalize step.
- `blob_chw` — reused `CV_32F` Mat for the HWC→CHW transpose (the final NCHW
  blob fed to the inference backend).

The old `dnn::blob_from_image` path allocated a fresh ~4.9 MB f32 Mat every
frame for the detector (plus 150 KB for the encoder).  The replacement is:

1. `imgproc::resize(rgb → resized, target_size, INTER_LINEAR)`
2. `resized.convert_to(float_hwc, CV_32F, alpha, beta)` — normalizes in one
   pass without an intermediate u8→f32 copy.
3. `float_hwc.reshape(1, H*W)` (zero-copy header) → `core::transpose` into
   `blob_chw` — the NCHW buffer, reused across frames.

The final f32 data is passed to `ModelRunner::infer(shape: [i32; 4], data:
&[f32])` — the old `infer(&Mat)` signature was replaced so input preparation
lives in `Recognizer` (which owns the reusable buffers) and inference only
sees a plain `&[f32]`.

**ncnn zero-copy input.** `ncnn::Mat::from_external_float_3d` wraps the
caller's f32 buffer (e.g. `blob_chw`'s data) without a memcpy, avoiding the
old 4.9 MB copy into a freshly allocated ncnn Mat. The external Mat must
outlive the last `Extractor::extract` call on that forward pass (the data
lives in `blob_chw` which outlives the `infer` call, so this is trivially
satisfied).

**Frame buffer recycling.** `redface-capture` recycles the `Vec<u8>` frame
buffer between the consumer thread and the producer thread via a
`FrameSlot::recycle` field. The GREY pass-through path does
`recycle_buffer.copy_from_slice(raw)` instead of `raw.to_vec()`, reusing the
allocation frame-over-frame. The `stream()` callback now takes `&Frame`
rather than `Frame` so the buffer stays owned by the capture loop for
recycling.

**Decode Vec reuse.** `Recognizer` owns `detection_indices: Vec<u32>`,
`detection_candidates: Vec<(Detection, f32)>`, `detection_kept`, and
`detection_results` — cleared per frame instead of allocated fresh.
`decode_detections` and `nms_reuse` take `&mut Vec` parameters directly.

### UI toolkit (redface-toolkit, redface-lock, redface-osd)

**Cache GL uniform locations.** The old `render()` called
`gl.get_uniform_location()` every frame for the bg and text passes (5 calls
per frame — driver hash-table lookups). All uniform locations are now
queried once at program-link time and stored in `Gpu` fields (same pattern
the shape pass already used).

**Avoid redundant scene rebuilds.** `build_scene()` allocates a fresh `Scene`
with new `Vec`s for shapes and texts. This only happens on state changes
(input, clock minute, resize, notification), not on pure animation frames
(where the cached `Scene` is reused). When changing hover state, only
trigger a redraw if the hit-test result actually changed (e.g. pointer
motion within the same region should not rebuild the scene).

**Reuse small Vecs in the UI.** `dot_births()` in redface-lock previously
allocated a new `Vec<f32>` each call — replaced with a reusable
`dot_births_cache` field on `UiState`, cleared and refilled. The lock's
`build_scene` takes `&mut UiState` to support this.

**Don't cache what the compositor owns.** `redface-osd`'s `Layout` is
pre-computed once (the surface size is fixed) and stored as a field rather
than recomputed on every pointer event. Layouts in `redface-lock` are
recomputed per `build_scene` because the surface can resize.

### Things to check when making changes

- Every `Vec::new()` / `collect()` / `to_owned()` / `clone()` in a function
  that runs per frame or per input event is suspect — can it be a reusable
  field?
- Every `dnn::blob_from_image` call in the recognition pipeline — can it be
  replaced with the resize + convert_to + reshape + transpose pattern?
- Every `ncnn::Mat::from_float_3d` call — can it use
  `from_external_float_3d` instead?
- Every `gl.get_uniform_location` in `render()` — cache it in `Gpu`.
- Every `build_scene` call path — is the scene really dirty, or is it a
  no-op state change (hover in same region, key press that doesn't change
  visible state)?

## Runtime layout (deployed)

- Config: `/etc/redface/config.json` (`device`, `inference_device`, `threshold`,
  `timeout`, `socket`, `pid_file`).
- Enrollments: `/etc/redface/models/<user>.face`; models: `/usr/share/redface/`.
- Socket `/var/run/redface.sock`, pidfile `/var/run/redface.pid`,
  systemd unit `data/redfaced.service`.
- Locker: config `~/.config/redface/lock.json` (per-user, no install step), PAM
  service `/etc/pam.d/redface-lock` (installed from `data/redface-lock.pam`).
