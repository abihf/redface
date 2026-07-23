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
- `crates/redface-record` — enrollment CLI (`redface-record`), writes `.face` files.
- `crates/redface-check` — CLI client that asks the daemon to authenticate.
- `crates/redface-lock` — Wayland session locker (`ext-session-lock-v1`, works on
  Hyprland). Rendered with wgpu on Vulkan (smithay-client-toolkit for the
  Wayland side; WGSL shaders in `shaders/`, loaded via `include_str!`); Vulkan
  is a hard requirement — there is no CPU fallback, so a working Vulkan
  driver/ICD is needed at runtime. Animations are evaluated in shaders from
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
make build                       # release binaries (pam, daemon, check, record, lock)
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
needs `wayland-client` and a compositor with `ext-session-lock-v1` (Hyprland
has it). `openvino`
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

## Runtime layout (deployed)

- Config: `/etc/redface/config.json` (`device`, `inference_device`, `threshold`,
  `timeout`, `socket`, `pid_file`).
- Enrollments: `/etc/redface/models/<user>.face`; models: `/usr/share/redface/`.
- Socket `/var/run/redface.sock`, pidfile `/var/run/redface.pid`,
  systemd unit `data/redfaced.service`.
- Locker: config `~/.config/redface/lock.json` (per-user, no install step), PAM
  service `/etc/pam.d/redface-lock` (installed from `data/redface-lock.pam`).
