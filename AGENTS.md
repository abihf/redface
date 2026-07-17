# AGENTS.md

Face-recognition PAM authentication for Linux (Howdy-style), using an embedded IR
camera (developed on a Dell XPS 9370: 340x340 8-bit grayscale ~60fps). Rust
workspace; the old Go implementation under `cmd/` is archived (`*.go.bak`, not
built — do not treat it as live code).

## Layout

- `crates/redface-core` — descriptor type: 512-dim `f32`, text format is Go-style
  hex floats, one descriptor per line (`*.face` files). Parsing/formatting lives
  here; keep it dependency-free.
- `crates/redface-capture` — V4L2 camera capture (GREY preferred, YUYV fallback),
  frames delivered as RGB24 with gray replicated across channels.
- `crates/redface-recognition` — the inference stack: CLAHE preprocessing,
  SCRFD detection, ArcFace alignment/encoding. Owns all OpenVINO interaction.
- `crates/redface-runtime` — config (`/etc/redface/config.json`), the unix-socket
  protocol, and the `verify()` loop shared by daemon and tools.
- `crates/redface-record` — enrollment CLI (`redface-record`), writes `.face` files.
- `crates/redface-check` — CLI client that asks the daemon to authenticate.
- `crates/redfaced` — daemon: owns the camera, serves auth requests on
  `/var/run/redface.sock`.
- `crates/pam-redface` — PAM module (`libpam_redface.so`), talks to the daemon.
- `data/` — models + systemd unit. ONNX models are untracked; fetch with
  `make fetch-data` (InsightFace `buffalo_l` pack, non-commercial license).

## Build & test

```sh
cargo build --workspace          # debug
make build                       # release binaries (pam, daemon, check, record)
cargo test --workspace           # full test suite
make fetch-data                  # download models into data/ (needed for smoke test)
```

System dependencies: `openvino` (required at build time — the `openvino` crate
links `libopenvino_c`), `openvino-intel-npu-plugin` + `intel-npu-driver` for NPU,
v4l2, PAM headers. Install/packaging goes through the `Makefile`
(`make install DESTDIR=...`); there is no distro package in the repo anymore.

Inference smoke test (loads real models, runs one pass):

```sh
cargo run -p redface-recognition --example smoke_test   # NPU default
DEVICE=CPU cargo run -p redface-recognition --example smoke_test
```

Run it (plus `cargo test --workspace`) after any change to
`crates/redface-recognition`.

## Inference conventions (redface-recognition)

- Runtime: `openvino` crate 0.11 (openvino-rs). Device selection via
  `DevicePref`: `Npu` (default) compiles for OpenVINO `"NPU"` and falls back to
  `"CPU"` with a stderr warning; `Cpu` forces CPU; `Auto` is a config-compat
  alias of `Npu` — do NOT reintroduce OpenVINO's `AUTO:NPU,CPU` meta-plugin
  (a broken NPU plugin install segfaults inside it, defeating the fallback).
- Detector: `det_10g.onnx` (SCRFD-10G). Outputs are 9 tensors in stride-major
  order (scores/bboxes/kps for strides 8/16/32), **2 anchors per feature point,
  adjacent per point** — entry `i` belongs to feature point `i / 2`. Getting
  this wrong was a real bug once; do not regress it.
- Encoder: `w600k_r50.onnx` (ArcFace R50), 112x112 aligned crop via Umeyama
  similarity transform on the 5 landmarks, BGR, `(x-127.5)/127.5`.
- Both models are reshaped to static input shapes at load (NPU plugin requires
  static shapes). SCRFD input is `(x-127.5)/128` at 640x640.
- `recognize()` applies CLAHE (8x8 tiles, clip 2.0, OpenCV semantics) to the
  grayscale plane before detection *and* encoding — required on these IR
  cameras (Howdy/Visage recipe). It shifts descriptors: enrollment and
  verification must both run the same preprocessing.
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
  deliberately lean (no image/cv crates — CLAHE, resize, alignment are
  hand-rolled).
- No NIR/IR-trained face models exist publicly; don't add model downloads
  without documenting license and source in `README.md` + `Makefile fetch-data`.

## Runtime layout (deployed)

- Config: `/etc/redface/config.json` (`device`, `inference_device`, `threshold`,
  `timeout`, `socket`, `pid_file`).
- Enrollments: `/etc/redface/models/<user>.face`; models: `/usr/share/redface/`.
- Socket `/var/run/redface.sock`, pidfile `/var/run/redface.pid`,
  systemd unit `data/redfaced.service`.
