# redface

## Intro
This is my work-in-progress project to enable face recognition based pam authentication.

## Documentation
[Wiki](https://github.com/abihf/redface/wiki)

## How it works
Face recognition runs on the InsightFace models via [ncnn](https://github.com/Tencent/ncnn) on the Vulkan GPU by default (with automatic CPU fallback; through the [`rust-ncnn`](https://github.com/tpoisonooo/rust-ncnn) FFI, vendored under `crates/ncnn-bind`), or the [`openvino`](https://crates.io/crates/openvino) crate (OpenVINO Runtime) with the opt-in `openvino` cargo feature:

1. **Preprocessing** — IR frames are contrast-normalized with OpenCV CLAHE (8x8 tiles, clip limit 2.0); raw IR frames are too low-contrast for RGB-trained models.
2. **Detection** — `det_10g.onnx` (SCRFD-10G, InsightFace) finds faces and 5-point landmarks.
3. **Alignment** — landmarks are warped to a 112x112 crop with a similarity transform (Umeyama) and `warpAffine`.
4. **Embedding** — `w600k_r50.onnx` (ArcFace ResNet-50) produces a 512-dim descriptor.

Authentication compares descriptors with **cosine similarity** (`threshold` in `/etc/redface/config.json`, default `0.9`, higher = stricter).

### Inference device
The `inference_device` config option (and `redface-record --inference-device`) selects the inference target. In an `openvino`-feature build it picks the OpenVINO device (table below); in a default ncnn build `NPU`/`AUTO` run on the Vulkan GPU (CPU fallback if no Vulkan device is present) and `CPU` forces CPU:

| Value | Behavior |
|-------|----------|
| `NPU` (default) | OpenVINO `NPU` device (e.g. Arrow Lake), falls back to `CPU` if the driver/plugin is unavailable |
| `CPU` | OpenVINO `CPU` device; works on any x86-64 |
| `AUTO` | Alias for `NPU` (config compatibility). We avoid OpenVINO's `AUTO:NPU,CPU` meta-plugin on purpose: a broken NPU plugin install segfaults inside AUTO instead of returning a catchable error, which would defeat the CPU fallback |

Pixel processing (CLAHE, resize, alignment warp) uses the `opencv` crate, which needs the system `opencv` package and `clang` (libclang) for binding generation. Inference uses ncnn: the vendored `ncnn-bind` crate links the system `libncnn` and generates its bindings from the system ncnn headers (`NCNN_DIR` in `.cargo/config.toml`, default `/usr`), so the `ncnn` package (library + headers, built with Vulkan support) must be installed. GPU inference also needs a Vulkan driver/ICD for your GPU (e.g. `vulkan-radeon` or `vulkan-intel`); without one, ncnn falls back to CPU. A default build has no OpenVINO dependency; opting into the `openvino` cargo feature makes the `openvino` crate link against the system OpenVINO Runtime (`libopenvino_c`) at build time, so the `openvino` package must be installed for that build. For NPU inference (OpenVINO build) you additionally need `openvino-intel-npu-plugin` and the `intel-npu-driver` kernel driver.

To build with OpenVINO, opt into the cargo feature and drop the default ncnn backend (`--no-default-features` keeps the `ncnn` crate and its `libncnn` link out of the build):

```sh
cargo build --workspace --no-default-features --features openvino
```

In a default build, inference runs on ncnn's Vulkan GPU for `NPU`/`AUTO` (a startup message on stderr notes the backend, and ncnn prints its Vulkan device on load) and falls back to CPU automatically when no Vulkan device is present; `CPU` forces CPU. The `openvino` build is still preferable when an Intel NPU is available. Enrollment and verification should use the same backend for consistent descriptors.

Note that the CLAHE preprocessing shifts descriptors: re-enroll with `redface-record` after upgrading to a build that changes preprocessing (e.g. the switch to OpenCV-based CLAHE/resize/alignment).

### Models
Models come from the InsightFace `buffalo_l` pack (non-commercial license). Fetch and prepare them with:

```sh
make fetch-data      # downloads det_10g.onnx + w600k_r50.onnx into data/
make convert-models  # converts them to ncnn .param/.bin with pnnx (run via pipx)
```

The default ncnn backend reads `det_10g.param`/`.bin` and `w600k_r50.param`/`.bin`. `make convert-models` runs [pnnx](https://github.com/pnnx/pnnx) through `pipx run` (fetched on demand, so `pipx` must be on `PATH` — e.g. `uv tool install pipx`); pnnx fixes the models' dynamic input shapes (640x640 detector, 112x112 encoder) and keeps fp32 weights (`fp16=0`). The opt-in `openvino` build reads the `.onnx` files directly.

## Reference
* https://github.com/boltgolt/howdy
* https://github.com/deepinsight/insightface
