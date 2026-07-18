# redface

## Intro
This is my work-in-progress project to enable face recognition based pam authentication.

## Documentation
[Wiki](https://github.com/abihf/redface/wiki)

## How it works
Face recognition runs fully on ONNX models, via OpenCV's DNN CPU backend by default, or the [`openvino`](https://crates.io/crates/openvino) crate (OpenVINO Runtime) with the opt-in `openvino` cargo feature:

1. **Preprocessing** — IR frames are contrast-normalized with OpenCV CLAHE (8x8 tiles, clip limit 2.0); raw IR frames are too low-contrast for RGB-trained models.
2. **Detection** — `det_10g.onnx` (SCRFD-10G, InsightFace) finds faces and 5-point landmarks.
3. **Alignment** — landmarks are warped to a 112x112 crop with a similarity transform (Umeyama) and `warpAffine`.
4. **Embedding** — `w600k_r50.onnx` (ArcFace ResNet-50) produces a 512-dim descriptor.

Authentication compares descriptors with **cosine similarity** (`threshold` in `/etc/redface/config.json`, default `0.9`, higher = stricter).

### Inference device
The `inference_device` config option (and `redface-record --inference-device`) selects the OpenVINO device in an `openvino`-feature build (in a default build every value runs on the OpenCV DNN CPU backend):

| Value | Behavior |
|-------|----------|
| `NPU` (default) | OpenVINO `NPU` device (e.g. Arrow Lake), falls back to `CPU` if the driver/plugin is unavailable |
| `CPU` | OpenVINO `CPU` device; works on any x86-64 |
| `AUTO` | Alias for `NPU` (config compatibility). We avoid OpenVINO's `AUTO:NPU,CPU` meta-plugin on purpose: a broken NPU plugin install segfaults inside AUTO instead of returning a catchable error, which would defeat the CPU fallback |

Pixel processing (CLAHE, resize, alignment warp) uses the `opencv` crate, which needs the system `opencv` package and `clang` (libclang) for binding generation. A default build has no OpenVINO dependency; opting into the `openvino` cargo feature makes the `openvino` crate link against the system OpenVINO Runtime (`libopenvino_c`) at build time, so the `openvino` package must be installed for that build. For NPU inference (OpenVINO build) you additionally need `openvino-intel-npu-plugin` and the `intel-npu-driver` kernel driver.

To build with OpenVINO, opt into the cargo feature:

```sh
cargo build --workspace --features openvino
```

In a default build, inference runs on OpenCV's DNN CPU backend for every `inference_device` value (a startup message on stderr notes the backend). The DNN backend is CPU-only — no NPU — and typically 2-5x slower than OpenVINO CPU, so the `openvino` build is preferable when OpenVINO is available. Enrollment and verification should use the same backend for consistent descriptors.

Note that the CLAHE preprocessing shifts descriptors: re-enroll with `redface-record` after upgrading to a build that changes preprocessing (e.g. the switch to OpenCV-based CLAHE/resize/alignment).

### Models
Models come from the InsightFace `buffalo_l` pack (non-commercial license). Fetch them with:

```sh
make fetch-data   # downloads det_10g.onnx + w600k_r50.onnx into data/
```

## Reference
* https://github.com/boltgolt/howdy
* https://github.com/deepinsight/insightface
