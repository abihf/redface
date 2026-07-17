# redface

## Intro
This is my work-in-progress project to enable face recognition based pam authentication.

## Documentation
[Wiki](https://github.com/abihf/redface/wiki)

## How it works
Face recognition runs fully on ONNX models via the [`openvino`](https://crates.io/crates/openvino) crate (OpenVINO Runtime):

1. **Preprocessing** — IR frames are contrast-normalized with CLAHE (8x8 tiles, clip limit 2.0); raw IR frames are too low-contrast for RGB-trained models.
2. **Detection** — `det_10g.onnx` (SCRFD-10G, InsightFace) finds faces and 5-point landmarks.
3. **Alignment** — landmarks are warped to a 112x112 crop with a similarity transform (Umeyama).
4. **Embedding** — `w600k_r50.onnx` (ArcFace ResNet-50) produces a 512-dim descriptor.

Authentication compares descriptors with **cosine similarity** (`threshold` in `/etc/redface/config.json`, default `0.9`, higher = stricter).

### Inference device
The `inference_device` config option (and `redface-record --inference-device`) selects the OpenVINO device:

| Value | Behavior |
|-------|----------|
| `NPU` (default) | OpenVINO `NPU` device (e.g. Arrow Lake), falls back to `CPU` if the driver/plugin is unavailable |
| `CPU` | OpenVINO `CPU` device; works on any x86-64 |
| `AUTO` | Alias for `NPU` (config compatibility). We avoid OpenVINO's `AUTO:NPU,CPU` meta-plugin on purpose: a broken NPU plugin install segfaults inside AUTO instead of returning a catchable error, which would defeat the CPU fallback |

The `openvino` crate links against the system OpenVINO Runtime (`libopenvino_c`) at build time, so the `openvino` package must be installed to build. For NPU inference you additionally need `openvino-intel-npu-plugin` and the `intel-npu-driver` kernel driver.

Note that the CLAHE preprocessing shifts descriptors: re-enroll with `redface-record` after upgrading to a build that includes it.

### Models
Models come from the InsightFace `buffalo_l` pack (non-commercial license). Fetch them with:

```sh
make fetch-data   # downloads det_10g.onnx + w600k_r50.onnx into data/
```

## Reference
* https://github.com/boltgolt/howdy
* https://github.com/deepinsight/insightface
