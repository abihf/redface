# redface

## Intro
This is my work-in-progress project to enable face recognition based pam authentication. For now, it's hardcoded to use embedded infrared camera on Dell XPS 9370. This camera can capture 340x340 8-bit grayscale images at 60fps.

## Documentation
[Wiki](https://github.com/abihf/redface/wiki)

## Rust Migration
Initial Rust implementation lives under `crates/redface-core`, `crates/redface-recognition`, `crates/redface-record`, `crates/redface-capture`, `crates/redface-runtime`, `crates/redfaced`, `crates/redface-check`, and `crates/pam-redface`.

Current validation entrypoint:

```sh
cargo test -p redface-core
cargo test -p redface-recognition --lib
cargo test -p redface-record --lib
cargo test -p redface-capture --lib
cargo test -p redface-runtime
cargo test -p redface-record --bin redface-record
```

Current Rust record binary:

```sh
cargo run -p redface-record --bin redface-record -- --device /dev/video0
```

Current Rust daemon and check binaries:

```sh
cargo run -p redfaced
cargo run -p redface-check
```

## Reference
* https://github.com/boltgolt/howdy
* http://dlib.net/dnn_face_recognition_ex.cpp.html
* https://github.com/Kagami/go-face
* https://github.com/ageitgey/face_recognition
