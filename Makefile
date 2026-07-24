DESTDIR = 
PREFIX = /usr
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib
DATADIR = $(PREFIX)/share
PAMDIR = $(LIBDIR)/security

TARGET_DIR = ./target/release

ENABLE_OPENVINO ?= $(shell pkg-config --exists openvino && echo 1 || echo 0)

# --no-default-features keeps the default ncnn backend out of OpenVINO builds.
OPENVINO_ARGS = 
ifeq ($(ENABLE_OPENVINO),1)
	OPENVINO_ARGS = --no-default-features --features=openvino
endif

#----------------------------------------------------------------------------------------
# BUILD
#----------------------------------------------------------------------------------------

RUSTFILES = Cargo.toml Cargo.lock $(shell find crates -name '*.rs' -o -name 'Cargo.toml')

MODELS = data/det_10g.onnx data/w600k_r50.onnx
NCNN_MODELS = data/det_10g.param data/det_10g.bin data/w600k_r50.param data/w600k_r50.bin
BUFFALO_URL = https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

fetch-data:
	curl -sL -o /tmp/buffalo_l.zip $(BUFFALO_URL)
	unzip -o -j /tmp/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d data/
	rm -f /tmp/buffalo_l.zip

# The default inference backend is ncnn, which reads .param/.bin converted from
# the ONNX models with pnnx (run via pipx, which fetches it on demand). pnnx
# fixes the dynamic input shapes and strips the SCRFD Shape/Gather nodes that
# ncnn cannot represent; fp16=0 keeps fp32 weights to match the source models.
# pnnx's intermediate artifacts (*.pnnx.*, *_pnnx.py, *_ncnn.py) are removed.
convert-models:
	@test -f data/det_10g.onnx -a -f data/w600k_r50.onnx || { \
		echo "error: ONNX models missing in data/; download them first (see fetch-data)" >&2; exit 1; }
	@command -v pipx >/dev/null 2>&1 || { \
		echo "error: pipx not found on PATH; install pipx (e.g. 'uv tool install pipx') to convert the models" >&2; exit 1; }
	cd data && pipx run pnnx det_10g.onnx 'inputshape=[1,3,640,640]' fp16=0 ncnnparam=det_10g.param ncnnbin=det_10g.bin
	cd data && pipx run pnnx w600k_r50.onnx 'inputshape=[1,3,112,112]' fp16=0 ncnnparam=w600k_r50.param ncnnbin=w600k_r50.bin
	rm -f data/*.pnnx.* data/*.pnnxsim.onnx data/*_pnnx.py data/*_ncnn.py

build: pam daemon check record lock osd

daemon: $(TARGET_DIR)/redfaced
pam: $(TARGET_DIR)/libpam_redface.so
check: $(TARGET_DIR)/redface-check
record: $(TARGET_DIR)/redface-record
toolkit: $(TARGET_DIR)/libredface_toolkit.so
lock: $(TARGET_DIR)/libredface_toolkit.so $(TARGET_DIR)/redface-lock
osd: $(TARGET_DIR)/libredface_toolkit.so $(TARGET_DIR)/redface-osd

$(TARGET_DIR)/libpam_redface.so: $(RUSTFILES)
	cargo build --release -p pam-redface $(OPENVINO_ARGS)

$(TARGET_DIR)/redfaced: $(RUSTFILES)
	cargo build --release -p redfaced $(OPENVINO_ARGS)

$(TARGET_DIR)/redface-check: $(RUSTFILES)
	cargo build --release -p redface-check $(OPENVINO_ARGS)

$(TARGET_DIR)/redface-record: $(RUSTFILES)
	cargo build --release -p redface-record --bin redface-record $(OPENVINO_ARGS)

# The shared Wayland/Vulkan UI toolkit is a dylib; the locker and the OSD link
# it dynamically. prefer-dynamic picks the .so over the rlib, LTO must be off
# (incompatible with prefer-dynamic), and the rpath covers both the target dir
# ($$ORIGIN) and the installed location (/usr/lib).
DYNAMIC_ARGS = -C prefer-dynamic -C link-args=-Wl,-rpath,/usr/lib,-rpath,\$$ORIGIN
DYNAMIC_CONFIG = --config profile.release.lto=false

# against it, so it must be built (and installed) alongside them.
# The locker does no inference itself (it talks to redfaced over the socket),
# so it is not gated by OPENVINO_ARGS. All three artifacts must come from ONE
# cargo invocation: dynamically linked binaries record the toolkit's metadata
# hash in their undefined symbols, and per-package builds can produce a .so
# with a different hash (runtime "symbol lookup error").
$(TARGET_DIR)/libredface_toolkit.so $(TARGET_DIR)/redface-lock $(TARGET_DIR)/redface-osd: $(RUSTFILES)
	RUSTFLAGS="$(DYNAMIC_ARGS)" cargo build --release -p redface-toolkit -p redface-lock -p redface-osd $(DYNAMIC_CONFIG)

#----------------------------------------------------------------------------------------
# INSTALL
#----------------------------------------------------------------------------------------

install: install-pam install-daemon install-check install-record install-ui install-data

install-pam: pam
	install $(TARGET_DIR)/libpam_redface.so $(DESTDIR)$(PAMDIR)/pam_redface.so

install-daemon: daemon
	install $(TARGET_DIR)/redfaced $(DESTDIR)$(BINDIR)/redfaced

install-unit:
	install data/redfaced.service $(DESTDIR)$(LIBDIR)/systemd/system/redfaced.service

install-check: check
	install $(TARGET_DIR)/redface-check $(DESTDIR)$(BINDIR)/redface-check

install-record: record
	install $(TARGET_DIR)/redface-record $(DESTDIR)$(BINDIR)/redface-record

install-ui: install-toolkit install-lock install-osd

install-toolkit: toolkit
	install $(TARGET_DIR)/libredface_toolkit.so $(DESTDIR)$(LIBDIR)/libredface_toolkit.so

install-lock: lock
	install $(TARGET_DIR)/redface-lock $(DESTDIR)$(BINDIR)/redface-lock
	install -m 644 data/redface-lock.pam $(DESTDIR)/etc/pam.d/redface-lock

install-osd: osd
	install $(TARGET_DIR)/redface-osd $(DESTDIR)$(BINDIR)/redface-osd

# Installs both formats: .param/.bin for the default ncnn backend, .onnx for
# opt-in openvino builds.
install-data:
	install -d -m 755 $(DESTDIR)$(DATADIR)/redface
	install data/det_10g.onnx $(DESTDIR)$(DATADIR)/redface/det_10g.onnx
	install data/w600k_r50.onnx $(DESTDIR)$(DATADIR)/redface/w600k_r50.onnx
	install data/det_10g.param data/det_10g.bin $(DESTDIR)$(DATADIR)/redface/
	install data/w600k_r50.param data/w600k_r50.bin $(DESTDIR)$(DATADIR)/redface/

#----------------------------------------------------------------------------------------
# CLEAN
#----------------------------------------------------------------------------------------

clean:
	rm -f $(TARGET_DIR)/pam_redface.so
	rm -f $(TARGET_DIR)/redfaced
	rm -f $(TARGET_DIR)/redface-check
	rm -f $(TARGET_DIR)/redface-record
	rm -f $(TARGET_DIR)/redface-lock
	rm -f $(TARGET_DIR)/redface-osd
	rm -f $(TARGET_DIR)/libredface_toolkit.so
