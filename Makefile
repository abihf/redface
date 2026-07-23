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

build: pam daemon check record lock

daemon: $(TARGET_DIR)/redfaced
pam: $(TARGET_DIR)/libpam_redface.so
check: $(TARGET_DIR)/redface-check
record: $(TARGET_DIR)/redface-record
lock: $(TARGET_DIR)/redface-lock

$(TARGET_DIR)/libpam_redface.so: $(RUSTFILES)
	cargo build --release -p pam-redface $(OPENVINO_ARGS)

$(TARGET_DIR)/redfaced: $(RUSTFILES)
	cargo build --release -p redfaced $(OPENVINO_ARGS)

$(TARGET_DIR)/redface-check: $(RUSTFILES)
	cargo build --release -p redface-check $(OPENVINO_ARGS)

$(TARGET_DIR)/redface-record: $(RUSTFILES)
	cargo build --release -p redface-record --bin redface-record $(OPENVINO_ARGS)

# The locker does no inference itself (it talks to redfaced over the socket),
# so it is not gated by OPENVINO_ARGS.
$(TARGET_DIR)/redface-lock: $(RUSTFILES)
	cargo build --release -p redface-lock

#----------------------------------------------------------------------------------------
# INSTALL
#----------------------------------------------------------------------------------------

install: install-pam install-daemon install-check install-record install-lock install-data 

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

install-lock: lock
	install $(TARGET_DIR)/redface-lock $(DESTDIR)$(BINDIR)/redface-lock
	install -m 644 data/redface-lock.pam $(DESTDIR)/etc/pam.d/redface-lock

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
