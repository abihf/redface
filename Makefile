DESTDIR = 
PREFIX = /usr
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib
DATADIR = $(PREFIX)/share
PAMDIR = $(LIBDIR)/security

TARGET_DIR = ./target/release

#----------------------------------------------------------------------------------------
# BUILD
#----------------------------------------------------------------------------------------

RUSTFILES = Cargo.toml Cargo.lock $(wildcard crates/**/*.rs) $(wildcard crates/**/Cargo.toml)

MODELS = data/det_10g.onnx data/w600k_r50.onnx
BUFFALO_URL = https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

fetch-data:
	curl -sL -o /tmp/buffalo_l.zip $(BUFFALO_URL)
	unzip -o -j /tmp/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d data/
	rm -f /tmp/buffalo_l.zip

build: pam daemon check record

daemon: $(TARGET_DIR)/redfaced
pam: $(TARGET_DIR)/libpam_redface.so
check: $(TARGET_DIR)/redface-check
record: $(TARGET_DIR)/redface-record

$(TARGET_DIR)/libpam_redface.so: $(RUSTFILES)
	cargo build --release -p pam-redface

$(TARGET_DIR)/redfaced: $(RUSTFILES)
	cargo build --release -p redfaced

$(TARGET_DIR)/redface-check: $(RUSTFILES)
	cargo build --release -p redface-check

$(TARGET_DIR)/redface-record: $(RUSTFILES)
	cargo build --release -p redface-record --bin redface-record

#----------------------------------------------------------------------------------------
# INSTALL
#----------------------------------------------------------------------------------------

install: install-pam install-daemon install-check install-record install-data 

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

install-data:
	install -d -m 755 $(DESTDIR)$(DATADIR)/redface
	install data/det_10g.onnx $(DESTDIR)$(DATADIR)/redface/det_10g.onnx
	install data/w600k_r50.onnx $(DESTDIR)$(DATADIR)/redface/w600k_r50.onnx

#----------------------------------------------------------------------------------------
# CLEAN
#----------------------------------------------------------------------------------------

clean:
	rm -f build/pam_redface.so
	rm -f build/redfaced
	rm -f build/redface-check
	rm -f build/redface-record
