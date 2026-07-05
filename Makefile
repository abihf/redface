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

RUSTFILES = Cargo.toml $(wildcard crates/**/*.rs) $(wildcard crates/**/Cargo.toml) $(wildcard vendor/dlib-face-recognition/src/**/*.rs)

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
	install $(BUILD_DIR)/redfaced $(DESTDIR)$(BINDIR)/redfaced
	install data/redfaced.service $(DESTDIR)$(LIBDIR)/systemd/system/redfaced.service

install-check: check
	install $(BUILD_DIR)/redface-check $(DESTDIR)$(BINDIR)/redface-check

install-record: record
	install $(BUILD_DIR)/redface-record $(DESTDIR)$(BINDIR)/redface-record

install-data:
	install -d -m 755 $(DESTDIR)$(DATADIR)/redface
	install data/dlib_face_recognition_resnet_model_v1.dat $(DESTDIR)$(DATADIR)/redface/dlib_face_recognition_resnet_model_v1.dat
	install data/shape_predictor_5_face_landmarks.dat $(DESTDIR)$(DATADIR)/redface/shape_predictor_5_face_landmarks.dat

#----------------------------------------------------------------------------------------
# CLEAN
#----------------------------------------------------------------------------------------

clean:
	rm -f build/pam_redface.so
	rm -f build/redfaced
	rm -f build/redface-check
	rm -f build/redface-record
