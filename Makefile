DESTDIR = ""
PREFIX = "/usr"
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib
DATADIR = $(PREFIX)/share

BUILD_DIR = "./build"

#----------------------------------------------------------------------------------------
# INSTALL
#----------------------------------------------------------------------------------------

build: pam util daemon

daemon: $(BUILD_DIR)/redfaced
util: $(BUILD_DIR)/redface
pam: $(BUILD_DIR)/pam_redface.so

$(BUILD_DIR)/pam_redface.so:
	go build -v -buildmode=c-shared -o build/pam_redface.so pam_redface/main.go

$(BUILD_DIR)/redface:
	go build -v -o build/redface cmd/redface/main.go

$(BUILD_DIR)/redfaced:
	go build -v -o build/redfaced cmd/redfaced/main.go

#----------------------------------------------------------------------------------------
# INSTALL
#----------------------------------------------------------------------------------------

install: install-pam install-util install-daemon install-data 

install-pam: pam
	install $(BUILD_DIR)/pam_redface.so $(DESTDIR)$(LIBDIR)/security/pam_redface.so

install-util: util
	install $(BUILD_DIR)/redface $(DESTDIR)$(BINDIR)/redface

install-daemon: daemon
	install $(BUILD_DIR)/redfaced $(DESTDIR)$(BINDIR)/redfaced
	install data/redfaced.service $(DESTDIR)$(LIBDIR)/systemd/system/redfaced.service

install-data:
	install -d -m 700 $(DESTDIR)$(DATADIR)/redface
	install data/dlib_face_recognition_resnet_model_v1.dat $(DESTDIR)$(DATADIR)/redface/dlib_face_recognition_resnet_model_v1.dat
	install data/shape_predictor_5_face_landmarks.dat $(DESTDIR)$(DATADIR)/redface/shape_predictor_5_face_landmarks.dat

#----------------------------------------------------------------------------------------
# CLEAN
#----------------------------------------------------------------------------------------

clean:
	rm -f build/pam_redface.so
	rm -f build/redface
	rm -f build/redfaced
