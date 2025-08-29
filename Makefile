DESTDIR = 
PREFIX = /usr
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib
DATADIR = $(PREFIX)/share
PAMDIR = $(LIBDIR)/security

BUILD_DIR = ./build

#----------------------------------------------------------------------------------------
# BUILD
#----------------------------------------------------------------------------------------

GOFILES = $(wildcard **/*.go) $(wildcard *.go) go.mod go.sum

build: pam daemon

daemon: $(BUILD_DIR)/redfaced
pam: $(BUILD_DIR)/pam_redface.so

$(BUILD_DIR)/pam_redface.so: $(GOFILES)
	go build -v -buildmode=c-shared -o $@ ./pam_redface/main.go

$(BUILD_DIR)/redfaced: $(GOFILES)
	go build -v -o $@ ./cmd/redfaced/main.go

#----------------------------------------------------------------------------------------
# INSTALL
#----------------------------------------------------------------------------------------

install: install-pam install-daemon install-data 

install-pam: pam
	install $(BUILD_DIR)/pam_redface.so $(DESTDIR)$(PAMDIR)/pam_redface.so

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
	rm -f build/redfaced
