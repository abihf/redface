build: pam util

util: build/redface
pam: build/pam_redface.so

build/pam_redface.so:
	go build -v -buildmode=c-shared -o build/pam_redface.so pam_redface/main.go

build/redface:
	go build -v -o build/redface cmd/redface/main.go

install: install-pam install-util

install-pam: pam
	@cp build/pam_redface.so /lib/security/pam_redface.so

install-util: util
	@cp build/redface /usr/lib/redface

clean:
	@rm -f build/pam_redface.so
	@rm -f build/redface
