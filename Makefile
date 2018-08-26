build: pam util

util: build/redface
pam: build/pam_redface.so

build/pam_redface.so:
	go build -v -buildmode=c-shared -ldflags="-s -w" -o build/pam_redface.so pam_redface/main.go

build/redface:
	go build -v -ldflags="-s -w" -o build/redface cmd/redface/main.go

install:
	@cp build/pam_redface.so /lib/security/pam_redface.so
	@cp build/redface /usr/lib/redface

clean:
	rm build/pam_redface.so
	rm build/redface
