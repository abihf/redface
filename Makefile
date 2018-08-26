build: pam

pam: build/pam_redface.so

build/pam_redface.so:
	go build -v -buildmode=c-shared -ldflags="-s -w" -o build/pam_redface.so pam_redface/main.go

install:
	@cp build/pam_redface.so /lib/security/pam_redface.so

clean:
	@rm build/pam_redface.so
