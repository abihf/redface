package main

import (
	"strings"
	"time"

	"github.com/abihf/redface"

	"github.com/abihf/redface/facerec"
	"github.com/pkg/errors"

	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"os/user"
	"path"
	"syscall"
)

const dataDir = "/usr/share/redface"

var modelFile string

func main() {
	if err := serve(); err != nil {
		log.Fatal(err)
	}
}

func serve() error {
	currentUser, err := user.Current()
	if err != nil {
		return errors.Wrap(err, "Can not get current user")
	}

	baseDir := fmt.Sprintf("/run/user/%s/redface", currentUser.Uid)
	modelFile = fmt.Sprintf("/etc/redface/models/%s.json", currentUser.Uid)

	procPath := path.Join(baseDir, "redfaced.pid")
	if _, err = os.Stat(procPath); !os.IsNotExist(err) {
		return errors.Errorf("%s already exist", procPath)
	}

	recognizer, err := facerec.NewRecognizer(dataDir)
	if err != nil {
		return errors.Wrap(err, "Can not initialize face recognizer")
	}

	os.MkdirAll(baseDir, 0700)
	writeLockFile(procPath)
	defer os.Remove(procPath)

	sockPath := path.Join(baseDir, "redfaced.sock")
	os.Remove(sockPath)

	log.Println("Starting echo server")
	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		return errors.Wrap(err, "Listen error")
	}
	defer ln.Close()

	os.Chmod(sockPath, 0600)

	go func() {
		for {
			fd, err := ln.Accept()

			if err != nil {
				if opErr, ok := err.(*net.OpError); ok {
					if opErr.Err.Error() == "use of closed network connection" {
						return
					}
				}
				log.Printf("Accept error: %v\n", err.Error())
				return
			}

			go handle(recognizer, fd)
		}
	}()

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, os.Interrupt, syscall.SIGTERM)

	sig := <-sigc
	log.Printf("Caught signal %s: shutting down.", sig)
	return nil
}

func writeLockFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}

	fmt.Fprintf(f, "%d", os.Getpid())
	return f.Close()
}

func handle(rec *facerec.Recognizer, c net.Conn) {
	defer c.Close()

	buf := make([]byte, 512)
	nr, err := c.Read(buf)
	if err != nil {
		return
	}

	data := string(buf[0:nr])
	if strings.HasPrefix(data, "AUTH ") {
		println(data)
		success, err := redface.Verify(rec, &redface.VerifyOption{
			ModelFile: modelFile,
			Timeout:   10 * time.Second,
			Threshold: 0.12,
		})
		if err != nil {
			fmt.Fprintf(c, "Error: %v", err)
			return
		}
		if !success {
			fmt.Fprint(c, "Access Denied")
			return
		}
		fmt.Fprint(c, "SUCCESS")
		return
	}
	fmt.Fprintf(c, "Invalid command")
}
