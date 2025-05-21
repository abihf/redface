package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/abihf/redface"
	"github.com/abihf/redface/facerec"
	"github.com/abihf/redface/protocol"
	"github.com/pkg/errors"
)

const dataDir = "/usr/share/redface"

func main() {
	if err := serve(); err != nil {
		log.Fatal(err)
	}
}

func serve() error {
	procPath := protocol.GetLockFile()
	if isAlreadyRun(procPath) {
		return errors.New("already run")
	}

	recognizer, err := facerec.NewRecognizer(dataDir)
	if err != nil {
		return errors.Wrap(err, "Can not initialize face recognizer")
	}

	writeLockFile(procPath)
	defer os.Remove(procPath)

	sockPath := protocol.GetSockAddress()
	os.Remove(sockPath)

	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		return errors.Wrap(err, "Listen error")
	}
	defer ln.Close()

	os.Chmod(sockPath, 0666)

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

func handle(rec *facerec.Recognizer, c net.Conn) {
	defer c.Close()

	for {
		req, err := protocol.ReadReq(c)
		if err != nil {
			if err.Error() != "EOF" {
				log.Println("Can not read request", err)
			}
			return
		}

		switch req.Action {
		case protocol.ActionAuthenticate:
			authReq := protocol.ToAuthReq(req)
			log.Printf("Authorizing %s\n", authReq.User)

			file := fmt.Sprintf("/etc/redface/models/%s.face", authReq.User)
			success, err := redface.Verify(rec, &redface.VerifyOption{
				Device:    "/dev/video2",
				ModelFile: file,
				Timeout:   10 * time.Second,
				Threshold: 0.1,
			})
			if err == nil && !success {
				err = errors.New("Access denied")
			}

			if err != nil {
				fmt.Printf("Authentication error: %v", err)
				protocol.WriteErrorRes(c, err)
			} else {
				protocol.WriteSuccessRes(c, nil)
			}
		}
	}
}

func isAlreadyRun(path string) bool {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return false
	}

	pidStr, err := os.ReadFile(path)
	if err != nil {
		log.Println("Can not read pid file", err)
		return false
	}
	pid, err := strconv.Atoi(string(pidStr))
	if err != nil {
		log.Println("Invalid existing pid file", err)
		return false
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		log.Println("Can not find current process", err)
		return false
	}

	err = proc.Signal(syscall.Signal(0))
	if err == nil {
		return true
	}

	return false
}

func writeLockFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}

	fmt.Fprintf(f, "%d", os.Getpid())
	return f.Close()
}
