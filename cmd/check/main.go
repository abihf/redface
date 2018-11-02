package main

import (
	"log"
	"net"

	"os/user"

	"github.com/abihf/redface/protocol"
)

func main() {
	conn, err := net.Dial("unix", protocol.GetSockAddress())
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	currentUser, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}

	err = protocol.WriteAuthReq(conn, currentUser.Uid, "check")
	if err != nil {
		log.Fatal(err)
	}

	res, err := protocol.ReadRes(conn)
	if err != nil {
		log.Fatal(err)
	}

	println("Result", res.Status)
}
