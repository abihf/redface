package main

import (
	"fmt"
	"net"
	"os"
	"os/user"

	"github.com/zro/pam"
)

const pamIgnore pam.Value = 25

type pamRedface struct{}

func (*pamRedface) Authenticate(hdl pam.Handle, args pam.Args) pam.Value {
	userName, err := hdl.GetUser()
	if err != nil {
		fmt.Printf("Can not get user: %s\n", err.Error())
		return pam.UserUnknown
	}

	u, err := user.Lookup(userName)
	if err != nil {
		fmt.Printf("Can not find user %s: %s\n", userName, err.Error())
		return pam.UserUnknown
	}

	sockPath := fmt.Sprintf("/run/user/%s/redface/redfaced.sock", u.Uid)
	if _, err = os.Stat(sockPath); os.IsNotExist(err) {
		return pamIgnore
	}

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		return pamIgnore
	}
	defer conn.Close()

	sendMessage(hdl, "Scanning face...", false)
	_, err = fmt.Fprint(conn, "AUTH pam")
	if err != nil {
		sendMessage(hdl, "Daemon error", true)
		return pam.CredentialUnavailable
	}

	buff := make([]byte, 1024)
	nr, err := conn.Read(buff)
	if err != nil {
		sendMessage(hdl, "Daemon error", true)
		return pam.CredentialUnavailable
	}
	data := string(buff[:nr])
	if data != "SUCCESS" {
		sendMessage(hdl, data, true)
		return pam.CredentialError
	}

	return pam.Success
}

func (*pamRedface) SetCredential(hdl pam.Handle, args pam.Args) pam.Value {
	return pam.Success
}

func sendMessage(hdl pam.Handle, msg string, isError bool) error {
	style := pam.MessageTextInfo
	if isError {
		style = pam.MessageErrorMsg
	}
	_, err := hdl.Conversation(pam.Message{
		Msg:   msg,
		Style: style,
	})
	return err
}

var instance pamRedface

func init() {
	pam.RegisterAuthHandler(&instance)
}

func main() {

}
