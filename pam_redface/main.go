package main

import (
	"bytes"
	"fmt"
	"net"
	"os"
	"os/user"
	"text/template"

	"github.com/abihf/redface/protocol"
	"github.com/zro/pam"
)

const pamIgnore pam.Value = 25

type pamRedface struct{}

func (*pamRedface) Authenticate(hdl pam.Handle, args pam.Args) pam.Value {
	sockPath := protocol.GetSockAddress()
	if _, err := os.Stat(sockPath); os.IsNotExist(err) {
		return pamIgnore
	}

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

	if conditionalFileTemplate, ok := args["ifexist"]; ok {

		conditionalFile, err := executeTemplate(conditionalFileTemplate, u)
		if err != nil {
			fmt.Printf("Can not parse conditional file: %s\n", err.Error())
			return pam.AuthError
		}

		if _, err = os.Stat(conditionalFile); os.IsNotExist(err) {
			return pamIgnore
		}
	}

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		return pam.CredentialUnavailable
	}
	defer conn.Close()

	sendMessage(hdl, "Scanning face...", false)

	clientName, ok := args["client"]
	if !ok {
		clientName = "pam"
	}

	err = protocol.WriteAuthReq(conn, u.Uid, clientName)
	if err != nil {
		sendMessage(hdl, "Daemon error", true)
		return pam.CredentialUnavailable
	}

	res, err := protocol.ReadRes(conn)
	if err != nil {
		sendMessage(hdl, "Daemon error", true)
		return pam.CredentialUnavailable
	}

	if res.Status != protocol.StatusSuccess {
		sendMessage(hdl, res.Error, true)
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

func executeTemplate(tplString string, data interface{}) (string, error) {
	tpl, err := template.New("conditional").Parse(tplString)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	err = tpl.Execute(&buf, data)
	if err != nil {
		return "", err
	}

	return buf.String(), nil
}

var instance pamRedface

func init() {
	pam.RegisterAuthHandler(&instance)
}

func main() {

}
