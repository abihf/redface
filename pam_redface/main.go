package main

import (
	"fmt"
	"os"
	"os/user"

	"github.com/abihf/redface"

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

	keyringFile := fmt.Sprintf("/run/user/%s/keyring/control", u.Uid)
	if _, err = os.Stat(keyringFile); os.IsNotExist(err) {
		return pamIgnore
	}

	modelFile := fmt.Sprintf("/etc/redface/models/%s.xml", userName)
	err = redface.Validate(modelFile)
	if err != nil {
		fmt.Printf("Auth failed: %s\n", err.Error())
		return pamIgnore
	}

	return pam.Success
}

func (*pamRedface) SetCredential(hdl pam.Handle, args pam.Args) pam.Value {
	return pam.Success
}

var instance pamRedface

func init() {
	pam.RegisterAuthHandler(&instance)
}

func main() {

}
