package protocol

import (
	"encoding/json"
	"io"
)

func GetSockAddress() string {
	return "/var/run/redface.sock"
}

func GetLockFile() string {
	return "/var/run/redface.pid"
}

type Action string

const (
	ActionAuthenticate Action = "AUTH"
)

type Req struct {
	Action Action            `json:"action"`
	Params map[string]string `json:"params"`
}

type AuthReq struct {
	Client string `json:"client"`
	User   string `json:"user"`
}

type Status string

const (
	StatusSuccess Status = "SUCCESS"
	StatusError          = "ERROR"
)

type Res struct {
	Status Status            `json:"status"`
	Error  string            `json:"error"`
	Extras map[string]string `json:"extras"`
}

func ReadReq(r io.Reader) (*Req, error) {
	var req Req
	err := json.NewDecoder(r).Decode(&req)
	return &req, err
}

func ReadRes(r io.Reader) (*Res, error) {
	var res Res
	err := json.NewDecoder(r).Decode(&res)
	return &res, err
}

func ToAuthReq(req *Req) *AuthReq {
	return &AuthReq{
		Client: req.Params["client"],
		User:   req.Params["user"],
	}
}

func WriteAuthReq(w io.Writer, user, client string) error {
	req := Req{
		Action: ActionAuthenticate,
		Params: map[string]string{
			"client": client,
			"user":   user,
		},
	}
	return json.NewEncoder(w).Encode(&req)
}

func WriteSuccessRes(w io.Writer, extras map[string]string) error {
	res := Res{
		Status: StatusSuccess,
		Extras: extras,
	}
	return json.NewEncoder(w).Encode(&res)
}

func WriteErrorRes(w io.Writer, err error) error {
	res := Res{
		Status: StatusError,
		Error:  err.Error(),
	}
	return json.NewEncoder(w).Encode(&res)
}
