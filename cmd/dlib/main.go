package main

import (
	"github.com/abihf/redface"
)

func main() {
	opt := &redface.VerifyOption{
		ModelFile: "data.json",
	}
	redface.Verify(opt)

}
