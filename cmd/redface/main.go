package main

import (
	"log"
	"os"

	"github.com/abihf/redface"
)

func main() {
	if len(os.Args) != 3 {
		help()
	}
	switch os.Args[1] {
	case "enroll":
		must(redface.Enroll(os.Args[2]))
		println("Done")
	case "validate":
		must(redface.Validate(os.Args[2]))
		println("OK")
	default:
		help()
	}
}

func must(err error) {
	if err != nil {
		panic(err.Error())
	}
}

func help() {
	log.Fatalf("Usage: %s <enroll|validate> [model file]", os.Args[0])
}
