package main

import (
	"fmt"
	"os"

	"github.com/abihf/redface/capture"
	"github.com/abihf/redface/config"
	"github.com/abihf/redface/facerec"
)

var conf = config.Load()

func main() {
	if err := mainE(); err != nil {
		fmt.Println(err)
	}
}

func mainE() error {
	rec, err := facerec.NewRecognizer("/usr/share/redface")
	if err != nil {
		return fmt.Errorf("can not initialize face recognizer: %w", err)
	}
	noFaceFrames := 0

	file, err := os.Create("capture.face")
	if err != nil {
		return fmt.Errorf("can not create capture.face: %w", err)
	}
	defer file.Close()

	var descriptor facerec.Descriptor
	first := true
	cam := capture.Open(conf.Device)
	defer cam.Close()
	for frame := range cam.Stream() {
		if frame == nil {
			break
		}
		faces, err := rec.Recognize(frame.Buffer, frame.Width, frame.Height, 1)
		if err != nil {
			return err
		}

		if len(faces) == 0 {
			noFaceFrames++
			fmt.Println("	- No face detected")
			continue
		}

		for i, face := range faces {
			distance := float64(1)
			if first {
				descriptor = face.Descriptor
				first = false
			} else {
				distance = face.Descriptor.Distance(&descriptor)
				descriptor = descriptor.Middle(&face.Descriptor)
			}
			descriptor.Marshal(file)
			file.Write([]byte{'\n'})
			fmt.Printf("  - Face [%d] (distance: %.3f)", i, distance)
			println()
		}
	}

	return cam.Err()
}
