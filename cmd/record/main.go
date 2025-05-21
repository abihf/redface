package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/abihf/redface/capture"
	"github.com/abihf/redface/facerec"
	"github.com/pkg/errors"
)

func main() {
	if err := mainE(); err != nil {
		fmt.Println(err)
	}
}

func mainE() error {
	rec, err := facerec.NewRecognizer("/usr/share/redface")
	if err != nil {
		return errors.Wrap(err, "Can not initialize face recognizer")
	}
	noFaceFrames := 0

	file, err := os.Create("capture.jsonl")
	if err != nil {
		return errors.Wrap(err, "Can not create capture.json")
	}
	defer file.Close()

	var descriptor facerec.Descriptor
	first := true
	cam := capture.Open("/dev/video2")
	defer cam.Close()
	for frame := range cam.Stream() {
		if frame == nil {
			break
		}
		rgb := grayToRGB(frame.Buffer)
		frame.Free()
		faces, err := rec.Recognize(rgb, frame.Width, frame.Height, 1)
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
			fmt.Fprintln(file)
			strings.NewReader("-0x1.bbef1p-04 0x1.0a6d32p-03 0x1.d8571p-04")
			fmt.Printf("  - Face [%d] (distance: %.3f)", i, distance)
			println()
		}
	}

	return cam.Err()
}

func grayToRGB(gray []byte) []byte {
	rgb := make([]byte, len(gray)*3)
	offset := 0
	for _, v := range gray {
		rgb[offset+0] = v
		rgb[offset+1] = v
		rgb[offset+2] = v
		offset += 3
	}
	return rgb
}
