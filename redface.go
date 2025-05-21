package redface

import (
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/abihf/redface/capture"
	"github.com/abihf/redface/facerec"
)

type VerifyOption struct {
	Device    string
	ModelFile string
	Timeout   time.Duration
	Threshold float64
}

func Verify(rec *facerec.Recognizer, opt *VerifyOption) (bool, error) {
	models, err := readModels(opt.ModelFile)
	if err != nil {
		return false, err
	}

	result := false

	var timeout time.Time
	if opt.Timeout > 0 {
		timeout = time.Now().Add(opt.Timeout)
	}

	noFaceFrames := 0
	cam := capture.Open(opt.Device)
	defer cam.Close()
	for frame := range cam.Stream() {
		if frame == nil {
			break
		}

		if opt.Timeout > 0 && time.Since(timeout) >= 0 {
			frame.Free()
			return false, fmt.Errorf("timeout %v", opt.Timeout)
		}

		rgb := grayToRGB(frame.Buffer)
		frame.Free()
		recStart := time.Now()
		faces, err := rec.Recognize(rgb, frame.Width, frame.Height, 0)
		if err != nil {
			return false, err
		}

		if len(faces) == 0 {
			noFaceFrames++
			continue
		}
		fmt.Printf("* Found %d faces in %v\n", len(faces), time.Since(recStart))

		for i, face := range faces {
			fmt.Printf("  - Face [%d]:", i)
			for _, model := range models {
				d := model.Distance(&face.Descriptor)
				fmt.Printf(" %.3f", d)
				if d < opt.Threshold {
					println(" (found)")
					return true, nil
				}
			}
			println()
		}
	}
	if noFaceFrames > 0 {
		fmt.Printf("> Frames without face found: %d\n\n", noFaceFrames)
	}

	if cam.Err() != nil {
		return false, cam.Err()
	}

	return result, nil
}

func readModels(file string) ([]facerec.Descriptor, error) {
	var res []facerec.Descriptor
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	for {
		var d facerec.Descriptor
		_, err := d.Unmarshal(f)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		res = append(res, d)
		_, err = fmt.Fscanf(f, "\n")
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
	}

	return res, nil
}

func grayToRGB(gray []byte) []byte {
	rgb := make([]byte, len(gray)*3)
	for i, v := range gray {
		offset := i * 3
		rgb[offset+0] = v
		rgb[offset+1] = v
		rgb[offset+2] = v
	}
	return rgb
}
