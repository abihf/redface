package redface

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/abihf/redface/capture"
	"github.com/abihf/redface/facerec"
)

const (
	infraredDevice = "/dev/video2"
	dataDir        = "/usr/share/redface"
)

type VerifyOption struct {
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

	capOption := &capture.Option{
		Device: infraredDevice,
	}
	noFaceFrames := 0
	err = capture.Capture(capOption, func(gray []byte, width, height int) (bool, error) {
		if opt.Timeout > 0 && time.Now().Sub(timeout) >= 0 {
			return false, fmt.Errorf("Timeout %v", opt.Timeout)
		}

		rgb := grayToRGB(gray)
		faces, err := rec.Recognize(rgb, width, height, 0)
		if err != nil {
			return false, err
		}

		if len(faces) == 0 {
			noFaceFrames++
			return true, nil
		}

		distance := math.MaxFloat64
		for i, face := range faces {
			fmt.Printf("Face [%d]:", i)
			for _, model := range models {
				d := facerec.GetDistance(model, face.Descriptor)
				fmt.Printf(" %.3f", d)
				if d < distance {
					distance = d
				}
			}
			println()
		}

		if distance > 0 && distance < opt.Threshold {
			result = true
			return false, nil
		}

		return true, nil
	})
	fmt.Printf("Frames without face found: %d\n", noFaceFrames)
	if err != nil {
		return false, err
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

	err = json.NewDecoder(f).Decode(&res)
	return res, err
}

func grayToRGB(gray []byte) []byte {
	rgb := make([]byte, len(gray)*3)
	for i := 0; i < len(gray); i++ {
		offset := i * 3
		rgb[offset+0] = gray[i]
		rgb[offset+1] = gray[i]
		rgb[offset+2] = gray[i]
	}
	return rgb
}
