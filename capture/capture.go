package capture

import (
	"fmt"
	"os"

	"github.com/blackjack/webcam"
	"github.com/pkg/errors"
)

type Processor func(img []byte, width, height int) (bool, error)

type Option struct {
	Device string
}

func Capture(opt *Option, processor Processor) error {
	cam, err := webcam.Open(opt.Device)
	if err != nil {
		return errors.Wrap(err, "Can not open device ")
	}
	defer cam.Close()

	err = cam.StartStreaming()
	if err != nil {
		return errors.Wrap(err, "Can not start streaming")
	}

	frameChan := make(chan []byte)
	var captureError error

	go func() {
		defer close(frameChan)

		for {
			err = cam.WaitForFrame(10)
			switch err.(type) {
			case nil:
			case *webcam.Timeout:
				fmt.Fprint(os.Stderr, err.Error())
				continue
			default:
				captureError = err
				return
			}

			frame, err := cam.ReadFrame()
			if err != nil {
				captureError = err
				return
			}

			if len(frameChan) > 0 {
				continue
			}

			if !isGoodImage(frame) {
				continue
			}

			frameChan <- frame
		}
	}()

	for {
		frame := <-frameChan
		if captureError != nil {
			return captureError
		}

		cont, err := processor(frame, 340, 340)
		if err != nil {
			return err
		}

		if !cont {
			break
		}
	}

	return nil
}

func isGoodImage(img []byte) bool {
	dark := 0
	total := len(img)
	for i := 0; i < total; i++ {
		if img[i] < 127 {
			dark++
		}
	}
	darkness := float64(dark) / float64(total)
	return darkness > 0.25 && darkness < 0.75
}
