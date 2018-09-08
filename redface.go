package redface

import (
	"fmt"
	"image/color"
	"os"
	"time"

	"github.com/blackjack/webcam"
	"github.com/pkg/errors"
	"gocv.io/x/gocv"
	"gocv.io/x/gocv/contrib"
)

const (
	infraredDevice = "/dev/video2"
	classifierFile = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"
)

type FaceCallback func(*gocv.Mat) (bool, error)

func Enroll(modelFile string) error {
	recognizer := contrib.NewLBPHFaceRecognizer()
	if _, err := os.Stat(modelFile); !os.IsNotExist(err) {
		recognizer.LoadFile(modelFile)
	}

	timeout := time.Now().Add(3 * time.Second)
	lastSecond := time.Duration(4)
	err := FindFace(true, func(face *gocv.Mat) (bool, error) {
		s := timeout.Sub(time.Now()).Round(time.Second)

		if s < 0 {
			recognizer.Update([]gocv.Mat{*face}, []int{1})
			return true, nil
		} else if s != lastSecond {
			fmt.Printf("Will take picture in %v...\n", s)
			lastSecond = s
		}
		return false, nil
	})
	if err != nil {
		return err
	}

	recognizer.SaveFile(modelFile)
	return nil
}

func Validate(modelFile string, showWindow bool) error {
	recognizer := contrib.NewLBPHFaceRecognizer()
	recognizer.LoadFile(modelFile)

	err := FindFace(showWindow, func(face *gocv.Mat) (bool, error) {
		res := recognizer.PredictExtendedResponse(*face)
		if showWindow {
			fmt.Printf("Confidence %v\n", res.Confidence)
		}
		return res.Confidence <= 40.0, nil
	})
	return err
}

func FindFace(showWindow bool, cb FaceCallback) error {
	cam, err := webcam.Open(infraredDevice)
	if err != nil {
		return errors.Wrap(err, "Can not open device ")
	}
	defer cam.Close()

	// load classifier to recognize faces
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load(classifierFile) {
		return errors.Errorf("Error reading cascade file: %v\n", classifierFile)
	}

	var window *gocv.Window
	if showWindow {
		window = gocv.NewWindow("Face Detect")
		defer window.Close()

	}

	white := color.RGBA{255, 255, 255, 0}

	err = cam.StartStreaming()
	if err != nil {
		return errors.Wrap(err, "Can not start streaming")
	}

	mask := gocv.NewMat()
	defer mask.Close()

	hist := gocv.NewMatWithSize(340, 340, gocv.MatTypeCV8U)
	defer hist.Close()

	for t := 360; t > 0; t-- {
		running, err := func() (bool, error) {
			err = cam.WaitForFrame(10)
			switch err.(type) {
			case nil:
			case *webcam.Timeout:
				fmt.Fprint(os.Stderr, err.Error())
				return true, nil
			default:
				return false, errors.Wrap(err, "Failed when waiting for frame")
			}

			frame, err := cam.ReadFrame()
			if err != nil {
				return false, errors.Wrap(err, "Can not read frame")
			}

			mat, err := decodeImage(frame)
			if err != nil {
				return false, errors.Wrap(err, "Can not decode image")
			}
			defer mat.Close()

			gocv.CalcHist(
				[]gocv.Mat{mat},
				[]int{0},
				mask,
				&hist,
				[]int{8},
				[]float64{0, 256},
				false,
			)
			// i have no idea
			firstHist := hist.GetFloatAt(0, 0)
			sumHist := float32(hist.Sum().Val1)
			if firstHist/sumHist > 0.5 {
				return true, nil
			}

			// detect faces
			rects := classifier.DetectMultiScale(mat)

			var face gocv.Mat
			if len(rects) == 1 {
				face = mat.Region(rects[0])
				defer face.Close()
			}

			if showWindow {
				for _, r := range rects {
					r.Inset(-2)
					gocv.Rectangle(&mat, r, white, 2)
				}

				window.IMShow(mat)
				window.WaitKey(1)
			}

			if len(rects) == 1 {
				ok, err := cb(&face)
				if err != nil {
					return false, err
				}
				if ok {
					return false, nil
				}
			}

			return true, nil
		}()

		if err != nil {
			return err
		}
		if !running {
			return nil
		}
	}

	return errors.Errorf("Can not find face")
}

func decodeImage(buf []byte) (gocv.Mat, error) {
	width := 340
	height := 340
	return gocv.NewMatFromBytes(width, height, gocv.MatTypeCV8UC1, buf)
}
