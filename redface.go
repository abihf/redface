package redface

import (
	"fmt"
	"image"
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

	err = cam.StartStreaming()
	if err != nil {
		return errors.Wrap(err, "Can not start streaming")
	}

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

	for t := 360; t > 0; t-- {
		err = cam.WaitForFrame(1000)
		switch err.(type) {
		case nil:
		case *webcam.Timeout:
			fmt.Fprint(os.Stderr, err.Error())
			continue
		default:
			return errors.Wrap(err, "Failed when waiting for frame")
		}

		frame, err := cam.ReadFrame()
		if err != nil {
			return errors.Wrap(err, "Can not read frame")
		}

		mat, err := decodeImage(frame)
		if err != nil {
			return errors.Wrap(err, "Can not decode image")
		}

		// detect faces
		rects := classifier.DetectMultiScale(mat)

		var face gocv.Mat
		if len(rects) == 1 {
			face = mat.Region(rects[0])
		}

		if showWindow {
			// draw a rectangle around each face on the original image,
			// along with text identifying as "Human"
			for _, r := range rects {
				gocv.Rectangle(&mat, r, white, 2)
			}
			window.IMShow(mat)
			if window.WaitKey(1) >= 0 {
				break
			}
		}

		if len(rects) == 1 {
			ok, err := cb(&face)
			face.Close()
			if err != nil {
				return err
			}
			if ok {
				return nil
			}
		}

		mat.Close()
	}

	return errors.Errorf("Can not find face")
}

func decodeImage(buf []byte) (gocv.Mat, error) {
	width := 340
	height := 340
	img := image.NewGray(image.Rectangle{Max: image.Point{X: 340, Y: 340}})
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetGray(x, y, color.Gray{Y: uint8(buf[y*width+x])})
		}
	}
	return gocv.ImageGrayToMatGray(img)
}
