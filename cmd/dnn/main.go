package main

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

func main() {
	deviceID := 0

	// open webcam
	webcam, err := gocv.VideoCaptureDevice(int(deviceID))
	if err != nil {
		fmt.Printf("error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	// open display window
	window := gocv.NewWindow("PVL")

	// prepare input image matrix
	img := gocv.NewMat()
	defer img.Close()

	// prepare grayscale image matrix
	imgGray := gocv.NewMat()
	defer imgGray.Close()

	// color to draw the rect for detected faces
	blue := color.RGBA{0, 0, 255, 0}

	// load PVL FaceDetector to recognize faces
	fd := gocv.ReadNetFromCaffe("face_detector.prototxt", "face_detector.caffemodel")
	defer fd.Close()

	fmt.Printf("start reading camera device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("cannot read device %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// convert image to grayscale for detection
		gocv.CvtColor(img, &imgGray, gocv.ColorBGRToGray)

		// detect faces
		faces := detectFace(fd, imgGray)
		fmt.Printf("found faces %v\n", faces)

		// draw a rectangle around each face on the original image
		for _, face := range faces {
			gocv.Rectangle(&img, face.Rectangle(), blue, 3)
		}

		// show the image in the window, and wait 1 millisecond
		window.IMShow(img)
		if window.WaitKey(1) > 0 {
			break
		}
	}

}

type face struct {
	left, top, right, bottom float32
}

func (f *face) Rectangle() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{
			X: int(f.left),
			Y: int(f.top),
		},
		Max: image.Point{
			X: int(f.right),
			Y: int(f.bottom),
		},
	}
}

func detectFace(net gocv.Net, img gocv.Mat) []face {
	blob := gocv.BlobFromImage(img, 1, image.Pt(300, 300),
		gocv.NewScalar(104, 177, 123, 0), false, false)
	defer blob.Close()

	net.SetInput(blob, "data")

	resOrig := net.Forward("detection_out")
	defer resOrig.Close()

	res := resOrig.Reshape(1, 1)
	defer res.Close()

	count := (res.Cols() * res.Rows()) / 4
	imgWidth := float32(img.Cols())
	imgHeight := float32(img.Rows())
	var faces []face

	for i := 0; i < count; i += 28 {
		confidence := res.GetFloatAt(0, i+8)
		left := res.GetFloatAt(0, i+12) * imgWidth
		top := res.GetFloatAt(0, i+16) * imgHeight
		right := res.GetFloatAt(0, i+20) * imgWidth
		bottom := res.GetFloatAt(0, i*24) * imgHeight
		if confidence > 0 {
			faces = append(faces, face{left, top, right, bottom})
		}
	}
	return faces
}
