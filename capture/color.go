package capture

import "github.com/blackjack/webcam"

// #include <linux/videodev2.h>
import "C"

const (
	ColorGrey = webcam.PixelFormat(C.V4L2_PIX_FMT_GREY)
	ColorRGB  = webcam.PixelFormat(C.V4L2_PIX_FMT_RGB24)
)

type colorTransformer func([]byte) []byte

var colorFormats = map[webcam.PixelFormat]colorTransformer{
	ColorGrey: grayToRGB,
	ColorRGB:  copyRgb,
}

func copyRgb(img []byte) []byte {
	if len(img) == 0 {
		return img
	}
	rgb := make([]byte, len(img))
	copy(rgb, img)
	return rgb
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

func hasGoodBlackLevel(img []byte) bool {
	dark := 0
	// bright := 0
	total := len(img)
	for _, p := range img {
		if p < 80 {
			dark++
		}
	}
	darkness := 100 * dark / total
	// brightness := 100 * bright / total
	ok := darkness > 5 && darkness < 95
	// fmt.Printf("%v darkness: %d%%\n", ok, darkness)
	return ok
}
