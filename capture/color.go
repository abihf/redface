package capture

import (
	"github.com/blackjack/webcam"
)

// #cgo CFLAGS: -O3 -Wall -O3 -march=native
// #include <linux/videodev2.h>
// #include "color.h"
import "C"

const (
	ColorGrey = webcam.PixelFormat(C.V4L2_PIX_FMT_GREY)
	ColorRGB  = webcam.PixelFormat(C.V4L2_PIX_FMT_RGB24)
	ColorYUYV = webcam.PixelFormat(C.V4L2_PIX_FMT_YUYV)
)

func init() {
	C.init_color_conversion()
}

type colorTransformer func([]byte) []byte

var colorFormats = map[webcam.PixelFormat]colorTransformer{
	ColorGrey: grayToRGB,
	ColorRGB:  copyRgb,
	ColorYUYV: yuv422toRgb,
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
	C.grey_to_rgb((*C.uint8_t)(&gray[0]), C.size_t(len(gray)), (*C.uint8_t)(&rgb[0]))
	return rgb
}

func fourccToString(fourcc webcam.PixelFormat) string {
	return string([]byte{
		byte(fourcc >> 24),
		byte(fourcc >> 16),
		byte(fourcc >> 8),
		byte(fourcc),
	})
}

func yuv422toRgb(yuv []byte) []byte {
	if len(yuv) == 0 {
		return yuv
	}
	rgb := make([]byte, len(yuv)*6/4)
	offset := 0
	for i := 0; i < len(yuv); i += 4 {
		y1 := yuv[i]
		u := yuv[i+1]
		y2 := yuv[i+2]
		v := yuv[i+3]

		rgb[offset+0] = y1 + (v-128)*2/3
		rgb[offset+1] = y1 - (u-128)/3 - (v-128)/3
		rgb[offset+2] = y1 + (u-128)*2/3

		rgb[offset+3] = y2 + (v-128)*2/3
		rgb[offset+4] = y2 - (u-128)/3 - (v-128)/3
		rgb[offset+5] = y2 + (u-128)*2/3

		offset += 6
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
