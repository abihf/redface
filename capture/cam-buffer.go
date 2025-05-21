package capture

import (
	"fmt"
	"os"
	"sync/atomic"

	"github.com/blackjack/webcam"
	"github.com/pkg/errors"
)

type Camera struct {
	frame    chan *Frame
	stopChan chan bool
	err      error
	device   string

	stopped atomic.Bool
}

func Open(device string) *Camera {
	c := &Camera{
		frame:    make(chan *Frame, 1),
		stopChan: make(chan bool, 1),
		err:      nil,
		device:   device,
	}
	c.stopped.Store(false)
	return c
}

func (c *Camera) Err() error {
	return c.err
}

func (c *Camera) Stream() chan *Frame {
	go func() {
		defer close(c.frame)
		err := c.start(c.device)
		if err != nil {
			c.err = err
		}
	}()
	return c.frame
}

type Frame struct {
	Buffer []byte
	Width  uint32
	Height uint32

	cam   *webcam.Webcam
	index uint32
}

func (c *Camera) start(device string) error {
	cam, err := webcam.Open(device)
	if err != nil {
		return errors.Wrap(err, "Can not open device ")
	}
	defer cam.Close()

	formats := cam.GetSupportedFormats()
	if len(formats) == 0 {
		return errors.New("No supported formats found")
	}
	width := uint32(0)
	height := uint32(0)
	for format := range formats {
		sizes := cam.GetSupportedFrameSizes(format)
		for _, size := range sizes {
			fmt.Printf("Supported size: %d-%d x %d-%d\n", size.MinWidth, size.MaxWidth, size.MinHeight, size.MaxHeight)
			if size.MaxWidth > width {
				width = size.MaxWidth
			}
			if size.MaxHeight > height {
				height = size.MaxHeight
			}
		}
		cam.SetImageFormat(format, width, height)
		break
	}

	err = cam.StartStreaming()
	if err != nil {
		return errors.Wrap(err, "Can not start streaming")
	}

	for {
		if c.isStopped() {
			break
		}

		err = cam.WaitForFrame(1)
		switch err.(type) {
		case nil:
		case *webcam.Timeout:
			fmt.Fprint(os.Stderr, err.Error())
			continue
		default:
			return errors.Wrap(err, "Frame wait failed")
		}

		if c.isStopped() {
			break
		}

		frame, frameIndex, err := cam.GetFrame()
		if err != nil {
			cam.ReleaseFrame(frameIndex)
			return errors.Wrap(err, "Read frame failed")
		}

		if c.isStopped() {
			cam.ReleaseFrame(frameIndex)
			break
		}

		if len(c.frame) > 0 || !hasGoodBlackLevel(frame) {
			cam.ReleaseFrame(frameIndex)
			continue
		}

		c.frame <- &Frame{
			Buffer: frame,
			Width:  width,
			Height: height,
			cam:    cam,
			index:  frameIndex,
		}
	}

	return nil
}

func (c *Camera) isStopped() bool {
	return c.stopped.Load()
}

func (c *Camera) Close() {
	c.stopped.Store(true)
}

func (f *Frame) Free() {
	f.cam.ReleaseFrame(f.index)
}
