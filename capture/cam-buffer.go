package capture

import (
	"fmt"
	"os"
	"sync/atomic"

	"github.com/blackjack/webcam"
	"github.com/pkg/errors"
)

type Camera struct {
	frame         chan *Frame
	stopChan      chan bool
	err           error
	device        string
	droppedFrames uint32

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
		if c.droppedFrames > 0 {
			fmt.Printf("Dropped %d frames\n", c.droppedFrames)
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
	var usedFormat webcam.PixelFormat = 0
	for format := range formats {
		if _, ok := colorFormats[format]; ok {
			usedFormat = format
			break
		}
	}

	sizes := cam.GetSupportedFrameSizes(usedFormat)
	for _, size := range sizes {
		fmt.Printf("Supported size: %s\n", size.GetString())
		if size.MaxWidth > width {
			width = size.MaxWidth
		}
		if size.MaxHeight > height {
			height = size.MaxHeight
		}
	}
	_, width, height, err = cam.SetImageFormat(usedFormat, width, height)
	if err != nil {
		return errors.Wrap(err, "Can not set image format")
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
		if err != nil {
			if errors.Is(err, &webcam.Timeout{}) {
				fmt.Fprintf(os.Stderr, "error waiting frame %v", err)
				continue
			} else {
				return errors.Wrap(err, "Wait for frame failed")
			}
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
			c.droppedFrames++
			cam.ReleaseFrame(frameIndex)
			continue
		}
		rgb := colorFormats[usedFormat](frame)
		cam.ReleaseFrame(frameIndex)

		c.frame <- &Frame{
			Buffer: rgb,
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
