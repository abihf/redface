package capture

import (
	"context"
	"fmt"
	"os"
	"sync/atomic"

	"github.com/blackjack/webcam"
	"github.com/pkg/errors"
)

type camBuffer struct {
	frame    chan *Frame
	stopChan chan bool
	err      error

	stopped atomic.Value
}

type Frame struct {
	Buffer []byte
	Width  int
	Height int

	cam   *webcam.Webcam
	index uint32
}

func (frame *Frame) Close() {
	frame.cam.ReleaseFrame(frame.index)
}

func newCamBuffer() *camBuffer {
	c := &camBuffer{
		frame:    make(chan *Frame, 1),
		stopChan: make(chan bool, 1),
		err:      nil,
	}
	c.stopped.Store(false)
	return c
}

type captureConfig struct {
	Device         string
	SkipBlackImage bool
}

func captureNew(ctx context.Context, conf *captureConfig, frameChan chan *Frame) error {
	cam, err := webcam.Open(conf.Device)
	if err != nil {
		return errors.Wrap(err, "Can not open device ")
	}
	defer cam.Close()

	err = cam.StartStreaming()
	if err != nil {
		return errors.Wrap(err, "Can not start streaming")
	}
	defer cam.StopStreaming()

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
			err = cam.WaitForFrame(1)
			switch err.(type) {
			case nil:
			case *webcam.Timeout:
				fmt.Fprint(os.Stderr, err.Error())
				continue
			default:
				return errors.Wrap(err, "Frame wait failed")
			}

			frame, frameIndex, err := cam.GetFrame()
			if err != nil {
				cam.ReleaseFrame(frameIndex)
				return errors.Wrap(err, "Read frame failed")
			}

			if len(frameChan) >= cap(frameChan) || (conf.SkipBlackImage && !hasGoodBlackLevel(frame)) {
				cam.ReleaseFrame(frameIndex)
				continue
			}

			frameChan <- &Frame{
				Buffer: frame,
				Width:  340,
				Height: 340,
				cam:    cam,
				index:  frameIndex,
			}
		}
	}
}

func (c *camBuffer) start(device string) {
	go func() {
		err := c._start(device)
		if err != nil {
			c.err = err
		}
		c.stopChan <- true
	}()
}

func (c *camBuffer) _start(device string) error {
	cam, err := webcam.Open(device)
	if err != nil {
		return errors.Wrap(err, "Can not open device ")
	}
	defer cam.Close()

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
			break
		}

		if len(c.frame) > 0 || !hasGoodBlackLevel(frame) {
			cam.ReleaseFrame(frameIndex)
			continue
		}

		c.frame <- &Frame{
			Buffer: frame,
			Width:  340,
			Height: 340,
			cam:    cam,
			index:  frameIndex,
		}
	}

	return nil
}

func (c *camBuffer) isStopped() bool {
	return c.stopped.Load().(bool)
}

func (c *camBuffer) stop() {
	c.stopped.Store(true)
}

func (f *Frame) Free() {
	f.cam.ReleaseFrame(f.index)
}
