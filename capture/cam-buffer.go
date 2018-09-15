package capture

import (
	"fmt"
	"os"
	"sync/atomic"

	"github.com/blackjack/webcam"
	"github.com/pkg/errors"
)

type camBuffer struct {
	frame    chan []byte
	stopChan chan bool
	err      error

	stopped atomic.Value
}

func newCamBuffer() *camBuffer {
	c := &camBuffer{
		frame:    make(chan []byte, 1),
		stopChan: make(chan bool, 1),
		err:      nil,
	}
	c.stopped.Store(false)
	return c
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
		if c.isRunning() {
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

		if c.isRunning() {
			break
		}

		frame, err := cam.ReadFrame()
		if err != nil {
			return errors.Wrap(err, "Read frame failed")
		}

		if c.isRunning() {
			break
		}

		if len(c.frame) > 0 {
			continue
		}

		if !hasGoodBlackLevel(frame) {
			continue
		}

		c.frame <- frame
	}

	return nil
}

func (c *camBuffer) isRunning() bool {
	return c.stopped.Load().(bool)
}

func (c *camBuffer) stop() {
	c.stopped.Store(true)
}
