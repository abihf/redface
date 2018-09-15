package capture

type Processor func(img []byte, width, height int) (bool, error)

type Option struct {
	Device string
}

func Capture(opt *Option, processor Processor) error {
	cam := newCamBuffer()
	cam.start(opt.Device)
	defer cam.stop()

	for {
		select {
		case <-cam.stopChan:
			return cam.err

		case frame := <-cam.frame:
			cont, err := processor(frame, 340, 340)
			if err != nil {
				return err
			}

			if !cont {
				return nil
			}

		}
	}
}
