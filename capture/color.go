package capture

func hasGoodBlackLevel(img []byte) bool {
	dark := 0
	total := len(img)
	for i := 0; i < total; i++ {
		if img[i] < 80 {
			dark++
		}
	}
	darkness := 100 * dark / total
	return darkness > 10 && darkness < 90
}
