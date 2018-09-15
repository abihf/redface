package capture

func hasGoodBlackLevel(img []byte) bool {
	dark := 0
	total := len(img)
	for i := 0; i < total; i++ {
		if img[i] < 80 {
			dark++
		}
	}
	darkness := float64(dark) / float64(total)
	return darkness > 0.1 && darkness < 0.7
}
