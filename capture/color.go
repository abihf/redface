package capture

func hasGoodBlackLevel(img []byte) bool {
	dark := 0
	total := len(img)
	for _, p := range img {
		if p < 80 {
			dark++
		}
	}
	darkness := 100 * dark / total
	return darkness > 10 && darkness < 90
}
