//! AVX2 fast path for the score pre-filter in `decode_detections` — the one
//! hand-rolled loop that still runs per frame at scale (~14,800 scores over
//! the three stride branches). The scalar fallback is always available;
//! dispatch is cached at first use.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

static HAS_AVX2: OnceLock<bool> = OnceLock::new();

fn has_avx2() -> bool {
	*HAS_AVX2.get_or_init(|| {
		#[cfg(target_arch = "x86_64")]
		{
			is_x86_feature_detected!("avx2")
		}
		#[cfg(not(target_arch = "x86_64"))]
		{
			false
		}
	})
}

/// Appends indices of all scores at or above `threshold` (ascending) to `out`.
pub fn above_threshold(scores: &[f32], threshold: f32, out: &mut Vec<u32>) {
	out.clear();
	#[cfg(target_arch = "x86_64")]
	if has_avx2() {
		// SAFETY: AVX2 support was detected at runtime.
		unsafe { above_threshold_avx2(scores, threshold, out) };
		return;
	}
	above_threshold_scalar(scores, threshold, out);
}

fn above_threshold_scalar(scores: &[f32], threshold: f32, out: &mut Vec<u32>) {
	for (index, &score) in scores.iter().enumerate() {
		if score >= threshold {
			out.push(index as u32);
		}
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn above_threshold_avx2(scores: &[f32], threshold: f32, out: &mut Vec<u32>) {
	unsafe {
		let broadcast = _mm256_set1_ps(threshold);
		let mut chunks = scores.chunks_exact(8);
		let mut base = 0_u32;
		for chunk in &mut chunks {
			let values = _mm256_loadu_ps(chunk.as_ptr());
			let mut bits = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(values, broadcast)) as u32;
			while bits != 0 {
				out.push(base + bits.trailing_zeros());
				bits &= bits - 1;
			}
			base += 8;
		}
		for (offset, &score) in chunks.remainder().iter().enumerate() {
			if score >= threshold {
				out.push(base + offset as u32);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn collect(scores: &[f32], threshold: f32) -> Vec<u32> {
		let mut out = Vec::new();
		above_threshold(scores, threshold, &mut out);
		out
	}

	#[test]
	fn above_threshold_matches_scalar() {
		// Deterministic pseudo-random scores in [0, 1), length chosen with a
		// tail (not a multiple of 8).
		let mut scores: Vec<f32> = (0..14_803_u32)
			.map(|i| (u64::from(i).wrapping_mul(2654435761) % 1000) as f32 / 1000.0)
			.collect();
		// Exact-threshold and just-below values exercise the >= predicate.
		scores[10] = 0.5;
		scores[11] = 0.499_999_97;
		scores[14_801] = 0.5;

		for threshold in [0.5, 0.0, 1.0, 0.123] {
			let mut expected = Vec::new();
			above_threshold_scalar(&scores, threshold, &mut expected);
			assert_eq!(collect(&scores, threshold), expected, "threshold {threshold}");
		}
	}

	#[test]
	fn above_threshold_handles_edge_lengths() {
		assert_eq!(collect(&[], 0.5), Vec::<u32>::new());
		assert_eq!(collect(&[0.5], 0.5), vec![0]);
		assert_eq!(collect(&[0.4; 7], 0.5), Vec::<u32>::new());
		let all: Vec<u32> = (0..8).collect();
		assert_eq!(collect(&[0.9; 8], 0.5), all);
	}

	#[test]
	fn above_threshold_reuses_buffer() {
		let mut out = vec![99; 4];
		above_threshold(&[0.1, 0.9, 0.2], 0.5, &mut out);
		assert_eq!(out, vec![1]);
	}
}
