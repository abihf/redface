//! AVX2+FMA fast path for `Descriptor::cosine_similarity` — 512-dim f32 dot
//! product and norms, accumulated in f64 like the scalar reference. The
//! scalar fallback is always available; dispatch is cached at first use.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

static HAS_AVX2_FMA: OnceLock<bool> = OnceLock::new();

fn has_avx2_fma() -> bool {
	*HAS_AVX2_FMA.get_or_init(|| {
		#[cfg(target_arch = "x86_64")]
		{
			is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
		}
		#[cfg(not(target_arch = "x86_64"))]
		{
			false
		}
	})
}

/// (a·b, a·a, b·b) over the paired elements of `a` and `b`, in f64.
pub fn dot_and_norms(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
	#[cfg(target_arch = "x86_64")]
	if has_avx2_fma() {
		// SAFETY: AVX2 and FMA support were detected at runtime.
		return unsafe { dot_and_norms_avx2(a, b) };
	}
	dot_and_norms_scalar(a, b)
}

fn dot_and_norms_scalar(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
	let mut dot = 0.0_f64;
	let mut norm_a = 0.0_f64;
	let mut norm_b = 0.0_f64;
	for (&left, &right) in a.iter().zip(b.iter()) {
		let left = f64::from(left);
		let right = f64::from(right);
		dot += left * right;
		norm_a += left * left;
		norm_b += right * right;
	}
	(dot, norm_a, norm_b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_and_norms_avx2(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
	unsafe {
		let mut dot = _mm256_setzero_pd();
		let mut norm_a = _mm256_setzero_pd();
		let mut norm_b = _mm256_setzero_pd();

		let mut a_chunks = a.chunks_exact(8);
		let mut b_chunks = b.chunks_exact(8);
		for (a_chunk, b_chunk) in (&mut a_chunks).zip(&mut b_chunks) {
			let a8 = _mm256_loadu_ps(a_chunk.as_ptr());
			let b8 = _mm256_loadu_ps(b_chunk.as_ptr());
			for (a4, b4) in [
				(_mm256_castps256_ps128(a8), _mm256_castps256_ps128(b8)),
				(_mm256_extractf128_ps(a8, 1), _mm256_extractf128_ps(b8, 1)),
			] {
				let ad = _mm256_cvtps_pd(a4);
				let bd = _mm256_cvtps_pd(b4);
				dot = _mm256_fmadd_pd(ad, bd, dot);
				norm_a = _mm256_fmadd_pd(ad, ad, norm_a);
				norm_b = _mm256_fmadd_pd(bd, bd, norm_b);
			}
		}

		let (mut dot_sum, mut norm_a_sum, mut norm_b_sum) =
			dot_and_norms_scalar(a_chunks.remainder(), b_chunks.remainder());
		let mut lanes = [0.0_f64; 4];
		_mm256_storeu_pd(lanes.as_mut_ptr(), dot);
		dot_sum += lanes.iter().sum::<f64>();
		_mm256_storeu_pd(lanes.as_mut_ptr(), norm_a);
		norm_a_sum += lanes.iter().sum::<f64>();
		_mm256_storeu_pd(lanes.as_mut_ptr(), norm_b);
		norm_b_sum += lanes.iter().sum::<f64>();
		(dot_sum, norm_a_sum, norm_b_sum)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Deterministic pseudo-random values in [-1, 1).
	fn pseudo_random(len: usize) -> Vec<f32> {
		(0..len as u64)
			.map(|i| (i.wrapping_mul(2654435761) % 2000) as f32 / 1000.0 - 1.0)
			.collect()
	}

	#[test]
	fn simd_matches_scalar() {
		let a = pseudo_random(512);
		let b = pseudo_random(512);
		let (dot, norm_a, norm_b) = dot_and_norms(&a, &b);
		let (s_dot, s_norm_a, s_norm_b) = dot_and_norms_scalar(&a, &b);

		assert!((dot - s_dot).abs() < 1e-12, "dot {dot} vs {s_dot}");
		assert!((norm_a - s_norm_a).abs() < 1e-12, "norm_a {norm_a} vs {s_norm_a}");
		assert!((norm_b - s_norm_b).abs() < 1e-12, "norm_b {norm_b} vs {s_norm_b}");
	}

	#[test]
	fn handles_tail_and_empty() {
		let a = pseudo_random(11);
		let b = pseudo_random(11);
		assert_eq!(dot_and_norms(&a[..0], &b[..0]), (0.0, 0.0, 0.0));
		let (dot, norm_a, norm_b) = dot_and_norms(&a, &b);
		let (s_dot, s_norm_a, s_norm_b) = dot_and_norms_scalar(&a, &b);
		assert!((dot - s_dot).abs() < 1e-12);
		assert!((norm_a - s_norm_a).abs() < 1e-12);
		assert!((norm_b - s_norm_b).abs() < 1e-12);
	}

	/// On-demand micro-benchmark:
	/// `cargo test -p redface-core --lib -- --ignored --nocapture`
	#[test]
	#[ignore]
	fn bench_dot_and_norms() {
		use std::hint::black_box;
		use std::time::Instant;

		let a = pseudo_random(512);
		let b = pseudo_random(512);
		for (name, f) in [
			("scalar", dot_and_norms_scalar as fn(&[f32], &[f32]) -> (f64, f64, f64)),
			("dispatched", dot_and_norms as fn(&[f32], &[f32]) -> (f64, f64, f64)),
		] {
			let start = Instant::now();
			for _ in 0..200_000 {
				black_box(f(black_box(&a), black_box(&b)));
			}
			println!("{name}: {:.1} ns/call", start.elapsed().as_secs_f64() * 1e9 / 200_000.0);
		}
	}
}
