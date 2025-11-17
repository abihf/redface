#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static __m128i M0;
static __m128i M1;
static __m128i M2;

static inline void pack16_gray_to_3x16_rgb(__m128i g, uint8_t *dst)
{
    
    _mm_storeu_si128((__m128i*)(dst +  0), _mm_shuffle_epi8(g, M0));   // 16 bytes
    _mm_storeu_si128((__m128i*)(dst + 16), _mm_shuffle_epi8(g, M1));   // 16 bytes
    _mm_storeu_si128((__m128i*)(dst + 32), _mm_shuffle_epi8(g, M2));   // 16 bytes  => total 48
}

void init_color_conversion(void)
{
    // Masks that turn 16 gray bytes g0..g15 into 48 RGB bytes:
    // chunk A: [g0,g0,g0, g1,g1,g1, ..., g4,g4,g4, g5]
    // chunk B: [g5,g5, g6,g6,g6, ..., g9,g9,g9, g10,g10]
    // chunk C: [g10, g11,g11,g11, ..., g15,g15,g15]
    M0 = _mm_setr_epi8(
        0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5
    );
    M1 = _mm_setr_epi8(
        5,5, 6,6,6, 7,7,7, 8,8,8, 9,9,9, 10,10
    );
    M2 = _mm_setr_epi8(
        10, 11,11,11, 12,12,12, 13,13,13, 14,14,14, 15,15,15
    );
}

void grey_to_rgb(const uint8_t *in, size_t len, uint8_t *out)
{
    size_t i = 0;
    uint8_t *o = out;

// #if defined(__AVX2__)
    // Process 32 pixels (32 gray bytes) -> 96 RGB bytes per iteration
    for (; i + 32 <= len; i += 32, o += 96) {
        __m256i v  = _mm256_loadu_si256((const __m256i*)(in + i));
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);

        // low 16 pixels -> first 48 bytes
        pack16_gray_to_3x16_rgb(lo, o + 0);

        // high 16 pixels -> next 48 bytes
        pack16_gray_to_3x16_rgb(hi, o + 48);
    }
// #endif

    // Process remaining blocks of 16 with SSSE3 (also works even without AVX2)
// #if defined(__SSSE3__)
    for (; i + 16 <= len; i += 16, o += 48) {
        __m128i g = _mm_loadu_si128((const __m128i*)(in + i));
        pack16_gray_to_3x16_rgb(g, o);
    }
// #endif

    // Scalar tail (<=15 pixels)
    for (; i < len; ++i, o += 3) {
        uint8_t g = in[i];
        o[0] = g; o[1] = g; o[2] = g;
    }
}