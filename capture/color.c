#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static __m128i M0;
static __m128i M1;
static __m128i M2;

#if defined(__AVX512VBMI__)
static __m512i IDX0_512;
static __m512i IDX1_512;
static __m512i IDX2_512;

#elif defined(__AVX512BW__)
static __m512i M0_512;
static __m512i M1_512;
static __m512i M2_512;
#endif

void init_color_conversion(void)
{
    // Masks that turn 16 gray bytes g0..g15 into 48 RGB bytes:
    M0 = _mm_setr_epi8(
        0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5
    );
    M1 = _mm_setr_epi8(
        5,5, 6,6,6, 7,7,7, 8,8,8, 9,9,9, 10,10
    );
    M2 = _mm_setr_epi8(
        10, 11,11,11, 12,12,12, 13,13,13, 14,14,14, 15,15,15
    );

#if defined(__AVX512VBMI__)
    // Build index vectors for vpermb:
    //
    // We want out[3*i+0..2] = g[i]  for i=0..63.
    // We produce 3 chunks of 64 bytes each:
    //   chunk c in {0,1,2}, position p in {0..63}
    //   global output index j = c*64 + p
    //   pixel index = j / 3
    //
    // So idx_c[p] = (c*64 + p)/3
    uint8_t idx0_bytes[64];
    uint8_t idx1_bytes[64];
    uint8_t idx2_bytes[64];
    for (int p = 0; p < 64; ++p) {
        idx0_bytes[p] = (uint8_t)((0*64 + p) / 3);
        idx1_bytes[p] = (uint8_t)((1*64 + p) / 3);
        idx2_bytes[p] = (uint8_t)((2*64 + p) / 3);
    }

    IDX0_512 = _mm512_loadu_si512((const __m512i*)idx0_bytes);
    IDX1_512 = _mm512_loadu_si512((const __m512i*)idx1_bytes);
    IDX2_512 = _mm512_loadu_si512((const __m512i*)idx2_bytes);
#elif defined(__AVX512BW__)
    // Broadcast the 16-byte masks to all four 128-bit lanes of a 512-bit register.
    M0_512 = _mm512_broadcast_i32x4(M0);
    M1_512 = _mm512_broadcast_i32x4(M1);
    M2_512 = _mm512_broadcast_i32x4(M2);
#endif
}

static inline void pack16_gray_to_3x16_rgb(__m128i g, uint8_t *dst)
{
    _mm_storeu_si128((__m128i*)(dst +  0), _mm_shuffle_epi8(g, M0));   // 16 bytes
    _mm_storeu_si128((__m128i*)(dst + 16), _mm_shuffle_epi8(g, M1));   // 16 bytes
    _mm_storeu_si128((__m128i*)(dst + 32), _mm_shuffle_epi8(g, M2));   // 16 bytes  => total 48
}

void grey_to_rgb(const uint8_t *in, size_t len, uint8_t *out)
{
    size_t i = 0;
    uint8_t *o = out;
#if defined(__AVX512VBMI__)
    // AVX-512VBMI: 64 pixels -> 192 RGB bytes per iteration
    for (; i + 64 <= len; i += 64, o += 192) {
        __m512i g = _mm512_loadu_si512((const void*)(in + i));

        // 3 × 64-byte stores = 192 bytes = 64 pixels × 3 bytes/pixel
        _mm512_storeu_si512((__m512i*)(o +   0), _mm512_permutexvar_epi8(IDX0_512, g));
        _mm512_storeu_si512((__m512i*)(o +  64), _mm512_permutexvar_epi8(IDX1_512, g));
        _mm512_storeu_si512((__m512i*)(o + 128), _mm512_permutexvar_epi8(IDX2_512, g));
    }
#elif defined(__AVX512BW__)
    // AVX-512BW: process 64 pixels (64 bytes) -> 192 RGB bytes per iteration
    // Input layout in v: [g0..g15 | g16..g31 | g32..g47 | g48..g63] per 128-bit lane.
    for (; i + 64 <= len; i += 64, o += 192) {
        __m512i v  = _mm512_loadu_si512((const void*)(in + i));

        // Each shuffle uses the same lane-local mask as the SSE version.
        __m512i r0 = _mm512_shuffle_epi8(v, M0_512);
        __m512i r1 = _mm512_shuffle_epi8(v, M1_512);
        __m512i r2 = _mm512_shuffle_epi8(v, M2_512);

        // lane 0
        _mm_storeu_si128((__m128i*)(o +   0), _mm512_castsi512_si128(r0));
        _mm_storeu_si128((__m128i*)(o +  16), _mm512_castsi512_si128(r1));
        _mm_storeu_si128((__m128i*)(o +  32), _mm512_castsi512_si128(r2));

        // lane 1
        _mm_storeu_si128((__m128i*)(o +  48), _mm512_extracti32x4_epi32(r0, 1));
        _mm_storeu_si128((__m128i*)(o +  64), _mm512_extracti32x4_epi32(r1, 1));
        _mm_storeu_si128((__m128i*)(o +  80), _mm512_extracti32x4_epi32(r2, 1));

        // lane 2
        _mm_storeu_si128((__m128i*)(o +  96), _mm512_extracti32x4_epi32(r0, 2));
        _mm_storeu_si128((__m128i*)(o + 112), _mm512_extracti32x4_epi32(r1, 2));
        _mm_storeu_si128((__m128i*)(o + 128), _mm512_extracti32x4_epi32(r2, 2));

        // lane 3
        _mm_storeu_si128((__m128i*)(o + 144), _mm512_extracti32x4_epi32(r0, 3));
        _mm_storeu_si128((__m128i*)(o + 160), _mm512_extracti32x4_epi32(r1, 3));
        _mm_storeu_si128((__m128i*)(o + 176), _mm512_extracti32x4_epi32(r2, 3));

    }
#endif

#if defined(__AVX2__)
    // AVX2: process 32 pixels (32 gray bytes) -> 96 RGB bytes per iteration
    for (; i + 32 <= len; i += 32, o += 96) {
        __m256i v  = _mm256_loadu_si256((const __m256i*)(in + i));
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);

        // low 16 pixels -> first 48 bytes
        pack16_gray_to_3x16_rgb(lo, o + 0);

        // high 16 pixels -> next 48 bytes
        pack16_gray_to_3x16_rgb(hi, o + 48);
    }
#endif

#if defined(__SSSE3__)
    // Process remaining blocks of 16 with SSSE3
    for (; i + 16 <= len; i += 16, o += 48) {
        __m128i g = _mm_loadu_si128((const __m128i*)(in + i));
        pack16_gray_to_3x16_rgb(g, o);
    }
#endif

    // Scalar tail (<=15 pixels)
    for (; i < len; ++i, o += 3) {
        uint8_t g = in[i];
        o[0] = g; o[1] = g; o[2] = g;
    }
}
