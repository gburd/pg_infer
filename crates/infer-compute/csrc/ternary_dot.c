/**
 * Ternary dot product kernel for BitNet b1.58 gate-KNN scoring.
 *
 * Packing: 4 ternary values per byte, 2 bits each (I2_S format).
 *   0b00 = 0 (skip), 0b01 = +1 (add), 0b10 = -1 (subtract)
 *
 * With -O3 -mavx2 (x86_64) or -O3 -march=armv8.2-a (aarch64) the
 * compiler auto-vectorizes the inner loop. The branch-free
 * (v==1) - (v==2) pattern compiles to comparison + subtraction.
 */

#include <stdint.h>
#include <stddef.h>

/**
 * Single-row ternary dot product: packed_row[bytes] × x[hidden] → float.
 */
static float ternary_dot_c(
    const uint8_t* packed_row,
    const float* x,
    size_t hidden
) {
    float acc = 0.0f;
    size_t bytes = hidden / 4;

    for (size_t i = 0; i < bytes; i++) {
        uint8_t byte = packed_row[i];
        size_t base = i * 4;
        uint8_t v0 = byte & 0x03;
        uint8_t v1 = (byte >> 2) & 0x03;
        uint8_t v2 = (byte >> 4) & 0x03;
        uint8_t v3 = (byte >> 6) & 0x03;

        /* Branch-free: (v==1)*x - (v==2)*x = ((v==1) - (v==2)) * x */
        acc += (float)((int)(v0 == 1) - (int)(v0 == 2)) * x[base];
        acc += (float)((int)(v1 == 1) - (int)(v1 == 2)) * x[base + 1];
        acc += (float)((int)(v2 == 1) - (int)(v2 == 2)) * x[base + 2];
        acc += (float)((int)(v3 == 1) - (int)(v3 == 2)) * x[base + 3];
    }
    return acc;
}

/**
 * Multi-row ternary matvec: compute scores for all rows.
 * packed[num_rows * bytes_per_row] × x[hidden] → scores[num_rows]
 */
void ternary_matvec_c(
    const uint8_t* packed,
    const float* x,
    float* scores,
    size_t num_rows,
    size_t hidden
) {
    size_t bytes_per_row = hidden / 4;
    for (size_t r = 0; r < num_rows; r++) {
        scores[r] = ternary_dot_c(packed + r * bytes_per_row, x, hidden);
    }
}
