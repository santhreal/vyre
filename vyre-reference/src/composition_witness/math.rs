//! Sequential mathematical witnesses for semiring linear algebra, polynomial filters, inversion, and hypervectors.

use vyre_spec::Semiring;

/// Sequential mathematical witness for a generalized semiring matrix multiplication: `C = A ⊗ B`.
///
/// Shape: `A` is `m × k`, `B` is `k × n`, output `C` is `m × n`.
///
/// # Panics
/// Panics if input slice dimensions do not match `m * k` and `k * n`.
#[must_use]
pub fn semiring_gemm_witness(
    a: &[u32],
    b: &[u32],
    m: usize,
    n: usize,
    k: usize,
    semiring: Semiring,
) -> Vec<u32> {
    assert_eq!(
        a.len(),
        m * k,
        "A dimension mismatch in semiring GEMM witness"
    );
    assert_eq!(
        b.len(),
        k * n,
        "B dimension mismatch in semiring GEMM witness"
    );

    let zero = semiring.identity();
    let mut c = vec![zero; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = zero;
            for p in 0..k {
                let a_val = a[i * k + p];
                let b_val = b[p * n + j];

                let term = match semiring {
                    Semiring::Real => a_val.wrapping_mul(b_val),
                    Semiring::MinPlus | Semiring::MaxPlus => a_val.saturating_add(b_val),
                    Semiring::MaxTimes => a_val.wrapping_mul(b_val),
                    Semiring::BoolOr => a_val & b_val,
                    Semiring::BoolAnd => a_val | b_val,
                    Semiring::Gf2 => a_val & b_val,
                    Semiring::Lineage => a_val | b_val,
                };

                acc = match semiring {
                    Semiring::Real => acc.wrapping_add(term),
                    Semiring::MinPlus => acc.min(term),
                    Semiring::MaxPlus | Semiring::MaxTimes => acc.max(term),
                    Semiring::BoolOr | Semiring::Lineage => acc | term,
                    Semiring::BoolAnd => acc & term,
                    Semiring::Gf2 => acc ^ term,
                };
            }
            c[i * n + j] = acc;
        }
    }

    c
}

/// Sequential XOR-binding witness for equal-width hypervectors.
#[must_use]
pub fn hypervector_xor_bind_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    lhs.iter().zip(rhs).map(|(&left, &right)| left ^ right).collect()
}

/// Sequential strict-majority bundle witness; ties resolve to zero.
#[must_use]
pub fn hypervector_majority_bundle_witness(hypervectors: &[Vec<u32>]) -> Vec<u32> {
    let width = hypervectors.first().map_or(0, Vec::len);
    assert!(hypervectors.iter().all(|vector| vector.len() == width));
    let threshold = hypervectors.len() / 2;
    (0..width)
        .map(|word| {
            let mut bundled = 0_u32;
            for bit in 0..32 {
                let count = hypervectors
                    .iter()
                    .filter(|vector| vector[word] & (1 << bit) != 0)
                    .count();
                if count > threshold {
                    bundled |= 1 << bit;
                }
            }
            bundled
        })
        .collect()
}

/// Sequential Chebyshev matrix-polynomial filter witness.
#[must_use]
pub fn chebyshev_filter_witness(
    laplacian: &[f32],
    signal: &[f32],
    coefficients: &[f32],
    n: u32,
    k_steps: u32,
) -> Vec<f32> {
    let n = n as usize;
    assert_eq!(laplacian.len(), n * n);
    assert_eq!(signal.len(), n);
    assert!(coefficients.len() > k_steps as usize);
    let multiply = |vector: &[f32]| {
        (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| laplacian[row * n + column] * vector[column])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>()
    };
    let mut previous = signal.to_vec();
    let mut output = previous
        .iter()
        .map(|value| coefficients[0] * value)
        .collect::<Vec<_>>();
    if k_steps == 0 {
        return output;
    }
    let mut current = multiply(&previous);
    for index in 0..n {
        output[index] += coefficients[1] * current[index];
    }
    for step in 2..=k_steps as usize {
        let multiplied = multiply(&current);
        let next = multiplied
            .iter()
            .zip(&previous)
            .map(|(&value, &old)| 2.0 * value - old)
            .collect::<Vec<_>>();
        for index in 0..n {
            output[index] += coefficients[step] * next[index];
        }
        previous = current;
        current = next;
    }
    output
}

/// Sequential per-block Gauss-Jordan inverse witness without pivoting.
#[must_use]
pub fn kfac_block_inverse_witness(blocks: &[f32], num_blocks: u32, n: u32) -> Vec<f32> {
    let n = n as usize;
    let num_blocks = num_blocks as usize;
    let cells = n * n;
    assert_eq!(blocks.len(), num_blocks * cells);
    let mut output = vec![0.0_f32; blocks.len()];
    for block in 0..num_blocks {
        let base = block * cells;
        let mut matrix = blocks[base..base + cells].to_vec();
        let inverse = &mut output[base..base + cells];
        for diagonal in 0..n {
            inverse[diagonal * n + diagonal] = 1.0;
        }
        for pivot in 0..n {
            let pivot_value = matrix[pivot * n + pivot];
            for column in 0..n {
                matrix[pivot * n + column] /= pivot_value;
                inverse[pivot * n + column] /= pivot_value;
            }
            for row in 0..n {
                if row == pivot {
                    continue;
                }
                let factor = matrix[row * n + pivot];
                for column in 0..n {
                    matrix[row * n + column] -= factor * matrix[pivot * n + column];
                    inverse[row * n + column] -= factor * inverse[pivot * n + column];
                }
            }
        }
    }
    output
}

/// Sequential split-conformal threshold witness.
#[must_use]
pub fn conformal_threshold_witness(scores: &[u32], alpha: f64) -> u32 {
    if scores.is_empty() || !(0.0 < alpha && alpha < 1.0) {
        return 0;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_unstable();
    let rank = ((1.0 - alpha) * (sorted.len() as f64 + 1.0)).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}
