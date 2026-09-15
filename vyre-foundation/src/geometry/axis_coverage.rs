//! Logical points a flat domain covers on each axis of a workgroup.
//!
//! A region domain is a flat element count: one declared value with one extent.
//! A workgroup is a shape. When the shape occupies more than the x axis, the
//! two disagree about how many points exist, and the launch derived from them
//! covers the domain on x while every lane on y and z reads the same handful of
//! ids. A program that addresses two axes then computes only the first
//! `workgroup[1]` columns of every row, which is a wrong answer rather than a
//! narrow launch.
//!
//! One decomposition answers it for every consumer: a square for a two-axis
//! shape, a cube for a three-axis one, with the remainder on the last axis the
//! shape occupies. Both the compiler that records a launch and the driver that
//! infers one for a program no artifact governs read this function, so an
//! artifact launch and an inferred launch of the same program have the same
//! shape.

/// Logical points a flat domain of `points` covers on each axis of `workgroup`.
///
/// The product of the returned axes is at least `points`, so a launch covering
/// them runs every invocation the domain states. A zero domain covers one
/// point: a launch of nothing is a shape no consumer can validate, and every
/// caller here has one region to cover.
#[must_use]
pub fn axis_coverage(points: u64, workgroup: [u32; 3]) -> [u64; 3] {
    let points = points.max(1);
    if workgroup[1] <= 1 && workgroup[2] <= 1 {
        return [points, 1, 1];
    }
    if workgroup[2] <= 1 {
        let side = ceil_sqrt(points);
        return [side, points.div_ceil(side), 1];
    }
    let side = ceil_cuberoot(points);
    let plane = side.saturating_mul(side).max(1);
    [side, side, points.div_ceil(plane)]
}

/// Smallest `n` with `n * n >= value`.
fn ceil_sqrt(value: u64) -> u64 {
    if value <= 1 {
        return 1;
    }
    let mut lo = 1_u64;
    let mut hi = 1_u64 << 32;
    while lo < hi {
        let mid = lo + ((hi - lo) / 2);
        match mid.checked_mul(mid) {
            Some(square) if square < value => lo = mid + 1,
            _ => hi = mid,
        }
    }
    lo
}

/// Smallest `n` with `n * n * n >= value`.
fn ceil_cuberoot(value: u64) -> u64 {
    if value <= 1 {
        return 1;
    }
    let mut lo = 1_u64;
    let mut hi = 1_u64 << 22;
    while lo < hi {
        let mid = lo + ((hi - lo) / 2);
        match mid
            .checked_mul(mid)
            .and_then(|square| square.checked_mul(mid))
        {
            Some(cube) if cube < value => lo = mid + 1,
            _ => hi = mid,
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::axis_coverage;

    /// A one-axis shape leaves the domain where it is stated.
    #[test]
    fn a_single_axis_workgroup_covers_the_flat_domain_on_x() {
        assert_eq!(axis_coverage(4096, [256, 1, 1]), [4096, 1, 1]);
        assert_eq!(axis_coverage(1, [1, 1, 1]), [1, 1, 1]);
        assert_eq!(axis_coverage(0, [64, 1, 1]), [1, 1, 1]);
    }

    /// Every axis the workgroup occupies carries points, and the product covers
    /// the domain.
    #[test]
    fn a_multi_axis_workgroup_spreads_the_domain_over_the_axes_it_occupies() {
        for points in [1_u64, 2, 15, 16, 1024, 65_536, 65_537, 1_000_003] {
            for workgroup in [[16, 16, 1], [8, 4, 1], [4, 4, 4], [16, 2, 8]] {
                let coverage = axis_coverage(points, workgroup);
                let product = coverage[0] * coverage[1] * coverage[2];
                assert!(
                    product >= points,
                    "coverage {coverage:?} for {points} points at workgroup {workgroup:?} covers only {product}"
                );
                for (axis, extent) in coverage.iter().enumerate() {
                    assert!(
                        *extent >= 1,
                        "coverage {coverage:?} states no point on axis {axis}"
                    );
                }
                if workgroup[2] <= 1 {
                    assert_eq!(
                        coverage[2], 1,
                        "a two-axis workgroup states no z coverage: {coverage:?}"
                    );
                }
            }
        }
    }

    /// The decomposition is the smallest square or cube that holds the domain,
    /// so a launch runs at most one workgroup of idle lanes per axis.
    #[test]
    fn the_decomposition_is_the_smallest_square_or_cube_that_holds_the_domain() {
        assert_eq!(axis_coverage(1024, [16, 16, 1]), [32, 32, 1]);
        assert_eq!(axis_coverage(65_536, [16, 16, 1]), [256, 256, 1]);
        assert_eq!(axis_coverage(65_537, [16, 16, 1]), [257, 256, 1]);
        assert_eq!(axis_coverage(1000, [4, 4, 4]), [10, 10, 10]);
        assert_eq!(axis_coverage(1001, [4, 4, 4]), [11, 11, 9]);
    }

    /// The decomposition is exact where the square and cube roots are, so a
    /// domain at the boundary covers no extra plane.
    #[test]
    fn the_decomposition_is_exact_at_large_roots() {
        assert_eq!(axis_coverage((1 << 32) - 1, [1, 2, 1]), [65_536, 65_536, 1]);
        assert_eq!(axis_coverage(1 << 32, [1, 2, 1]), [65_536, 65_536, 1]);
        let cube = 2_642_245_u64.pow(3);
        assert_eq!(
            axis_coverage(cube, [2, 2, 2]),
            [2_642_245, 2_642_245, 2_642_245]
        );
        assert_eq!(
            axis_coverage(cube - 1, [2, 2, 2]),
            [2_642_245, 2_642_245, 2_642_245]
        );
    }
}
