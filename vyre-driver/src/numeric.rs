//! Backend-neutral numeric boundary conversions.
//!
//! Concrete GPU backends cross the same host/API boundaries: host sizes become
//! API `u64`s, high-resolution timers become telemetry `u64`s, and device
//! timestamp deltas arrive as rounded floating-point nanoseconds. This module is
//! the single policy for those lossy or fallible conversions; backend crates add
//! only the backend label that makes the diagnostic actionable.

use std::time::Instant;

use crate::BackendError;

/// Integer basis-point denominator: 10_000 bps = 100%.
pub const BASIS_POINTS_DENOMINATOR: u32 = 10_000;

/// Backend-bound numeric conversion policy.
///
/// Backends should keep their label in one constant of this type instead of
/// cloning one local wrapper per numeric helper. The free functions below remain
/// available for backend-neutral callers and tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendNumericPolicy {
    backend: &'static str,
}

impl BackendNumericPolicy {
    /// Create a numeric policy that annotates diagnostics with `backend`.
    #[must_use]
    pub const fn new(backend: &'static str) -> Self {
        Self { backend }
    }

    /// Return the backend label used in numeric diagnostics.
    #[must_use]
    pub const fn backend(self) -> &'static str {
        self.backend
    }

    /// Convert a host `usize` to a backend/API `u64`.
    ///
    /// # Errors
    /// Returns [`BackendError::InvalidProgram`] when the value cannot fit in
    /// the backend/API boundary type.
    pub fn usize_to_u64(self, value: usize, label: &str) -> Result<u64, BackendError> {
        usize_to_u64(value, label, self.backend)
    }

    /// Convert a wide counter to telemetry `u64`.
    ///
    /// # Errors
    /// Returns [`BackendError::InvalidProgram`] when the counter does not fit in
    /// telemetry storage.
    pub fn u128_to_u64(self, value: u128, label: &str) -> Result<u64, BackendError> {
        u128_to_u64(value, label, self.backend)
    }

    /// Convert elapsed wall-clock time to telemetry nanoseconds.
    ///
    /// # Errors
    /// Returns [`BackendError::InvalidProgram`] when the elapsed nanoseconds
    /// cannot fit in telemetry storage.
    pub fn elapsed_nanos_u64(self, started: Instant, label: &str) -> Result<u64, BackendError> {
        elapsed_nanos_u64(started, label, self.backend)
    }

    /// Round a finite floating-point nanosecond value into telemetry storage.
    ///
    /// # Errors
    /// Returns [`BackendError::InvalidProgram`] when the rounded value is
    /// negative, non-finite, or too large for telemetry storage.
    pub fn rounded_f64_to_u64(self, value: f64, label: &str) -> Result<u64, BackendError> {
        rounded_f64_to_u64(value, label, self.backend)
    }

    /// Compute `part / whole` as floor basis points in a `u32` telemetry domain.
    #[must_use]
    pub fn ratio_basis_points_u64(
        self,
        part: u64,
        whole: u64,
        denominator_zero_value: u32,
        label: &str,
    ) -> u32 {
        ratio_basis_points_u64(part, whole, denominator_zero_value, label, self.backend)
    }

    /// Compute `part / whole` as floor basis points in a `u64` telemetry domain.
    #[must_use]
    pub fn ratio_basis_points_u64_wide(
        self,
        part: u64,
        whole: u64,
        denominator_zero_value: u64,
        label: &str,
    ) -> u64 {
        ratio_basis_points_u64_wide(part, whole, denominator_zero_value, label, self.backend)
    }

    /// Compute `part / whole` as floor parts-per-million.
    #[must_use]
    pub fn ratio_parts_per_million_u64(
        self,
        part: u64,
        whole: u64,
        denominator_zero_value: u32,
        label: &str,
    ) -> u32 {
        ratio_parts_per_million_u64(part, whole, denominator_zero_value, label, self.backend)
    }

    /// Compose two basis-point multipliers into a `u32` result.
    #[must_use]
    pub fn compose_basis_points_u32(self, left: u32, right: u32, label: &str) -> u32 {
        compose_basis_points_u32(left, right, label, self.backend)
    }

    /// Apply rounded basis-point scaling with optional high clamp.
    #[must_use]
    pub fn scale_u64_by_basis_points_round_clamped(
        self,
        base: u64,
        scale_bps: u32,
        zero_scale_value: u64,
        max_scale_bps: u32,
        label: &str,
    ) -> u64 {
        scale_u64_by_basis_points_round_clamped(
            base,
            scale_bps,
            zero_scale_value,
            max_scale_bps,
            label,
            self.backend,
        )
    }

    /// Apply floor basis-point scaling with a lower bound.
    #[must_use]
    pub fn scale_u64_by_basis_points_floor_min(
        self,
        base: u64,
        scale_bps: u32,
        min_value: u64,
        label: &str,
    ) -> u64 {
        scale_u64_by_basis_points_floor_min(base, scale_bps, min_value, label, self.backend)
    }

    /// Convert finite non-negative floating-point telemetry to `u32` by truncation.
    #[must_use]
    pub fn finite_f64_to_u32_trunc(self, value: f64, label: &str) -> u32 {
        finite_f64_to_u32_trunc(value, label, self.backend)
    }

    /// Convert finite non-negative floating-point telemetry to rounded `u32`.
    #[must_use]
    pub fn finite_f64_to_u32_round(self, value: f64, label: &str) -> u32 {
        finite_f64_to_u32_round(value, label, self.backend)
    }

    /// Convert a finite floating-point ratio into floor basis points.
    #[must_use]
    pub fn finite_f64_ratio_basis_points_trunc(
        self,
        numerator: f64,
        denominator: f64,
        invalid_numerator_value: u32,
        invalid_denominator_value: u32,
        label: &str,
    ) -> u32 {
        finite_f64_ratio_basis_points_trunc(
            numerator,
            denominator,
            invalid_numerator_value,
            invalid_denominator_value,
            label,
            self.backend,
        )
    }

    /// Convert a finite floating-point ratio into rounded basis points.
    #[must_use]
    pub fn finite_f64_ratio_basis_points_round(
        self,
        numerator: f64,
        denominator: f64,
        invalid_numerator_value: u32,
        invalid_denominator_value: u32,
        label: &str,
    ) -> u32 {
        finite_f64_ratio_basis_points_round(
            numerator,
            denominator,
            invalid_numerator_value,
            invalid_denominator_value,
            label,
            self.backend,
        )
    }

    /// Convert a finite scalar where `1.0 == 10_000 bps` into floor basis points.
    #[must_use]
    pub fn finite_f64_unit_basis_points_trunc(
        self,
        value: f64,
        invalid_value: u32,
        label: &str,
    ) -> u32 {
        finite_f64_unit_basis_points_trunc(value, invalid_value, label, self.backend)
    }

    /// Compute `ceil(value / divisor)` in `u64`, returning `None` for zero
    /// divisors or arithmetic overflow.
    #[must_use]
    pub fn checked_ceil_div_u64(self, value: u64, divisor: u64) -> Option<u64> {
        checked_ceil_div_u64(value, divisor)
    }

    /// Multiply three `u32` launch dimensions into a `u64` without wraparound.
    #[must_use]
    pub fn checked_dim_product_u64(self, dims: [u32; 3]) -> Option<u64> {
        checked_dim_product_u64(dims)
    }

    /// Multiply three `u32` launch dimensions into a `u32` without wraparound.
    #[must_use]
    pub fn checked_dim_product_u32(self, dims: [u32; 3]) -> Option<u32> {
        checked_dim_product_u32(dims)
    }

    /// Align `value` upward to `alignment`, after applying `min_value`.
    ///
    /// # Errors
    /// Returns [`BackendError::InvalidProgram`] when `alignment` is zero or the
    /// padded value would overflow `u64`.
    pub fn align_up_u64(
        self,
        value: u64,
        alignment: u64,
        min_value: u64,
        label: &str,
    ) -> Result<u64, BackendError> {
        align_up_u64(value, alignment, min_value, label, self.backend)
    }

    /// Align `value` upward to `alignment`, after applying `min_value`.
    ///
    /// # Errors
    /// Returns [`BackendError::InvalidProgram`] when `alignment` is zero or the
    /// padded value would overflow `usize`.
    pub fn align_up_usize(
        self,
        value: usize,
        alignment: usize,
        min_value: usize,
        label: &str,
    ) -> Result<usize, BackendError> {
        align_up_usize(value, alignment, min_value, label, self.backend)
    }
}

/// Convert a host `usize` to a backend/API `u64`.
///
/// # Errors
/// Returns [`BackendError::InvalidProgram`] when the value cannot fit in the
/// backend/API boundary type.
pub fn usize_to_u64(value: usize, label: &str, backend: &str) -> Result<u64, BackendError> {
    u64::try_from(value).map_err(|source| BackendError::InvalidProgram {
        fix: format!(
            "Fix: {backend} {label} cannot fit u64: {source}; split the workload before crossing the host/device boundary."
        ),
    })
}

/// Convert a wide counter to telemetry `u64`.
///
/// # Errors
/// Returns [`BackendError::InvalidProgram`] when the counter does not fit in
/// telemetry storage.
pub fn u128_to_u64(value: u128, label: &str, backend: &str) -> Result<u64, BackendError> {
    u64::try_from(value).map_err(|source| BackendError::InvalidProgram {
        fix: format!(
            "Fix: {backend} {label} cannot fit u64: {source}; split the dispatch before telemetry overflows."
        ),
    })
}

/// Convert elapsed wall-clock time to telemetry nanoseconds.
///
/// # Errors
/// Returns [`BackendError::InvalidProgram`] when the elapsed nanoseconds cannot
/// fit in telemetry storage.
pub fn elapsed_nanos_u64(
    started: Instant,
    label: &str,
    backend: &str,
) -> Result<u64, BackendError> {
    u128_to_u64(started.elapsed().as_nanos(), label, backend)
}

/// Round a finite floating-point nanosecond value into telemetry storage.
///
/// # Errors
/// Returns [`BackendError::InvalidProgram`] when the rounded value is negative,
/// non-finite, or too large for telemetry storage.
pub fn rounded_f64_to_u64(value: f64, label: &str, backend: &str) -> Result<u64, BackendError> {
    let rounded = value.round();
    if !rounded.is_finite() || rounded < 0.0 || rounded > u64::MAX as f64 {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {backend} {label} value {value} cannot fit u64 after rounding; inspect device timing and split the dispatch before telemetry overflows."
            ),
        });
    }
    u64::try_from(rounded as u128).map_err(|source| BackendError::InvalidProgram {
        fix: format!(
            "Fix: {backend} {label} rounded value cannot fit u64: {source}; inspect device timing and split the dispatch before telemetry overflows."
        ),
    })
}

/// Compute `part / whole` as floor basis points with explicit zero-denominator
/// policy and saturating telemetry overflow.
///
/// Release-path planners use the same ratio encoding for memory pressure,
/// readback savings, and device-side compaction. Keeping the arithmetic here
/// prevents each backend module from carrying its own unchecked `as u32` cast.
#[must_use]
pub fn ratio_basis_points_u64(
    part: u64,
    whole: u64,
    denominator_zero_value: u32,
    label: &str,
    backend: &str,
) -> u32 {
    let value = ratio_basis_points_u64_wide(
        part,
        whole,
        u64::from(denominator_zero_value),
        label,
        backend,
    );
    if value > u64::from(u32::MAX) {
        tracing::error!(
            "{backend} {label} basis-points value exceeded u32. Fix: shard or normalize the telemetry domain before release-path planning."
        );
        return u32::MAX;
    }
    value as u32
}

/// Compute `part / whole` as floor basis points in a `u64` telemetry domain
/// with explicit zero-denominator policy and loud overflow pinning.
#[must_use]
pub fn ratio_basis_points_u64_wide(
    part: u64,
    whole: u64,
    denominator_zero_value: u64,
    label: &str,
    backend: &str,
) -> u64 {
    if whole == 0 {
        return denominator_zero_value;
    }
    let value = (u128::from(part) * u128::from(BASIS_POINTS_DENOMINATOR)) / u128::from(whole);
    if value > u128::from(u64::MAX) {
        tracing::error!(
            "{backend} {label} basis-points value exceeded u64. Fix: shard or normalize the telemetry domain before release-path planning."
        );
        return u64::MAX;
    }
    value as u64
}

/// Compute `part / whole` as floor parts-per-million with explicit
/// zero-denominator policy and loud `u32` overflow pinning.
#[must_use]
pub fn ratio_parts_per_million_u64(
    part: u64,
    whole: u64,
    denominator_zero_value: u32,
    label: &str,
    backend: &str,
) -> u32 {
    if whole == 0 {
        return denominator_zero_value;
    }
    let value = (u128::from(part) * 1_000_000) / u128::from(whole);
    if value > u128::from(u32::MAX) {
        tracing::error!(
            "{backend} {label} parts-per-million value exceeded u32. Fix: shard or normalize telemetry before release-path planning."
        );
        return u32::MAX;
    }
    value as u32
}

/// Compose two basis-point multipliers as `(left * right) / 10_000`, with
/// widened arithmetic and loud `u32` overflow pinning.
#[must_use]
pub fn compose_basis_points_u32(left: u32, right: u32, label: &str, backend: &str) -> u32 {
    let value = (u128::from(left) * u128::from(right)) / u128::from(BASIS_POINTS_DENOMINATOR);
    if value > u128::from(u32::MAX) {
        tracing::error!(
            "{backend} {label} composed basis-points value exceeded u32. Fix: normalize chained multipliers before release-path planning."
        );
        return u32::MAX;
    }
    value as u32
}

/// Compose two basis-point multipliers as `(left * right) / 10_000`, returning
/// `None` rather than saturating when the composed value cannot fit `u64`.
#[must_use]
pub fn checked_compose_basis_points_u64(left: u64, right: u64) -> Option<u64> {
    let value = (u128::from(left) * u128::from(right)) / u128::from(BASIS_POINTS_DENOMINATOR);
    u64::try_from(value).ok()
}

/// Apply a basis-point multiplier to a `u64` with nearest-integer rounding,
/// optional high clamp, and explicit zero-scale policy.
#[must_use]
pub fn scale_u64_by_basis_points_round_clamped(
    base: u64,
    scale_bps: u32,
    zero_scale_value: u64,
    max_scale_bps: u32,
    label: &str,
    backend: &str,
) -> u64 {
    if scale_bps == 0 {
        return zero_scale_value;
    }
    let clamped = if max_scale_bps == 0 {
        scale_bps
    } else {
        scale_bps.min(max_scale_bps)
    };
    let value = (u128::from(base) * u128::from(clamped) + u128::from(BASIS_POINTS_DENOMINATOR / 2))
        / u128::from(BASIS_POINTS_DENOMINATOR);
    if value > u128::from(u64::MAX) {
        tracing::error!(
            "{backend} {label} rounded basis-point scaling exceeded u64. Fix: shard or normalize the cost domain before extraction."
        );
        return u64::MAX;
    }
    value as u64
}

/// Apply a basis-point multiplier to a `u64` with floor rounding and an output
/// lower bound.
#[must_use]
pub fn scale_u64_by_basis_points_floor_min(
    base: u64,
    scale_bps: u32,
    min_value: u64,
    label: &str,
    backend: &str,
) -> u64 {
    let value = (u128::from(base) * u128::from(scale_bps)) / u128::from(BASIS_POINTS_DENOMINATOR);
    if value > u128::from(u64::MAX) {
        tracing::error!(
            "{backend} {label} floor basis-point scaling exceeded u64. Fix: shard or normalize the cost domain before extraction."
        );
        return u64::MAX;
    }
    (value as u64).max(min_value)
}

/// Weight a `u64` cost by basis points into a widened exact `u128` domain.
#[must_use]
pub fn weighted_u64_by_basis_points_u128(value: u64, basis_points: u32) -> u128 {
    (u128::from(value) * u128::from(basis_points)) / u128::from(BASIS_POINTS_DENOMINATOR)
}

/// Convert a finite non-negative floating-point telemetry value to `u32` by
/// truncating toward zero, with loud saturation on invalid or oversized input.
#[must_use]
pub fn finite_f64_to_u32_trunc(value: f64, label: &str, backend: &str) -> u32 {
    if !value.is_finite() {
        tracing::error!(
            "{backend} {label} value {value} is not finite. Fix: normalize telemetry before release-path planning."
        );
        return u32::MAX;
    }
    if value <= 0.0 {
        return 0;
    }
    if value > f64::from(u32::MAX) {
        tracing::error!(
            "{backend} {label} value {value} cannot fit u32. Fix: shard or normalize telemetry before release-path planning."
        );
        return u32::MAX;
    }
    value as u32
}

/// Convert a finite non-negative floating-point telemetry value to `u32` after
/// rounding to the nearest integer, with loud saturation on invalid input.
#[must_use]
pub fn finite_f64_to_u32_round(value: f64, label: &str, backend: &str) -> u32 {
    let rounded = value.round();
    if !rounded.is_finite() {
        tracing::error!(
            "{backend} {label} rounded value {rounded} is not finite. Fix: normalize telemetry before release-path planning."
        );
        return u32::MAX;
    }
    if rounded <= 0.0 {
        return 0;
    }
    if rounded > f64::from(u32::MAX) {
        tracing::error!(
            "{backend} {label} rounded value {rounded} cannot fit u32. Fix: shard or normalize telemetry before release-path planning."
        );
        return u32::MAX;
    }
    rounded as u32
}

/// Convert a finite floating-point ratio into floor basis points, with separate
/// policies for invalid numerators and denominators.
#[must_use]
pub fn finite_f64_ratio_basis_points_trunc(
    numerator: f64,
    denominator: f64,
    invalid_numerator_value: u32,
    invalid_denominator_value: u32,
    label: &str,
    backend: &str,
) -> u32 {
    finite_f64_ratio_basis_points(
        numerator,
        denominator,
        invalid_numerator_value,
        invalid_denominator_value,
        label,
        backend,
        finite_f64_to_u32_trunc,
    )
}

/// Convert a finite floating-point ratio into rounded basis points, with
/// separate policies for invalid numerators and denominators.
#[must_use]
pub fn finite_f64_ratio_basis_points_round(
    numerator: f64,
    denominator: f64,
    invalid_numerator_value: u32,
    invalid_denominator_value: u32,
    label: &str,
    backend: &str,
) -> u32 {
    finite_f64_ratio_basis_points(
        numerator,
        denominator,
        invalid_numerator_value,
        invalid_denominator_value,
        label,
        backend,
        finite_f64_to_u32_round,
    )
}

/// Convert a finite scalar where `1.0 == 10_000 bps` into floor basis points.
#[must_use]
pub fn finite_f64_unit_basis_points_trunc(
    value: f64,
    invalid_value: u32,
    label: &str,
    backend: &str,
) -> u32 {
    if !value.is_finite() {
        tracing::error!(
            "{backend} {label} value {value} is not finite. Fix: normalize telemetry before release-path planning."
        );
        return invalid_value;
    }
    finite_f64_to_u32_trunc(
        value.max(0.0) * f64::from(BASIS_POINTS_DENOMINATOR),
        label,
        backend,
    )
}

fn finite_f64_ratio_basis_points(
    numerator: f64,
    denominator: f64,
    invalid_numerator_value: u32,
    invalid_denominator_value: u32,
    label: &str,
    backend: &str,
    convert: fn(f64, &str, &str) -> u32,
) -> u32 {
    if !numerator.is_finite() {
        tracing::error!(
            "{backend} {label} numerator {numerator} is not finite. Fix: record finite dispatch timing before release-path planning."
        );
        return invalid_numerator_value;
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        tracing::error!(
            "{backend} {label} denominator {denominator} is not finite and positive. Fix: record finite dispatch timing before release-path planning."
        );
        return invalid_denominator_value;
    }
    if numerator <= 0.0 {
        return 0;
    }
    convert(
        (numerator / denominator) * f64::from(BASIS_POINTS_DENOMINATOR),
        label,
        backend,
    )
}

/// Compute `ceil(value / divisor)` in `u64`, returning `None` for zero divisors
/// or arithmetic overflow.
#[must_use]
pub fn checked_ceil_div_u64(value: u64, divisor: u64) -> Option<u64> {
    if divisor == 0 {
        return None;
    }
    if value == 0 {
        return Some(0);
    }
    ((value - 1) / divisor).checked_add(1)
}

/// Multiply three `u32` dimensions into a `u64` without wraparound.
///
/// Backend and runtime launch geometry all cross this same host/device
/// boundary. Keeping the primitive here prevents each backend from carrying a
/// slightly different overflow policy for `[x, y, z]` launch dimensions.
#[must_use]
pub fn checked_dim_product_u64(dims: [u32; 3]) -> Option<u64> {
    u64::from(dims[0])
        .checked_mul(u64::from(dims[1]))
        .and_then(|xy| xy.checked_mul(u64::from(dims[2])))
}

/// Multiply three `u32` dimensions into a `u32` without wraparound.
#[must_use]
pub fn checked_dim_product_u32(dims: [u32; 3]) -> Option<u32> {
    u32::try_from(checked_dim_product_u64(dims)?).ok()
}

macro_rules! define_align_up {
    ($name:ident, $ty:ty) => {
        #[doc = concat!(
            "Align a `",
            stringify!($ty),
            "` value upward after applying a minimum."
        )]
        ///
        /// # Errors
        ///
        /// Returns [`BackendError::InvalidProgram`] when `alignment` is zero or
        /// the padded value would overflow.
        pub fn $name(
            value: $ty,
            alignment: $ty,
            min_value: $ty,
            label: &str,
            backend: &str,
        ) -> Result<$ty, BackendError> {
            if alignment == 0 {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: {backend} {label} alignment must be non-zero before padding."
                    ),
                });
            }
            let normalized = value.max(min_value);
            let remainder = normalized % alignment;
            if remainder == 0 {
                return Ok(normalized);
            }
            normalized.checked_add(alignment - remainder).ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: {backend} {label} overflows {} while padding to {alignment}-byte alignment; split the workload before crossing the host/device boundary.",
                        stringify!($ty)
                    ),
                }
            })
        }
    };
}

define_align_up!(align_up_u64, u64);
define_align_up!(align_up_usize, usize);
