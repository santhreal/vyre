//! What the 32-bit snapshot ABI refuses, and why.
//!
//! Four failures are distinct and a caller acts differently on each: an e-graph
//! too large for a 32-bit column, a snapshot whose columns disagree with each
//! other, an image that cannot be packed, and a bridge run that failed at one
//! of those seams. Each carries the figure it read and the fix that applies.

use std::fmt;

use crate::optimizer::eqsat::EGraphError;

/// Error returned when a CPU e-graph cannot be represented by the current
/// 32-bit GPU snapshot ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuEGraphSnapshotError {
    context: &'static str,
    value: usize,
}

impl GpuEGraphSnapshotError {
    fn new(context: &'static str, value: usize) -> Self {
        Self { context, value }
    }

    /// Human-readable conversion context.
    #[must_use]
    pub const fn context(&self) -> &'static str {
        self.context
    }

    /// Host-side value that could not fit the GPU snapshot ABI.
    #[must_use]
    pub const fn value(&self) -> usize {
        self.value
    }
}

impl fmt::Display for GpuEGraphSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU e-graph snapshot {} value {} exceeds the u32 column ABI. Fix: shard the e-graph snapshot or widen the GPU snapshot ABI before upload.",
            self.context, self.value
        )
    }
}

impl std::error::Error for GpuEGraphSnapshotError {}

/// Error returned when a GPU e-graph snapshot is structurally malformed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuEGraphSnapshotIntegrityError {
    context: &'static str,
    row: usize,
    value: u32,
}

impl GpuEGraphSnapshotIntegrityError {
    fn new(context: &'static str, row: usize, value: u32) -> Self {
        Self {
            context,
            row,
            value,
        }
    }

    /// Human-readable validation context.
    #[must_use]
    pub const fn context(&self) -> &'static str {
        self.context
    }

    /// Snapshot row that failed validation.
    #[must_use]
    pub const fn row(&self) -> usize {
        self.row
    }

    /// Row-local value that failed validation.
    #[must_use]
    pub const fn value(&self) -> u32 {
        self.value
    }
}

impl fmt::Display for GpuEGraphSnapshotIntegrityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU e-graph snapshot integrity error at row {}: {} value {} is invalid. Fix: rebuild the snapshot from canonical e-graph rows before upload.",
            self.row, self.context, self.value
        )
    }
}

impl std::error::Error for GpuEGraphSnapshotIntegrityError {}

/// Error returned when a GPU e-graph snapshot cannot be packed for upload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuEGraphDeviceImageError {
    /// Snapshot rows are structurally malformed.
    Integrity(GpuEGraphSnapshotIntegrityError),
    /// Snapshot or derived index columns exceed the current u32 device ABI.
    Layout(GpuEGraphSnapshotError),
}

impl fmt::Display for GpuEGraphDeviceImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integrity(error) => error.fmt(f),
            Self::Layout(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for GpuEGraphDeviceImageError {}

impl From<GpuEGraphSnapshotIntegrityError> for GpuEGraphDeviceImageError {
    fn from(error: GpuEGraphSnapshotIntegrityError) -> Self {
        Self::Integrity(error)
    }
}

impl From<GpuEGraphSnapshotError> for GpuEGraphDeviceImageError {
    fn from(error: GpuEGraphSnapshotError) -> Self {
        Self::Layout(error)
    }
}

/// Error returned by the measured CPU/GPU e-graph bridge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuEGraphBridgeError {
    /// CPU e-graph snapshot construction failed.
    Snapshot(GpuEGraphSnapshotError),
    /// Snapshot packing into the uploadable device image failed.
    DeviceImage(GpuEGraphDeviceImageError),
    /// CPU e-graph extraction failed during parity proof.
    EGraph(EGraphError),
}

impl fmt::Display for GpuEGraphBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Snapshot(error) => write!(f, "GPU e-graph bridge snapshot failed: {error}"),
            Self::DeviceImage(error) => {
                write!(f, "GPU e-graph bridge device image failed: {error}")
            }
            Self::EGraph(error) => write!(f, "GPU e-graph bridge extraction failed: {error}"),
        }
    }
}

impl std::error::Error for GpuEGraphBridgeError {}

impl From<GpuEGraphSnapshotError> for GpuEGraphBridgeError {
    fn from(error: GpuEGraphSnapshotError) -> Self {
        Self::Snapshot(error)
    }
}

impl From<GpuEGraphDeviceImageError> for GpuEGraphBridgeError {
    fn from(error: GpuEGraphDeviceImageError) -> Self {
        Self::DeviceImage(error)
    }
}

impl From<EGraphError> for GpuEGraphBridgeError {
    fn from(error: EGraphError) -> Self {
        Self::EGraph(error)
    }
}

/// Narrow a column length to the 32-bit width the device image uses.
///
/// Every column is a `u32`, so a length that does not fit is the ABI limit
/// rather than an allocation failure, and the error names which column hit it.
#[inline]
pub(super) fn u32_len(value: usize, context: &'static str) -> Result<u32, GpuEGraphSnapshotError> {
    u32::try_from(value).map_err(|_| GpuEGraphSnapshotError::new(context, value))
}

#[cfg(test)]
mod tests {
    use super::u32_len;

    /// WHY: `u32_len` is the one place the 32-bit column limit is enforced and
    /// it is private to this module, so no integration test can reach it. A
    /// saturating conversion here would hand the device a truncated column
    /// length and corrupt every row past it.
    #[test]
    fn a_column_length_over_the_abi_width_is_refused_with_both_fixes() {
        let error = u32_len(u32::MAX as usize + 1, "test overflow")
            .expect_err("Fix: GPU e-graph snapshot must not silently saturate oversized columns");

        assert_eq!(error.context(), "test overflow");
        assert_eq!(error.value(), u32::MAX as usize + 1);
        assert!(
            error.to_string().contains("shard the e-graph snapshot")
                && error.to_string().contains("widen the GPU snapshot ABI"),
            "oversized GPU snapshot errors must explain both viable fixes"
        );
    }
}
