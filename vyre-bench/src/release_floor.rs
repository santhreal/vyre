//! The device a release measurement is allowed to have been taken on.
//!
//! The memory floor is derived from the registered benchmark catalog, not
//! declared as a device class. Every case states the resident bytes it
//! allocates, so the floor is the largest of those plus what a CUDA context
//! holds before the first workload byte, and a case that grows raises the floor
//! without anyone editing this file. The catalog is this crate's, so the
//! derivation is this crate's: a reader in the release tooling and a reader in
//! the recorded-artifact contracts both ask here, and neither restates a number.
//!
//! It used to be a flat 16384 MiB, restated at five call sites. No case came
//! close to needing it: the largest declared working set in the catalog is
//! 128 MiB. The number rejected every consumer card in the fleet on a claim
//! about what class of device the published figures came from, and it made
//! release evidence impossible to record on hardware that runs every workload
//! correctly and at full occupancy.
//!
//! The compute-capability floor stays, because that one is a feature floor: the
//! CUDA backend is built for `cuda-12000` and emits asynchronous copies and
//! cooperative launches that a pre-Ampere device cannot execute at all.

/// Device memory a CUDA context holds before a single workload byte is
/// allocated: the loaded module images, the runtime heap and the driver's own
/// reserve. Held well above the observed few hundred MiB so a device that only
/// just clears the largest workload is still rejected.
const CUDA_CONTEXT_RESERVE_MIB: u64 = 1024;

/// Smallest total device memory a CUDA release measurement may report, in MiB.
///
/// Derived at run time from the registered cases so a new workload that needs
/// more memory than any existing one moves the floor by itself. A catalog that
/// declares no resident bytes still yields the context reserve, which is the
/// smallest device that can hold a CUDA context at all.
#[must_use]
pub fn min_cuda_release_memory_mib() -> u64 {
    let largest_workload_bytes = crate::registry::collect_all()
        .iter()
        .filter_map(|case| case.requirements().min_vram_bytes)
        .max()
        .unwrap_or(0);
    largest_workload_bytes.div_ceil(1024 * 1024) + CUDA_CONTEXT_RESERVE_MIB
}

/// Smallest CUDA compute capability a release measurement may report.
pub const RELEASE_COMPUTE_CAPABILITY_FLOOR: (u64, u64) = (8, 0);

/// Whether a probed device may carry a release measurement.
///
/// The one owner of the comparison. The live `nvidia-smi` probe, the
/// recorded-JSON release checks and the artifact contracts all route here so a
/// device cannot qualify for one and be rejected by another.
#[must_use]
pub fn device_meets_release_floor(
    memory_total_mib: Option<u64>,
    compute_capability: Option<(u64, u64)>,
) -> bool {
    memory_total_mib.is_some_and(|mib| mib >= min_cuda_release_memory_mib())
        && compute_capability.is_some_and(|found| found >= RELEASE_COMPUTE_CAPABILITY_FLOOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the floor exists to reject a device that cannot hold the workload
    /// it claims to have measured, so it has to cover the largest registered
    /// case. A constant would answer this too, and would stop answering it the
    /// day somebody registers a bigger workload; deriving it is the whole
    /// point, so the test reads the catalog the same way the floor does.
    #[test]
    fn the_floor_covers_every_registered_workload() {
        let floor_bytes = min_cuda_release_memory_mib() * 1024 * 1024;
        for case in crate::registry::collect_all().iter() {
            let Some(needed) = case.requirements().min_vram_bytes else {
                continue;
            };
            assert!(
                needed < floor_bytes,
                "Fix: case `{}` declares {needed} resident bytes, which the release memory floor of {floor_bytes} bytes does not cover.",
                case.id().0,
            );
        }
    }

    /// WHY: a device with room for the workload and nothing else still cannot
    /// run it, because the context is allocated first. The margin is the part a
    /// derived floor can silently lose if the reserve is ever folded away.
    #[test]
    fn the_floor_leaves_room_for_the_cuda_context() {
        let largest = crate::registry::collect_all()
            .iter()
            .filter_map(|case| case.requirements().min_vram_bytes)
            .max()
            .unwrap_or(0);
        assert!(
            min_cuda_release_memory_mib() - largest.div_ceil(1024 * 1024)
                >= CUDA_CONTEXT_RESERVE_MIB,
            "Fix: the release memory floor must exceed the largest workload by the CUDA context reserve."
        );
    }

    /// WHY: both floors are inclusive, and both are compared by callers that
    /// used to hold their own copy of the comparison. An off-by-one at either
    /// boundary either rejects a device that runs every workload or admits one
    /// that runs none, and neither shows up as a test failure anywhere else:
    /// the live probe path needs real hardware to reach.
    #[test]
    fn the_predicate_admits_the_boundary_and_rejects_everything_under_it() {
        let floor = min_cuda_release_memory_mib();
        let (major, minor) = RELEASE_COMPUTE_CAPABILITY_FLOOR;

        assert!(device_meets_release_floor(
            Some(floor),
            Some((major, minor))
        ));
        assert!(!device_meets_release_floor(
            Some(floor - 1),
            Some((major, minor))
        ));
        assert!(!device_meets_release_floor(
            Some(floor),
            Some((major - 1, 9))
        ));
        assert!(device_meets_release_floor(
            Some(floor),
            Some((major, minor + 9))
        ));
        assert!(!device_meets_release_floor(None, Some((major, minor))));
        assert!(!device_meets_release_floor(Some(floor), None));
    }
}
