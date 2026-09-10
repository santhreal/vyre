//! Contract tests for Platform Support Matrix & Wire Conversions.

use vyre_foundation::platform::*;

/// Every declared cell states pointer width and byte order that match the
/// architecture naming it, and the running host appears exactly once.
///
/// The cell space is read from `HostOs::ALL` and `HostArch::ALL`, so a new
/// operating system or architecture is covered the moment it is declared.
/// Nothing here pins a width or an order for the running host: this suite is
/// executed on 32-bit and big-endian cells and a pinned host fact would make
/// those runs fail for being what they were selected to be.
#[test]
fn every_declared_cell_carries_the_architecture_facts_of_its_arch() {
    let matrix = PlatformSupportMatrix::canonical();
    assert_eq!(
        matrix.schema_version,
        PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION
    );
    assert_eq!(matrix.canonical_endianness, Endianness::LittleEndian);
    assert_eq!(matrix.rust_version, CANONICAL_RUST_VERSION);
    assert_eq!(matrix.cells.len(), HostOs::ALL.len() * HostArch::ALL.len());

    for (cell, tier) in &matrix.cells {
        assert_eq!(
            cell.pointer_width,
            cell.arch.pointer_width(),
            "cell {cell} states a pointer width its architecture does not have"
        );
        assert_eq!(
            cell.endianness,
            cell.arch.endianness(),
            "cell {cell} states a byte order its architecture does not have"
        );
        assert_eq!(*tier, PlatformSupportMatrix::tier(cell.os, cell.arch));
    }

    let current = HostCell::current();
    let occurrences = matrix
        .cells
        .iter()
        .filter(|(cell, _)| *cell == current)
        .count();
    assert_eq!(
        occurrences, 1,
        "the running host {current} must appear in the matrix exactly once"
    );
}

/// The running host carries a claim, and a runtime claim is asserted only
/// where one is made.
///
/// An excluded host that still builds and runs this suite is the silent
/// degradation the matrix exists to prevent, so it is a failure here.
#[test]
fn the_running_host_is_never_an_excluded_cell() {
    let matrix = PlatformSupportMatrix::canonical();
    let current = HostCell::current();
    let tier = PlatformSupportMatrix::tier(current.os, current.arch);

    match tier {
        HostSupportTier::Runtime => {
            let validated = matrix
                .validate_active_environment()
                .expect("a runtime-tier host must validate");
            assert_eq!(validated, current);
            assert!(matrix.runtime_cells().contains(&current));
        }
        HostSupportTier::Encoding => {
            let err = matrix
                .validate_active_environment()
                .expect_err("an encoding-tier host must not pass a runtime check");
            assert!(matches!(
                err,
                UnsupportedPlatformError::EncodingOnlyHost {
                    cell,
                    tier: HostSupportTier::Encoding,
                } if cell == current
            ));
            assert!(matrix.encoding_cells().contains(&current));
        }
        HostSupportTier::Excluded => panic!(
            "this suite is running on {current}, which the matrix excludes from every claim"
        ),
    }
}

/// Both rejections name the cell, the remedy, and the artifact that records
/// claims, and the encoding rejection also names the tier the cell does hold.
#[test]
fn rejected_cells_fail_closed_with_actionable_typed_diagnostics() {
    let matrix = PlatformSupportMatrix::canonical();

    let excluded = matrix
        .cells
        .iter()
        .find(|(_, tier)| *tier == HostSupportTier::Excluded)
        .map(|(cell, _)| *cell)
        .expect("the matrix declares at least one excluded cell");
    let err = matrix
        .require_runtime(&excluded)
        .expect_err("an excluded cell must be rejected");
    assert!(matches!(
        err,
        UnsupportedPlatformError::ExcludedHost { cell } if cell == excluded
    ));
    let msg = err.to_string();
    for fragment in [
        excluded.os.id(),
        excluded.arch.id(),
        "Fix:",
        "docs/generated/platform-support-matrix.toml",
    ] {
        assert!(
            msg.contains(fragment),
            "exclusion diagnostic must name {fragment}, got: {msg}"
        );
    }

    let encoding = matrix
        .encoding_cells()
        .first()
        .copied()
        .expect("the matrix declares at least one encoding-tier cell");
    let err = matrix
        .require_runtime(&encoding)
        .expect_err("an encoding-tier cell must be rejected for runtime use");
    assert!(matches!(
        err,
        UnsupportedPlatformError::EncodingOnlyHost {
            cell,
            tier: HostSupportTier::Encoding,
        } if cell == encoding
    ));
    let msg = err.to_string();
    for fragment in [
        encoding.os.id(),
        encoding.arch.id(),
        HostSupportTier::Encoding.id(),
        "Fix:",
    ] {
        assert!(
            msg.contains(fragment),
            "encoding-tier diagnostic must name {fragment}, got: {msg}"
        );
    }
}

/// Every checked conversion is exercised at its boundary on whatever host is
/// running, and the canonical wire types keep their little-endian layout
/// there.
///
/// The boundaries are computed from `usize::BITS`, not written down. A test
/// that spelled `usize::MAX` as an overflowing input would pass on a 64-bit
/// host and silently prove nothing on a 32-bit one, where `usize::MAX` fits
/// a `u32` exactly.
#[test]
fn checked_conversions_reject_every_value_that_does_not_fit() {
    assert_eq!(checked_usize_to_u32(1024).unwrap(), 1024);
    assert_eq!(checked_usize_to_u64(1024).unwrap(), 1024);
    assert_eq!(checked_u64_to_usize(1024).unwrap(), 1024);
    assert_eq!(checked_isize_to_i32(-500).unwrap(), -500);
    assert_eq!(checked_isize_to_i64(-500).unwrap(), -500);
    assert_eq!(checked_i64_to_isize(-500).unwrap(), -500);

    assert_eq!(checked_usize_to_u32(usize::MAX).is_err(), usize::BITS > 32);
    assert_eq!(checked_isize_to_i32(isize::MAX).is_err(), usize::BITS > 32);
    assert_eq!(checked_isize_to_i32(isize::MIN).is_err(), usize::BITS > 32);

    // Widening on every host this crate builds for, so these never reject
    // here; the check exists for a host with a wider pointer.
    assert_eq!(checked_usize_to_u64(usize::MAX).unwrap(), usize::MAX as u64);
    assert_eq!(checked_isize_to_i64(isize::MIN).unwrap(), isize::MIN as i64);

    // Narrowing from the wire to a host index. On a 32-bit host this is the
    // rejection that stops a payload indexing an element the host cannot
    // address.
    assert_eq!(checked_u64_to_usize(u64::MAX).is_err(), usize::BITS < 64);
    assert_eq!(checked_i64_to_isize(i64::MIN).is_err(), usize::BITS < 64);

    if usize::BITS < 64 {
        let err = checked_u64_to_usize(u64::MAX).unwrap_err();
        assert!(matches!(
            err,
            ConversionError::OverflowHostPointer {
                val: u64::MAX,
                host_pointer_bits,
            } if host_pointer_bits == usize::BITS
        ));
        assert!(err.to_string().contains("Fix:"));
    } else {
        let err = checked_usize_to_u32(usize::MAX).unwrap_err();
        assert!(matches!(
            err,
            ConversionError::Overflow32 { val } if val == usize::MAX as i128
        ));
        assert!(err.to_string().contains("Fix:"));
    }

    // Canonical layout is fixed little-endian regardless of host byte order.
    let u32_val = CanonicalU32::new(0x1234_5678);
    assert_eq!(u32_val.to_le_bytes(), [0x78, 0x56, 0x34, 0x12]);
    assert_eq!(
        CanonicalU32::from_le_bytes([0x78, 0x56, 0x34, 0x12]),
        u32_val
    );

    let u64_val = CanonicalU64::new(0x0102_0304_0506_0708);
    assert_eq!(
        u64_val.to_le_bytes(),
        [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
    );
    assert_eq!(
        CanonicalU64::from_le_bytes([0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]),
        u64_val
    );
}

#[test]
fn platform_adapters_provide_typed_behavior_and_safe_cleanup() {
    // Clock adapter
    let t0 = ClockAdapter::monotonic_now_ns();
    let elapsed = ClockAdapter::elapsed_ns(t0);
    assert!(elapsed < 1_000_000_000); // Less than 1 second

    // Scratch directory lifecycle
    let scratch_path;
    {
        let scratch =
            FileSystemAdapter::create_scratch_dir("vyre_test_scratch").expect("create scratch dir");
        scratch_path = scratch.path().to_path_buf();
        assert!(scratch_path.exists());

        let target_file = scratch_path.join("atomic_test.bin");
        FileSystemAdapter::atomic_write(&target_file, b"canonical_wire_bytes")
            .expect("atomic write");

        let read_back = FileSystemAdapter::read_bounded(&target_file, 1024).expect("read bounded");
        assert_eq!(read_back, b"canonical_wire_bytes");

        // Quota exceed test
        let quota_err = FileSystemAdapter::read_bounded(&target_file, 5).unwrap_err();
        assert!(matches!(
            quota_err,
            PlatformAdapterError::QuotaExceeded { .. }
        ));
    }
    // Scratch directory must be cleaned up on drop
    assert!(!scratch_path.exists());

    // Thread adapter
    let handle = ThreadAdapter::spawn_named("vyre_contract_worker", 2 * 1024 * 1024, || {
        ThreadAdapter::yield_now();
        42u32
    })
    .expect("thread spawn");

    assert_eq!(handle.join().unwrap(), 42);
}

/// A payload one version behind the current schema is rejected, and the
/// rejection names both versions.
///
/// The expectation reads the constant, so a schema bump that forgets the
/// reader turns this red instead of pinning a version nothing writes.
#[test]
fn stale_platform_matrix_schema_fails_closed() {
    let stale = PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION - 1;
    let mut matrix = PlatformSupportMatrix::canonical();
    matrix.schema_version = stale;

    let toml = toml::to_string(&matrix).unwrap();
    let err = PlatformSupportMatrix::from_toml(&toml).unwrap_err();
    assert!(matches!(
        err,
        UnsupportedPlatformError::StaleSchemaVersion { expected, found }
            if expected == PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION && found == stale
    ));
}
