//! Contract tests for Platform Support Matrix & Wire Conversions (Row 118).

use vyre_foundation::platform::*;

#[test]
fn canonical_platform_support_matrix_covers_tier_1_hosts() {
    let matrix = PlatformSupportMatrix::canonical();
    assert_eq!(
        matrix.schema_version,
        PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION
    );
    assert_eq!(matrix.canonical_endianness, Endianness::LittleEndian);

    // Current host must validate successfully
    let current = matrix
        .validate_active_environment()
        .expect("current host should be supported");
    assert_eq!(current.endianness, Endianness::LittleEndian);
    assert_eq!(current.pointer_width, PointerWidth::Bits64);
}

#[test]
fn unsupported_host_cell_fails_closed_with_typed_diagnostic() {
    let matrix = PlatformSupportMatrix::canonical();
    let unsupported = HostCell {
        os: HostOs::Other("obscure_os".to_string()),
        arch: HostArch::Armv7,
        pointer_width: PointerWidth::Bits32,
        endianness: Endianness::BigEndian,
        rust_version: "1.70.0".to_string(),
    };

    let result = matrix.is_supported(&unsupported);
    let err = result.expect_err("unsupported host cell must return Err");
    assert!(matches!(
        err,
        UnsupportedPlatformError::UnsupportedHost {
            ref os,
            ref arch,
            pointer_width: PointerWidth::Bits32,
            endianness: Endianness::BigEndian,
        } if matches!(os, HostOs::Other(name) if name == "obscure_os") && matches!(arch, HostArch::Armv7)
    ));

    let msg = err.to_string();
    assert!(
        msg.contains("other(obscure_os)"),
        "diagnostic must name the offending OS, got: {msg}"
    );
    assert!(
        msg.contains("armv7"),
        "diagnostic must name the offending architecture, got: {msg}"
    );
    assert!(
        msg.contains("32-bit"),
        "diagnostic must name pointer width, got: {msg}"
    );
    assert!(
        msg.contains("big-endian"),
        "diagnostic must name endianness, got: {msg}"
    );
    assert!(
        msg.contains("Fix:"),
        "diagnostic must provide an actionable Fix: hint, got: {msg}"
    );
    assert!(
        msg.contains("docs/generated/platform-support-matrix.toml"),
        "diagnostic must reference the generated support matrix, got: {msg}"
    );
}

#[test]
fn unsupported_pointer_width_and_endianness_cells_fail_with_actionable_diagnostics() {
    let matrix = PlatformSupportMatrix::canonical();

    // 32-bit Linux cell is unsupported for tier 1 runtime execution
    let cell_32bit = HostCell {
        os: HostOs::Linux,
        arch: HostArch::X86_64,
        pointer_width: PointerWidth::Bits32,
        endianness: Endianness::LittleEndian,
        rust_version: "1.85".to_string(),
    };
    let err_32bit = matrix.is_supported(&cell_32bit).unwrap_err();
    let msg_32bit = err_32bit.to_string();
    assert!(msg_32bit.contains("32-bit"));
    assert!(msg_32bit.contains("Fix:"));

    // Big-endian Linux cell is unsupported for runtime execution
    let cell_be = HostCell {
        os: HostOs::Linux,
        arch: HostArch::X86_64,
        pointer_width: PointerWidth::Bits64,
        endianness: Endianness::BigEndian,
        rust_version: "1.85".to_string(),
    };
    let err_be = matrix.is_supported(&cell_be).unwrap_err();
    let msg_be = err_be.to_string();
    assert!(msg_be.contains("big-endian"));
    assert!(msg_be.contains("Fix:"));
}

#[test]
fn checked_conversions_prevent_overflow_and_truncation() {
    // Valid usize to u32
    assert_eq!(checked_usize_to_u32(1024).unwrap(), 1024);
    // Overflow usize to u32
    assert!(checked_usize_to_u32(usize::MAX).is_err());

    // Valid isize to i32
    assert_eq!(checked_isize_to_i32(-500).unwrap(), -500);
    assert_eq!(checked_isize_to_i32(500).unwrap(), 500);

    // Fixed wire encodings
    let u32_val = CanonicalU32::new(0x12345678);
    assert_eq!(u32_val.to_le_bytes(), [0x78, 0x56, 0x34, 0x12]);
    assert_eq!(
        CanonicalU32::from_le_bytes([0x78, 0x56, 0x34, 0x12]),
        u32_val
    );

    let u64_val = CanonicalU64::new(0x01020304_05060708);
    assert_eq!(
        u64_val.to_le_bytes(),
        [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
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
