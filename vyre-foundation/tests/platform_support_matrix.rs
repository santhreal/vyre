//! Contract tests for Platform Support Matrix & Wire Conversions (Row 118).

use vyre_foundation::platform::*;

#[test]
fn canonical_platform_support_matrix_covers_tier_1_hosts() {
    let matrix = PlatformSupportMatrix::canonical();
    assert_eq!(matrix.schema_version, PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION);
    assert_eq!(matrix.canonical_endianness, Endianness::LittleEndian);

    // Current host must validate successfully
    let current = matrix.validate_active_environment().expect("current host should be supported");
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
    assert!(matches!(result, Err(UnsupportedPlatformError::UnsupportedHost { .. })));
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
    assert_eq!(CanonicalU32::from_le_bytes([0x78, 0x56, 0x34, 0x12]), u32_val);

    let u64_val = CanonicalU64::new(0x01020304_05060708);
    assert_eq!(u64_val.to_le_bytes(), [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
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
        let scratch = FileSystemAdapter::create_scratch_dir("vyre_test_scratch").expect("create scratch dir");
        scratch_path = scratch.path().to_path_buf();
        assert!(scratch_path.exists());

        let target_file = scratch_path.join("atomic_test.bin");
        FileSystemAdapter::atomic_write(&target_file, b"canonical_wire_bytes").expect("atomic write");

        let read_back = FileSystemAdapter::read_bounded(&target_file, 1024).expect("read bounded");
        assert_eq!(read_back, b"canonical_wire_bytes");

        // Quota exceed test
        let quota_err = FileSystemAdapter::read_bounded(&target_file, 5).unwrap_err();
        assert!(matches!(quota_err, PlatformAdapterError::QuotaExceeded { .. }));
    }
    // Scratch directory must be cleaned up on drop
    assert!(!scratch_path.exists());

    // Thread adapter
    let handle = ThreadAdapter::spawn_named("vyre_contract_worker", 2 * 1024 * 1024, || {
        ThreadAdapter::yield_now();
        42u32
    }).expect("thread spawn");

    assert_eq!(handle.join().unwrap(), 42);
}

#[test]
fn stale_platform_matrix_schema_fails_closed() {
    let mut matrix = PlatformSupportMatrix::canonical();
    matrix.schema_version = 0; // Stale version

    let toml = toml::to_string(&matrix).unwrap();
    let err = PlatformSupportMatrix::from_toml(&toml).unwrap_err();
    assert!(matches!(err, UnsupportedPlatformError::StaleSchemaVersion { expected: 1, found: 0 }));
}
