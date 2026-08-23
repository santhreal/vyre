use super::*;
use vyre_driver::ResidentOwner;

/// The mapped-at-creation upload fast path (the StagingBelt replacement)
/// must produce a buffer whose contents byte-for-byte equal the input across
/// every boundary class: sub-word, exactly-a-word, word+tail, and a large
/// payload (the catalog-scale path). A regression here is a silent data
/// corruption on the ~1 GB DFA-catalog upload.
#[cfg(feature = "device-tests")]
#[test]
fn mapped_upload_roundtrips_exact_bytes_across_boundaries() {
    let arc = crate::runtime::cached_device()
        .expect("Fix: live GPU device required for mapped upload roundtrip test");
    let (device, queue) = &*arc;
    // 1,3 exercise the 4-byte tail; 4 is exactly aligned; 5 is word+tail;
    // 257 is multi-word+tail; 1 MiB + 3 is the large, tail-padded path that
    // used to route through the slow per-write StagingBelt.
    for &len in &[1usize, 3, 4, 5, 257, (1 << 20) + 3] {
        let contents: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let handle =
            GpuBufferHandle::upload(device, queue, &contents, wgpu::BufferUsages::COPY_SRC)
                .expect("Fix: mapped upload should succeed at every size");
        let mut out = Vec::new();
        handle
            .readback(device, queue, &mut out)
            .expect("Fix: readback should succeed");
        assert_eq!(
            out, contents,
            "mapped upload corrupted {len}-byte payload: readback != input"
        );
    }
}

/// `write_padded_into_mapped` (the fast-path filler) must copy the logical
/// bytes and zero the alignment tail deterministically, proved without a
/// GPU so the contract holds on every host.
#[test]
fn write_padded_into_mapped_zeroes_the_tail() {
    // Allocation of 8 bytes, 5 logical: bytes 0..5 copied, 5..8 zeroed even
    // if the destination started with garbage.
    let mut mapped = [0xAAu8; 8];
    let bytes = [1u8, 2, 3, 4, 5];
    crate::padded_upload::write_padded_into_mapped(&mut mapped, &bytes)
        .expect("Fix: filling a large-enough mapped slice must succeed");
    assert_eq!(&mapped[..5], &bytes);
    assert_eq!(&mapped[5..], &[0u8, 0, 0], "alignment tail must be zeroed");
    // A slice smaller than the data must fail closed, never truncate.
    let mut too_small = [0u8; 2];
    assert!(crate::padded_upload::write_padded_into_mapped(&mut too_small, &bytes).is_err());
}

#[test]
fn foreign_resident_handle_is_refused_not_resolved() {
    // A handle minted outside the WGPU namespace carries an id that is
    // perfectly valid here, so resolving it by bare id would hand back an
    // unrelated live buffer. The boundary must refuse instead.
    let foreign = ResidentOwner::new().expect("Fix: owner ids must be available");
    let native = resident::resident_owner().expect("Fix: WGPU resident owner must be available");
    assert_ne!(foreign, native);

    let error = GpuBufferHandle::from_resident_handle(foreign.handle(1), "refusal test")
        .expect_err("Fix: a foreign resident handle must never resolve to a WGPU buffer");
    let BackendError::InvalidProgram { fix } = error else {
        panic!("Fix: foreign resident handle refusal must be BackendError::InvalidProgram");
    };
    assert!(
        fix.contains("owned by backend instance") && fix.contains("Fix: "),
        "Fix: refusal must name the owning instance and carry actionable text, got {fix}"
    );

    assert!(
        check_resident_owner(native.handle(u64::MAX), "refusal test").is_ok(),
        "Fix: a handle from this namespace must pass the owner check even when its buffer is gone"
    );
}

#[cfg(feature = "device-tests")]
#[test]
fn resident_registry_handles_concurrent_lookup_and_drop() {
    let arc = crate::runtime::cached_device()
        .expect("Fix: GPU device is required for resident registry concurrency test");
    let (device, queue) = &*arc;
    let handle =
        GpuBufferHandle::upload(device, queue, &[1, 2, 3, 4], wgpu::BufferUsages::COPY_SRC)
            .expect("Fix: upload should register a resident buffer");
    let id = handle.id();

    // Phase 1: while the handle is alive, 8 concurrent readers
    // must always resolve the resident id. Join BEFORE the drop so
    // there is no readers-vs-drop race producing flaky panics.
    let readers = (0..8)
        .map(|_| {
            std::thread::spawn(move || {
                for _ in 0..1_000 {
                    let resident = GpuBufferHandle::from_resident_id(id)
                        .expect("Fix: resident id must resolve while handle is alive");
                    assert_eq!(resident.id(), id);
                }
            })
        })
        .collect::<Vec<_>>();
    for reader in readers {
        reader
            .join()
            .expect("Fix: concurrent resident lookups must not panic");
    }

    // Phase 2: dropping the handle must remove the id from the
    // registry so subsequent lookups return None.
    drop(handle);
    assert!(
        GpuBufferHandle::from_resident_id(id).is_none(),
        "dropped handles must be removed from the resident registry"
    );
}
