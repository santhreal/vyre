//! The replay log's on-disk format survives every way a run can end.
//!
//! # The class this closes
//!
//! The log is the only record of what the host published, and it is read by a
//! later process than the one that wrote it: a build that reads its own writes
//! and nothing else proves nothing about the format. Every failure here is
//! silent by construction - a truncated header, a capacity the reader would
//! allocate against, a slot a sector fault zeroed - so each is driven from raw
//! bytes on disk rather than through the writer that produced them.
//!
//! # What it does not catch
//!
//! It does not prove two concurrent writers on one path interleave safely; the
//! log takes one writer by contract.

use std::path::Path;

use vyre_driver::BackendError;
use vyre_runtime::replay::{
    RecordedSlot, ReplayFailureClass, ReplayFailureEvidence, ReplayLogError, RingLog, HEADER_BYTES,
    LOG_MAGIC, LOG_VERSION, MAX_REPLAY_RECORDS, RECORD_BYTES,
};

fn rec(slot_idx: u32, epoch: u32) -> RecordedSlot {
    RecordedSlot {
        timestamp_ns: 1_000_000 + slot_idx as u64,
        slot_idx,
        tenant_id: 0,
        opcode: 0x4000_0000 + slot_idx,
        args: [slot_idx, slot_idx * 2, slot_idx * 3, slot_idx * 4],
        epoch,
    }
}

/// Write a header with the real magic and version but caller-chosen counts.
fn write_header(path: &Path, capacity: u64, cursor: u64) {
    use std::io::Write as _;

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(LOG_MAGIC).unwrap();
    f.write_all(&LOG_VERSION.to_le_bytes()).unwrap();
    f.write_all(&0u32.to_le_bytes()).unwrap();
    f.write_all(&capacity.to_le_bytes()).unwrap();
    f.write_all(&cursor.to_le_bytes()).unwrap();
}

#[test]
fn open_rejects_zero_capacity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    let err = RingLog::open(&path, 0).expect_err("zero capacity must reject");
    assert!(matches!(err, ReplayLogError::ZeroCapacity));
}

#[test]
fn append_and_replay_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    let mut log = RingLog::open(&path, 4)
        .expect("Fix: open fresh log; restore this invariant before continuing.");
    log.append(rec(1, 10)).unwrap();
    log.append(rec(2, 11)).unwrap();
    log.sync().unwrap();

    let replay = log
        .replay_all()
        .expect("Fix: replay; restore this invariant before continuing.");
    assert_eq!(replay.len(), 2);
    assert_eq!(replay[0].slot_idx, 1);
    assert_eq!(replay[0].epoch, 10);
    assert_eq!(replay[1].slot_idx, 2);
    assert_eq!(replay[1].epoch, 11);
}

#[test]
fn append_with_failure_round_trips_reproduction_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    let mut log = RingLog::open(&path, 4)
        .expect("Fix: open fresh log; restore this invariant before continuing.");
    let backend_error = BackendError::DeviceLost {
        backend: "fixture".to_string(),
        device: "fixture-0".to_string(),
        generation: 7,
        message: "device loss after queue submit".to_string(),
    };
    let failure = ReplayFailureEvidence::from_backend_error(3, &backend_error, b"partial-output");

    assert_eq!(failure.failure_class, ReplayFailureClass::DeviceLoss);
    assert_eq!(failure.backend_error_code, backend_error.code().stable_id());
    assert_ne!(failure.output_digest, 0);

    log.append_with_failure(rec(7, 44), failure).unwrap();
    log.sync().unwrap();

    let replay = log
        .replay_records()
        .expect("Fix: replay records; restore this invariant before continuing.");
    assert_eq!(replay.len(), 1);
    assert_eq!(replay[0].slot.slot_idx, 7);
    assert_eq!(replay[0].slot.epoch, 44);
    assert_eq!(replay[0].failure, Some(failure));
}

#[test]
fn append_without_failure_has_no_failure_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    let mut log = RingLog::open(&path, 2)
        .expect("Fix: open fresh log; restore this invariant before continuing.");

    log.append(rec(1, 10)).unwrap();

    let replay = log
        .replay_records()
        .expect("Fix: replay records; restore this invariant before continuing.");
    assert_eq!(replay.len(), 1);
    assert_eq!(replay[0].slot.slot_idx, 1);
    assert_eq!(replay[0].failure, None);
}

#[test]
fn log_rollover_preserves_most_recent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    let mut log =
        RingLog::open(&path, 3).expect("Fix: open; restore this invariant before continuing.");
    for i in 0..5 {
        log.append(rec(i, 100 + i)).unwrap();
    }
    let replay = log
        .replay_all()
        .expect("Fix: replay; restore this invariant before continuing.");
    assert_eq!(replay.len(), 3, "capacity=3 must retain exactly 3 records");
    let slot_ids: Vec<u32> = replay.iter().map(|r| r.slot_idx).collect();
    // Publish order: 0, 1, 2, 3, 4. After 2 wraps, live records
    // are [3, 4, 2] in ring-physical order; replay starts at
    // next_slot = 2 so the visible order is [2, 3, 4].
    assert_eq!(slot_ids, vec![2, 3, 4]);
}

#[test]
fn reopen_restores_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    {
        let mut log = RingLog::open(&path, 4)
            .expect("Fix: open fresh; restore this invariant before continuing.");
        log.append(rec(1, 10)).unwrap();
        log.append(rec(2, 11)).unwrap();
        log.sync().unwrap();
    }
    let mut reopened =
        RingLog::open(&path, 4).expect("Fix: reopen; restore this invariant before continuing.");
    assert_eq!(reopened.cursor(), 2);
    let replay = reopened.replay_all().unwrap();
    assert_eq!(replay.len(), 2);
}

#[test]
fn corrupted_magic_rejected() {
    use std::io::Write as _;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    {
        // Create a "log" file with the wrong magic.
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"XXXX0001").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&4u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        // Ensure enough bytes for the subsequent reads in open() (headers >= 32 B).
        f.set_len(HEADER_BYTES + 4 * RECORD_BYTES).unwrap();
    }
    let err = RingLog::open(&path, 4).expect_err("wrong magic must reject");
    assert!(matches!(err, ReplayLogError::HeaderMismatch { .. }));
}

#[test]
fn existing_log_zero_capacity_rejected_before_cursor_modulo() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    write_header(&path, 0, 0);

    let err = RingLog::open(&path, 4).expect_err("header capacity=0 must reject");
    assert!(matches!(err, ReplayLogError::ZeroCapacity));
}

#[test]
fn existing_log_huge_capacity_rejected_before_replay_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    write_header(&path, MAX_REPLAY_RECORDS + 1, 0);

    let err = RingLog::open(&path, 4).expect_err("huge header capacity must reject");
    assert!(matches!(
        err,
        ReplayLogError::CapacityOverflow {
            count,
            max: MAX_REPLAY_RECORDS
        } if count == MAX_REPLAY_RECORDS + 1
    ));
}

#[test]
fn capacity_overflow_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    let err =
        RingLog::open(&path, MAX_REPLAY_RECORDS + 1).expect_err("over-size capacity must reject");
    assert!(matches!(
        err,
        ReplayLogError::CapacityOverflow {
            count,
            max: MAX_REPLAY_RECORDS
        } if count == MAX_REPLAY_RECORDS + 1
    ));
}

/// A zeroed live slot is skipped, and the caller can see that it was.
///
/// The skip used to be silent, so an operator reading a replay shorter than the
/// number of appended records had no signal that a zero-magic record had been
/// hit. Tracing output is not assertable here, but the observable contract is: a
/// zero-magic slot mid-scan must not error, and the returned length must be
/// short by exactly the zeroed slots, which is what makes the gap visible.
#[test]
fn replay_zero_magic_mid_sequence_skips_gracefully_and_produces_shorter_result() {
    use std::io::{Seek, SeekFrom, Write as _};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.vrrl");
    let mut log = RingLog::open(&path, 4)
        .expect("Fix: open fresh log; restore this invariant before continuing.");

    // Append 3 records into a 4-slot capacity log.
    log.append(rec(10, 100)).unwrap();
    log.append(rec(20, 200)).unwrap();
    log.append(rec(30, 300)).unwrap();
    log.sync().unwrap();

    // Verify a clean replay first: cursor = 3, scan starts at slot 3 (empty),
    // then wraps to 0, 1, 2 (so we get exactly 3 records).
    {
        let records = log
            .replay_all()
            .expect("Fix: replay of 3 records must succeed");
        assert_eq!(records.len(), 3, "Fix: 3 appended records must all replay");
    }

    // Now zero out the record at slot 1 (record 20) directly via file I/O.
    // This simulates a sector fault / partial crash zeroing a live slot.
    let slot1_offset = HEADER_BYTES + RECORD_BYTES; // slot 0 is at HEADER_BYTES; slot 1 follows
    {
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(slot1_offset)).unwrap();
        f.write_all(&[0u8; RECORD_BYTES as usize]).unwrap();
        f.sync_all().unwrap();
    }

    // Re-open the log to pick up the zeroed slot.
    let mut log2 = RingLog::open(&path, 4).expect("Fix: reopen after zeroing must succeed");

    // Replay must not return Err (the zero-magic skip is graceful).
    let records = log2
        .replay_all()
        .expect("Fix: replay with a zeroed slot must not error");

    // We should now see only 2 records (slot 0 = rec(10) and slot 2 = rec(30)).
    // The scan order from cursor=3: slots 3 (empty), 0 (rec 10), 1 (zeroed -> skip), 2 (rec 30).
    assert_eq!(
        records.len(),
        2,
        "Fix: zeroed slot must be skipped, yielding 2 out of 3 records; got: {:?}",
        records.iter().map(|r| r.slot_idx).collect::<Vec<_>>()
    );
    // Record 10 must come before record 30 in publish order.
    assert_eq!(
        records[0].slot_idx, 10,
        "Fix: first replayed record must be slot_idx=10"
    );
    assert_eq!(
        records[1].slot_idx, 30,
        "Fix: second replayed record must be slot_idx=30"
    );
}
