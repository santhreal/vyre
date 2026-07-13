//! W3-2 evidence: the async scan path supports a TWO-BATCH PIPELINE, two
//! dispatches in flight at once, each decoding to the correct result.
//!
//! The single-path async tests prove `async == sync` for one dispatch. This gate
//! proves the property the "overlap on a two-batch pipeline" evidence bullet
//! names: a caller can SUBMIT scan A, SUBMIT scan B (both now in flight), then
//! await both, the second submit does not have to wait for the first to finish,
//! and neither handle's result is corrupted by the other's in-flight buffers.
//!
//! It asserts the PIPELINE CORRECTNESS (both results exact) rather than a brittle
//! wall-clock overlap ratio: on a pipelining backend (wgpu/cuda) the two run
//! concurrently; on the synchronous default they serialize, either way both
//! results must be exact, and a corruption from sharing/reusing device buffers
//! across two in-flight handles would surface as a wrong bitmap or triple set.
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_async_two_batch_pipeline --release -- --nocapture

use vyre_driver_reference::CpuRefBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

const LITERALS: &[&[u8]] = &[b"alpha", b"kilo", b"tango"];
const MAX_MATCHES: u32 = 128;

/// Two DIFFERENT batches so a cross-handle buffer mixup would produce a wrong
/// (not coincidentally-equal) result: batch A fires `alpha`+`tango`, batch B
/// fires `kilo` only.
fn batch_a() -> Vec<u8> {
    b"__alpha__then__tango__and__alpha__again".to_vec()
}
fn batch_b() -> Vec<u8> {
    b"....kilo....kilo....".to_vec()
}

/// Drive the two-batch pipeline on `backend`: submit both presence scans, hold
/// both handles in flight, then await both and check each against its sync twin.
fn check_two_batch_presence<B: vyre::VyreBackend + ?Sized>(backend: &B) {
    let set = GpuLiteralSet::compile(LITERALS);
    let a = batch_a();
    let b = batch_b();

    let sync_a = set.scan_presence(backend, &a).expect("sync A");
    let sync_b = set.scan_presence(backend, &b).expect("sync B");
    // The two batches must genuinely differ, else the test can't detect a mixup.
    assert_ne!(
        sync_a, sync_b,
        "fixture batches must produce different bitmaps"
    );

    // BOTH in flight before EITHER await.
    let pending_a = set.scan_presence_async(backend, &a).expect("submit A");
    let pending_b = set.scan_presence_async(backend, &b).expect("submit B");

    // Await in submit order; each must decode to its OWN batch's result.
    let got_a = pending_a.await_words().expect("await A");
    let got_b = pending_b.await_words().expect("await B");

    assert_eq!(
        got_a, sync_a,
        "batch A async result must equal its sync twin"
    );
    assert_eq!(
        got_b, sync_b,
        "batch B async result must equal its sync twin"
    );
}

/// Same, but await in REVERSE submit order, a handle must not depend on being
/// awaited first, and the later-submitted scan's buffers must stay valid.
fn check_two_batch_presence_reverse_await<B: vyre::VyreBackend + ?Sized>(backend: &B) {
    let set = GpuLiteralSet::compile(LITERALS);
    let a = batch_a();
    let b = batch_b();
    let sync_a = set.scan_presence(backend, &a).expect("sync A");
    let sync_b = set.scan_presence(backend, &b).expect("sync B");

    let pending_a = set.scan_presence_async(backend, &a).expect("submit A");
    let pending_b = set.scan_presence_async(backend, &b).expect("submit B");
    // Await B first, then A.
    let got_b = pending_b.await_words().expect("await B");
    let got_a = pending_a.await_words().expect("await A");
    assert_eq!(
        got_b, sync_b,
        "batch B (awaited first) must equal its sync twin"
    );
    assert_eq!(
        got_a, sync_a,
        "batch A (awaited second) must equal its sync twin"
    );
}

/// A mixed pipeline: a presence scan and a position scan in flight together 
/// two different program shapes sharing the backend at once.
fn check_mixed_pipeline<B: vyre::VyreBackend + ?Sized>(backend: &B) {
    let set = GpuLiteralSet::compile(LITERALS);
    let a = batch_a();

    let sync_presence = set.scan_presence(backend, &a).expect("sync presence");
    let mut sync_matches: Vec<Match> = Vec::new();
    set.scan_into(backend, &a, MAX_MATCHES, &mut sync_matches)
        .expect("sync scan_into");

    let pending_presence = set
        .scan_presence_async(backend, &a)
        .expect("submit presence");
    let pending_matches = set
        .scan_into_async(backend, &a, MAX_MATCHES)
        .expect("submit matches");

    let got_presence = pending_presence.await_words().expect("await presence");
    let got_matches = pending_matches.await_matches().expect("await matches");

    assert_eq!(
        got_presence, sync_presence,
        "mixed-pipeline presence must match"
    );
    assert_eq!(
        got_matches, sync_matches,
        "mixed-pipeline matches must match"
    );
    assert!(
        !got_matches.is_empty(),
        "position batch must be non-vacuous"
    );
}

#[test]
fn two_batch_pipeline_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping async two-batch pipeline GPU test");
            return;
        }
    };
    check_two_batch_presence(backend.as_ref());
    check_two_batch_presence_reverse_await(backend.as_ref());
    check_mixed_pipeline(backend.as_ref());
}

#[test]
fn two_batch_pipeline_on_cpu_reference() {
    // Synchronous default: the handles serialize, but both results must still be
    // exact (the degraded path must not corrupt or swap batch outputs (Law 10)).
    check_two_batch_presence(&CpuRefBackend);
    check_two_batch_presence_reverse_await(&CpuRefBackend);
    check_mixed_pipeline(&CpuRefBackend);
}
