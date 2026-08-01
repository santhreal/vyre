//! A resident handle is only valid on the backend instance that minted it.
//!
//! Resident ids come from a counter private to one `CudaBackend`, so two
//! instances hand out the same small ids. Before handles carried their owner,
//! presenting instance A's handle to instance B resolved against B's live
//! buffer of the same id: same size, different contents, no diagnostic. These
//! tests pin the refusal that replaced that silent hit.
//!
//! Scoping a handle to one call frame, which is what callers did instead, is
//! safe only because it never reuses. It buys that safety by re-uploading the
//! payload on every call, and the moment a caller keeps the handle on a
//! long-lived object to stop paying that (the whole point of resident memory)
//! the window opens. Concurrency does not open it by itself: separate
//! instances running at once with colliding ids still never alias, which
//! `concurrent_instances_handing_out_the_same_local_id_never_alias` witnesses
//! directly. What concurrency changes is the odds once a handle DOES cross,
//! because every live instance holds a buffer under the same low ids.

use vyre_driver::{BackendError, Resource, VyreBackend};
use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration};

fn acquire() -> CudaBackendRegistration {
    CudaBackendRegistration::new(
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host."),
    )
}

fn assert_refused(result: Result<impl std::fmt::Debug, BackendError>, operation: &str) {
    let error = match result {
        Ok(value) => panic!(
            "Fix: {operation} accepted a resident handle from another backend instance and produced {value:?}. A foreign handle must be refused, never resolved by bare id."
        ),
        Err(error) => error,
    };
    let BackendError::InvalidProgram { fix } = error else {
        panic!("Fix: {operation} must refuse a foreign resident handle with BackendError::InvalidProgram, got {error:?}");
    };
    assert!(
        fix.contains("owned by backend instance") && fix.contains("Fix: "),
        "Fix: {operation} refusal must name the owning instance and carry actionable repair text, got {fix}"
    );
}

#[test]
fn foreign_resident_handle_is_refused_by_every_transfer_entry_point() {
    let first = acquire();
    let second = acquire();

    let owned_by_first = first
        .allocate_resident(32)
        .expect("Fix: instance A resident allocation failed.");
    let owned_by_second = second
        .allocate_resident(32)
        .expect("Fix: instance B resident allocation failed.");

    first
        .upload_resident(&owned_by_first, &[0xAAu8; 32])
        .expect("Fix: instance A resident upload failed.");
    second
        .upload_resident(&owned_by_second, &[0xBBu8; 32])
        .expect("Fix: instance B resident upload failed.");

    assert_refused(
        second.download_resident(&owned_by_first),
        "cross-instance download",
    );
    assert_refused(
        first.download_resident(&owned_by_second),
        "cross-instance download",
    );
    assert_refused(
        second.upload_resident(&owned_by_first, &[0xCCu8; 32]),
        "cross-instance upload",
    );
    assert_refused(
        second.free_resident(owned_by_first.clone()),
        "cross-instance free",
    );

    // The owning instance still sees exactly what it wrote, so refusal costs
    // the legitimate holder nothing.
    let readback = first
        .download_resident(&owned_by_first)
        .expect("Fix: the owning instance must still resolve its own handle.");
    assert_eq!(readback, vec![0xAAu8; 32]);

    first
        .free_resident(owned_by_first)
        .expect("Fix: the owning instance must be able to free its own handle.");
    second
        .free_resident(owned_by_second)
        .expect("Fix: the owning instance must be able to free its own handle.");
}

#[test]
fn two_instances_mint_distinct_handles_for_the_same_local_id() {
    let first = acquire();
    let second = acquire();

    let a = first
        .allocate_resident(32)
        .expect("Fix: instance A resident allocation failed.");
    let b = second
        .allocate_resident(32)
        .expect("Fix: instance B resident allocation failed.");

    let (Resource::Resident(a_handle), Resource::Resident(b_handle)) = (&a, &b) else {
        panic!("Fix: allocate_resident must return Resource::Resident");
    };
    assert_eq!(
        a_handle.id(),
        b_handle.id(),
        "Fix: this test only witnesses the hazard when both instances hand out the same local id"
    );
    assert_ne!(
        a_handle, b_handle,
        "Fix: handles from different backend instances must never compare equal, or a stale handle silently names a live foreign buffer"
    );
    assert_ne!(a_handle.owner(), b_handle.owner());

    first.free_resident(a).expect("Fix: free on owner failed.");
    second.free_resident(b).expect("Fix: free on owner failed.");
}

/// Buffer size every concurrent thread allocates, so an alias would fit.
///
/// Aliasing is only invisible when the wrong buffer is the right SIZE. Equal
/// sizes everywhere is what makes a miss silent instead of a length error.
const CONCURRENT_BYTES: usize = 16 * 1024;

/// Threads in the concurrency witnesses. Enough to interleave allocation.
const CONCURRENT_THREADS: usize = 8;

/// Allocate/upload/download rounds each concurrent thread performs.
const CONCURRENT_ROUNDS: usize = 24;

#[test]
fn concurrent_instances_handing_out_the_same_local_id_never_alias() {
    // Concurrency alone does NOT create a crossing. Each thread allocates,
    // uploads, reads back and frees against its OWN instance, which is how a
    // per-call-frame caller uses resident memory. The instances hand out
    // colliding local ids the whole time (asserted below), so if colliding ids
    // were sufficient to alias, this would read another thread's fill byte.
    let observed_ids: Vec<Vec<u64>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..CONCURRENT_THREADS)
            .map(|index| {
                scope.spawn(move || {
                    let backend = acquire();
                    let fill = u8::try_from(index + 1).expect("thread count fits u8");
                    let mut ids = Vec::with_capacity(CONCURRENT_ROUNDS);
                    for round in 0..CONCURRENT_ROUNDS {
                        let resource = backend
                            .allocate_resident(CONCURRENT_BYTES)
                            .expect("Fix: concurrent resident allocation failed.");
                        let Resource::Resident(handle) = &resource else {
                            panic!("Fix: allocate_resident must return Resource::Resident");
                        };
                        ids.push(handle.id());
                        backend
                            .upload_resident(&resource, &vec![fill; CONCURRENT_BYTES])
                            .expect("Fix: concurrent resident upload failed.");
                        let readback = backend
                            .download_resident(&resource)
                            .expect("Fix: concurrent resident download failed.");
                        assert_eq!(
                            readback.len(),
                            CONCURRENT_BYTES,
                            "Fix: thread {index} round {round} read a buffer of the wrong length"
                        );
                        assert!(
                            readback.iter().all(|byte| *byte == fill),
                            "Fix: thread {index} round {round} read another instance's bytes. A resident handle must resolve only against the instance that minted it."
                        );
                        backend
                            .free_resident(resource)
                            .expect("Fix: concurrent resident free failed.");
                    }
                    ids
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("Fix: a witness thread panicked"))
            .collect()
    });

    // The precondition for the hazard held throughout: separate instances were
    // live at the same time and handed out the same local ids.
    let first = &observed_ids[0];
    assert!(
        observed_ids
            .iter()
            .skip(1)
            .any(|ids| ids.iter().any(|id| first.contains(id))),
        "Fix: this witness only means something when concurrent instances reuse local ids, and none collided"
    );
}

#[test]
fn a_handle_crossing_threads_between_live_instances_is_refused() {
    // The crossing the previous test does not perform: a handle held past its
    // own instance while BOTH instances are hot. This is what a caller does the
    // moment it caches a resident handle on a long-lived object instead of
    // scoping it to one call, which is exactly the reuse this change exists to
    // make safe. It must fail closed rather than resolve.
    let lender = acquire();
    let lent = lender
        .allocate_resident(CONCURRENT_BYTES)
        .expect("Fix: lender resident allocation failed.");
    lender
        .upload_resident(&lent, &vec![0xA5u8; CONCURRENT_BYTES])
        .expect("Fix: lender resident upload failed.");

    std::thread::scope(|scope| {
        let lent = &lent;
        for index in 0..CONCURRENT_THREADS {
            scope.spawn(move || {
                let borrower = acquire();
                // Keep the borrower's own id counter moving so it holds a live
                // buffer under the same local id the lent handle names.
                let own = borrower
                    .allocate_resident(CONCURRENT_BYTES)
                    .expect("Fix: borrower resident allocation failed.");
                borrower
                    .upload_resident(&own, &vec![0x5Au8; CONCURRENT_BYTES])
                    .expect("Fix: borrower resident upload failed.");

                // What makes the refusal load-bearing rather than decorative:
                // the lent handle names a local id the borrower ALSO has live,
                // at the same size. A bare-id lookup here would hit the
                // borrower's own buffer and return 0x5A bytes as if they were
                // the lender's.
                let (Resource::Resident(lent_handle), Resource::Resident(own_handle)) =
                    (lent, &own)
                else {
                    panic!("Fix: allocate_resident must return Resource::Resident");
                };
                assert_eq!(
                    lent_handle.id(),
                    own_handle.id(),
                    "Fix: this witness only means something when the foreign handle collides with a live local id on thread {index}"
                );
                assert_ne!(lent_handle.owner(), own_handle.owner());

                assert_refused(
                    borrower.download_resident(lent),
                    &format!("concurrent cross-instance download on thread {index}"),
                );
                assert_refused(
                    borrower.upload_resident(lent, &vec![0xFFu8; CONCURRENT_BYTES]),
                    &format!("concurrent cross-instance upload on thread {index}"),
                );

                // The refusal left the borrower's own buffer untouched.
                let own_bytes = borrower
                    .download_resident(&own)
                    .expect("Fix: borrower must still resolve its own handle.");
                assert!(
                    own_bytes.iter().all(|byte| *byte == 0x5A),
                    "Fix: a refused foreign handle must not disturb the borrower's own buffer"
                );
                borrower
                    .free_resident(own)
                    .expect("Fix: borrower resident free failed.");
            });
        }
    });

    // The lender's bytes survived every refused access.
    let lender_bytes = lender
        .download_resident(&lent)
        .expect("Fix: the owning instance must still resolve its own handle.");
    assert!(
        lender_bytes.iter().all(|byte| *byte == 0xA5),
        "Fix: refused cross-instance writes must never reach the owner's buffer"
    );
    lender
        .free_resident(lent)
        .expect("Fix: lender resident free failed.");
}
