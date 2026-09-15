//! Proving a case must not cost a backend acquisition.
//!
//! # Why this exists
//!
//! Conformance asked every registered backend for its compile facts once per
//! case, and twice more per case pair, to recompute one constant: the facts
//! describe the device the artifact will run on and do not change while the
//! process lives. A 353-operation run over sixteen workers therefore acquired
//! each backend more than a thousand times and held many of those acquisitions
//! open at once.
//!
//! An acquisition is not free and is not meant to be. A wgpu acquisition
//! creates its own device, which is what lets device-loss recovery replace one
//! backend's device without disturbing another's; sixty-four held at once open
//! 629 file descriptors on an NVIDIA host. The SPIR-V driver additionally built
//! a whole Vulkan context per acquisition, loader included. The process reached
//! its descriptor limit mid-run, `dlopen` of `libvulkan.so.1` failed with `Too
//! many open files`, and 269 of 288 cases were refused for want of a Vulkan
//! loader on a host that had one. Tearing those contexts down again unloaded
//! the ICD under the wgpu driver's process-wide instance, so every wgpu route
//! in the same run reported `Parent device is lost` and the certificate covered
//! nothing.
//!
//! Descriptors are a process resource, so one driver exhausting them refuses
//! every other driver's work, and none of the resulting output distinguishes it
//! from a host with no GPU. The assertion is therefore the acquisition count
//! itself rather than a resource the exhaustion happens to reach first: reading
//! facts for any number of cases acquires each backend once. It is read from
//! the registry at run time, so a backend added later is covered without this
//! file being edited.
//!
//! # What it does not catch
//!
//! Only the facts path. A route that acquires a backend per case elsewhere
//! passes this and is the same defect, and a driver that leaks memory or
//! threads per acquisition is invisible here.

use vyre_conform::backend_selection::{semantic_execution_backends, target_facts_acquisitions};
use vyre_conform::{registered_target_facts, target_facts_digest};

/// Cases a proof run of this workspace's operation set is worth. Facts are read
/// at least once per case, so this is the repetition the invariant survives.
const CASES: usize = 256;

#[test]
fn reading_target_facts_once_per_case_acquires_each_backend_once() {
    let backends = semantic_execution_backends()
        .expect("the registry initializes in a binary that links concrete drivers");
    assert!(
        !backends.is_empty(),
        "no semantic-execution backend is linked; this target requires `device-tests`, which pulls in `gpu`"
    );

    let mut measured = 0_usize;
    for backend in backends {
        // Resolve once first. Whether this is the acquisition or a hit from an
        // earlier test in the same binary, the count below starts from what the
        // backend has already cost and measures only the repetition.
        let Some(first) = target_facts_digest(backend.id) else {
            continue;
        };

        let baseline = target_facts_acquisitions(backend.id);

        for case in 1..CASES {
            let digest = target_facts_digest(backend.id).unwrap_or_else(|| {
                panic!(
                    "backend `{}` reported target facts and then reported none at case {case}. Facts describe the device the artifact runs on, and a pair whose two sides disagree about them is refused, so a digest that comes and goes refuses cases by scheduling.",
                    backend.id
                )
            });
            assert_eq!(
                digest, first,
                "backend `{}` digested different target facts at case {case} than at case 0",
                backend.id
            );
        }

        assert_eq!(
            target_facts_acquisitions(backend.id),
            baseline,
            "backend `{}` was acquired again while reading one constant for {} further cases. A real run holds those acquisitions open across its workers, reaches the process descriptor limit, and the loader then refuses the next driver that asks it for a file.",
            backend.id,
            CASES - 1
        );
        measured += 1;
    }

    assert!(
        measured > 0,
        "no linked backend could be acquired on this host, so nothing was measured"
    );
}

#[test]
fn the_facts_a_policy_compiles_against_are_the_facts_a_receipt_digests() {
    let backends = semantic_execution_backends()
        .expect("the registry initializes in a binary that links concrete drivers");

    for backend in backends {
        let Ok(facts) = registered_target_facts(backend) else {
            continue;
        };
        let digest = target_facts_digest(backend.id).unwrap_or_else(|| {
            panic!(
                "backend `{}` resolved target facts for the compiler and none for the receipt. A verifier compares the receipt digest against the facts the artifact compiled against, so two answers for one backend refuse every pair it produced.",
                backend.id
            )
        });
        assert_eq!(
            digest,
            blake3::hash(format!("{facts:?}").as_bytes())
                .to_hex()
                .to_string(),
            "backend `{}` digests target facts that are not the ones it compiles against",
            backend.id
        );
    }
}

#[test]
fn a_refused_backend_refuses_every_acquisition_the_same_way() {
    let backends = semantic_execution_backends()
        .expect("the registry initializes in a binary that links concrete drivers");

    for backend in backends {
        let Err(first) = backend.acquire() else {
            continue;
        };
        let second = backend.acquire().err().unwrap_or_else(|| {
            panic!(
                "backend `{}` refused acquisition and then admitted one. A host either has the backend or does not, and a refusal that depends on call order makes a certificate's backend coverage depend on scheduling.",
                backend.id
            )
        });
        assert_eq!(
            first.to_string(),
            second.to_string(),
            "backend `{}` gave two different refusals for the same host",
            backend.id
        );
    }
}
