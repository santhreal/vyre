//! WHY: every resident-ring geometry, capacity, overflow, bounds, and protocol
//! fault was reported as `PipelineError::QueueFull`, whose message states that
//! an io_uring submission queue is at capacity. A slot index past the ring, a
//! ring length off a slot multiple, and a publish over an in-flight slot all
//! rendered as a full queue, and the only assertion protecting them was
//! `matches!(err, PipelineError::QueueFull { .. })`, which every one of those
//! faults satisfies equally. Those assertions could not fail on a call that
//! started failing for a different reason.
//!
//! Closes: the transition-legality space, the fault classification space, and
//! the naming of both. Every case below is derived from `RingSlotTransition`,
//! `RingEncodingFault`, and `slot::STATUSES` at run time, so a transition, a
//! fault class, or a status word added without a recorded decision turns this
//! suite red rather than passing silently.
//!
//! Does not catch: a call site classified into the wrong fault. The fault a
//! given call reports is asserted at that call's own test.

use vyre_runtime::resident_work_queue::protocol::{slot, SLOT_WORDS, STATUS_WORD};
use vyre_runtime::resident_work_queue::{ResidentWorkQueue, RingSlotTransition};
use vyre_runtime::{PipelineError, RingEncodingFault};

/// Exhaustive with no catch-all: adding a transition fails to compile here, and
/// the length assertion below then fails until `ALL` lists it.
fn transition_index(transition: RingSlotTransition) -> usize {
    match transition {
        RingSlotTransition::Publish => 0,
        RingSlotTransition::Claim => 1,
        RingSlotTransition::Done => 2,
        RingSlotTransition::Fault => 3,
        RingSlotTransition::Cancel => 4,
    }
}

/// Exhaustive with no catch-all, for the same reason.
fn fault_index(fault: RingEncodingFault) -> usize {
    match fault {
        RingEncodingFault::Geometry => 0,
        RingEncodingFault::Capacity => 1,
        RingEncodingFault::Overflow => 2,
        RingEncodingFault::OutOfBounds => 3,
        RingEncodingFault::Protocol => 4,
    }
}

fn status_offset(slot_idx: u32) -> usize {
    (slot_idx * SLOT_WORDS + STATUS_WORD) as usize * 4
}

fn ring_with_status(slot_count: u32, slot_idx: u32, status: u32) -> Vec<u8> {
    let mut ring =
        ResidentWorkQueue::encode_empty_ring(slot_count).expect("a small ring must encode");
    let at = status_offset(slot_idx);
    ring[at..at + 4].copy_from_slice(&status.to_le_bytes());
    ring
}

fn read_status(ring: &[u8], slot_idx: u32) -> u32 {
    let at = status_offset(slot_idx);
    u32::from_le_bytes(ring[at..at + 4].try_into().expect("four status bytes"))
}

#[test]
fn every_transition_is_listed_in_all() {
    assert_eq!(
        RingSlotTransition::ALL.len(),
        5,
        "a transition was added to the enum without extending ALL"
    );
    for (position, transition) in RingSlotTransition::ALL.iter().enumerate() {
        assert_eq!(
            transition_index(*transition),
            position,
            "ALL must list transitions in index order so the space is covered once each"
        );
    }
}

#[test]
fn every_fault_is_listed_in_all() {
    assert_eq!(
        RingEncodingFault::ALL.len(),
        5,
        "a fault class was added to the enum without extending ALL"
    );
    for (position, fault) in RingEncodingFault::ALL.iter().enumerate() {
        assert_eq!(
            fault_index(*fault),
            position,
            "ALL must list faults in index order so the space is covered once each"
        );
    }
}

#[test]
fn every_transition_names_itself_distinctly() {
    let mut names: Vec<&str> = RingSlotTransition::ALL
        .iter()
        .map(|transition| transition.label())
        .collect();
    names.sort_unstable();
    let distinct = names.len();
    names.dedup();
    assert_eq!(
        names.len(),
        distinct,
        "two transitions share a label, so a rejection message cannot say which one was attempted"
    );
    for transition in RingSlotTransition::ALL {
        assert!(
            !transition.label().is_empty(),
            "{transition:?} has no label for a rejection message"
        );
    }
}

#[test]
fn every_fault_describes_itself_distinctly() {
    let mut descriptions: Vec<String> = RingEncodingFault::ALL
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    descriptions.sort();
    let distinct = descriptions.len();
    descriptions.dedup();
    assert_eq!(
        descriptions.len(),
        distinct,
        "two fault classes render the same text, so a message cannot say which class was detected"
    );
    for fault in RingEncodingFault::ALL {
        let rendered = fault.to_string();
        assert!(!rendered.is_empty(), "{fault:?} renders as an empty string");
        assert!(
            !rendered.contains("queue at capacity"),
            "{fault:?} renders as an io_uring queue-full message: {rendered}"
        );
    }
}

#[test]
fn every_status_word_round_trips_through_one_name() {
    let mut names: Vec<&str> = slot::STATUSES.iter().map(|(_, name)| *name).collect();
    names.sort_unstable();
    let distinct = names.len();
    names.dedup();
    assert_eq!(names.len(), distinct, "two status words share a name");

    for (word, name) in slot::STATUSES {
        assert_eq!(
            slot::status_name(word),
            Some(name),
            "status word {word} must resolve to exactly the name the table gives it"
        );
    }
    let undefined = slot::STATUSES
        .iter()
        .map(|(word, _)| *word)
        .max()
        .expect("the status table is not empty")
        + 1;
    assert_eq!(
        slot::status_name(undefined),
        None,
        "a word the protocol never defined must not be given a name"
    );
}

#[test]
fn a_transition_is_legal_from_exactly_the_statuses_it_permits() {
    for transition in RingSlotTransition::ALL {
        let permitted = transition.permitted();
        assert!(
            !permitted.is_empty(),
            "{transition:?} permits no status, so it can never succeed"
        );
        for word in permitted {
            assert!(
                slot::status_name(*word).is_some(),
                "{transition:?} permits {word}, which is not a defined status word"
            );
        }
    }
}

/// The whole matrix: five transitions against every defined status word. A
/// legal pair must be accepted and must write the transition's target status; an
/// illegal pair must be rejected as an illegal transition carrying the status
/// the slot actually held, never as a full io_uring queue.
#[test]
fn every_transition_status_pair_is_accepted_or_rejected_by_its_permitted_set() {
    for transition in RingSlotTransition::ALL {
        for (word, name) in slot::STATUSES {
            let mut ring = ring_with_status(2, 0, word);
            let outcome = ResidentWorkQueue::transition_slot_status(&mut ring, 0, transition);

            // Publish is refused through this entry point regardless of status:
            // it would flip the status word before any payload word is written.
            if transition == RingSlotTransition::Publish {
                assert!(
                    matches!(
                        outcome,
                        Err(PipelineError::RingEncoding {
                            fault: RingEncodingFault::Protocol,
                            ..
                        })
                    ),
                    "publish through transition_slot_status must report a Protocol fault \
                     from {name}, got {outcome:?}"
                );
                continue;
            }

            if transition.permitted().contains(&word) {
                let previous = outcome.unwrap_or_else(|error| {
                    panic!("{transition:?} must be legal from {name}, got {error:?}")
                });
                assert_eq!(
                    previous, word,
                    "{transition:?} must report the status it replaced"
                );
                assert_eq!(
                    read_status(&ring, 0),
                    transition.target_status(),
                    "{transition:?} from {name} must leave the slot at its target status"
                );
                continue;
            }

            let error = outcome.expect_err(&format!("{transition:?} must be illegal from {name}"));
            match error {
                PipelineError::IllegalSlotTransition {
                    transition: named,
                    permitted,
                    current_status,
                    ..
                } => {
                    assert_eq!(
                        named,
                        transition.label(),
                        "the rejection must name the attempted transition"
                    );
                    assert_eq!(
                        permitted,
                        transition.permitted(),
                        "the rejection must state the same permitted set the predicate enforces"
                    );
                    assert_eq!(
                        current_status, word,
                        "the rejection must state the status the slot held, not a placeholder"
                    );
                }
                other => panic!(
                    "{transition:?} from {name} must be an illegal transition, got {other:?}"
                ),
            }

            assert_eq!(
                read_status(&ring, 0),
                word,
                "a rejected {transition:?} must not write the status word"
            );
        }
    }
}

#[test]
fn an_illegal_transition_message_states_the_status_by_name() {
    // CLAIMED is not claimable: Claim permits PUBLISHED, YIELD, and REQUEUE.
    let mut ring = ring_with_status(1, 0, slot::CLAIMED);
    let error = ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Claim)
        .expect_err("claiming a CLAIMED slot must be rejected");
    let rendered = error.to_string();
    assert!(
        rendered.contains("claim"),
        "the message must name the attempted transition: {rendered}"
    );
    assert!(
        rendered.contains("CLAIMED"),
        "the message must name the status the slot held: {rendered}"
    );
    assert!(
        rendered.contains("PUBLISHED"),
        "the message must name the statuses claim permits: {rendered}"
    );
    assert!(
        !rendered.contains("queue at capacity"),
        "an illegal transition must not be reported as a full io_uring queue: {rendered}"
    );
}
