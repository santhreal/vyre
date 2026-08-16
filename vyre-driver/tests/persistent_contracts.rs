//! Contracts for `vyre_driver::persistent`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::persistent::{PersistentEngine, PersistentWorkItem, QueueFull};

use std::sync::Arc;
use std::thread;

fn item(i: u32) -> PersistentWorkItem {
    PersistentWorkItem {
        input_offset: i * 1024,
        input_len: 1024,
        rule_set_id: 0,
        correlation: i,
    }
}

#[test]
fn invalid_ring_size_has_explicit_error_api() {
    let err = PersistentEngine::try_new(7).unwrap_err();
    assert!(err.contains("Fix:"));
    assert!(PersistentEngine::try_new(0).is_err());
}

#[test]
fn infallible_constructor_normalizes_ring_size() {
    assert_eq!(PersistentEngine::new(7).ring_size(), 8);
    assert_eq!(PersistentEngine::new(0).ring_size(), 1);
}

#[test]
fn enqueue_claim_fifo_single_thread() {
    let eng = PersistentEngine::new(8);
    for i in 0..8 {
        assert_eq!(eng.enqueue(item(i)).unwrap(), i);
    }
    for i in 0..8 {
        assert_eq!(eng.claim().unwrap().correlation, i);
    }
    assert!(eng.claim().is_none());
}

#[test]
fn queue_full_on_overflow() {
    let eng = PersistentEngine::new(4);
    for i in 0..4 {
        eng.enqueue(item(i)).unwrap();
    }
    assert_eq!(eng.enqueue(item(99)), Err(QueueFull));
}

#[test]
fn space_reclaims_after_claim() {
    let eng = PersistentEngine::new(4);
    for i in 0..4 {
        eng.enqueue(item(i)).unwrap();
    }
    assert!(eng.enqueue(item(99)).is_err());
    let claimed = eng.claim().unwrap();
    assert_eq!(claimed.correlation, 0);
    assert!(eng.enqueue(item(99)).is_ok());
}

#[test]
fn in_flight_tracks_correctly() {
    let eng = PersistentEngine::new(16);
    assert_eq!(eng.in_flight(), 0);
    for i in 0..5 {
        eng.enqueue(item(i)).unwrap();
    }
    assert_eq!(eng.in_flight(), 5);
    eng.claim().unwrap();
    eng.claim().unwrap();
    assert_eq!(eng.in_flight(), 3);
}

#[test]
fn done_marker_flows_through() {
    let eng = PersistentEngine::new(4);
    let slot = eng.enqueue(item(1)).unwrap();
    assert!(!eng.is_done(slot).unwrap());
    let claimed = eng.claim().unwrap();
    assert_eq!(claimed.correlation, 1);
    eng.mark_done(slot).unwrap();
    assert!(eng.is_done(slot).unwrap());
}

#[test]
fn multi_producer_single_consumer_no_item_lost() {
    let eng = Arc::new(PersistentEngine::new(128));
    let producers = 4;
    let items_per_producer = 16;
    let mut handles = Vec::new();
    for p in 0..producers {
        let eng = Arc::clone(&eng);
        handles.push(thread::spawn(move || {
            for i in 0..items_per_producer {
                let corr = (p * 1000 + i) as u32;
                loop {
                    if eng.enqueue(item(corr)).is_ok() {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        }));
    }
    let consumer_eng = Arc::clone(&eng);
    let consumer = thread::spawn(move || {
        let total = (producers * items_per_producer) as usize;
        let mut seen = Vec::with_capacity(total);
        while seen.len() < total {
            if let Some(it) = consumer_eng.claim() {
                seen.push(it.correlation);
            } else {
                std::hint::spin_loop();
            }
        }
        seen
    });
    for h in handles {
        h.join().unwrap();
    }
    let seen = consumer.join().unwrap();
    let mut sorted = seen.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), seen.len(), "duplicate items consumed");
    for p in 0..producers {
        for i in 0..items_per_producer {
            let expected = (p * 1000 + i) as u32;
            assert!(
                seen.contains(&expected),
                "missing correlation id {expected}"
            );
        }
    }
}

#[test]
fn wrap_around_works_for_large_throughput() {
    let eng = PersistentEngine::new(16);
    let passes = 10;
    for p in 0..passes {
        for i in 0..16 {
            let corr = (p * 1000 + i) as u32;
            assert!(eng.enqueue(item(corr)).is_ok());
        }
        for i in 0..16 {
            let corr = (p * 1000 + i) as u32;
            assert_eq!(eng.claim().unwrap().correlation, corr);
        }
    }
    assert_eq!(eng.head(), (passes * 16) as u32);
    assert_eq!(eng.tail(), (passes * 16) as u32);
    assert_eq!(eng.in_flight(), 0);
}

#[test]
fn multi_consumer_no_double_claim() {
    let eng = Arc::new(PersistentEngine::new(128));
    let total = 100_u32;
    for i in 0..total {
        eng.enqueue(item(i)).unwrap();
    }
    let consumers = 4;
    let mut handles = Vec::new();
    let shared_consumed = Arc::new(std::sync::Mutex::new(Vec::new()));
    for _ in 0..consumers {
        let eng = Arc::clone(&eng);
        let out = Arc::clone(&shared_consumed);
        handles.push(thread::spawn(move || {
            let mut local = Vec::new();
            while let Some(it) = eng.claim() {
                local.push(it.correlation);
            }
            out.lock().unwrap().extend(local);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let mut consumed = Arc::try_unwrap(shared_consumed)
        .unwrap()
        .into_inner()
        .unwrap();
    consumed.sort();
    assert_eq!(consumed.len(), total as usize);
    for (i, c) in consumed.iter().enumerate() {
        assert_eq!(*c, i as u32, "duplicated or missing item at idx {i}");
    }
}

#[test]
fn queue_full_error_display_is_useful() {
    let s = format!("{QueueFull}");
    assert!(s.contains("ring buffer"));
}
