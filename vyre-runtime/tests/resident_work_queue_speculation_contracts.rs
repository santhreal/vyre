//! Contracts for `vyre_runtime::resident_work_queue::speculation`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::autotune_store::{AutotuneRecord, AutotuneStore};
use vyre_driver::speculation_verdict::{
    decide_speculation, SpeculationObservation, SpeculationVerdict,
};
use vyre_driver::{
    record_speculative_variant_race, SpeculativeVariantDecision, SpeculativeVariantKeys,
    SpeculativeVariantRace,
};
use vyre_runtime::resident_work_queue::speculation::{
    PairedSpeculationSample, PairedSpeculationWindow,
};

use vyre_driver::SpecCacheKey;
use vyre_driver::SpeculativeVariantKind;

fn key(id: u64) -> SpecCacheKey {
    SpecCacheKey {
        shader_hash: id,
        binding_sig: id << 8,
        workgroup_size: [64, 1, 1],
        spec_hash: id << 16,
    }
}

fn record(workgroup: u32) -> AutotuneRecord {
    AutotuneRecord {
        workgroup_size: [workgroup, 1, 1],
        unroll: 1,
        tile: [0, 0, 0],
        recorded_at: "2026-05-02".to_string(),
    }
}

fn sample(conservative_ns: u64, speculative_ns: u64) -> PairedSpeculationSample {
    PairedSpeculationSample {
        conservative_dispatch_ns: conservative_ns,
        speculative_dispatch_ns: speculative_ns,
        conservative_compile_ns: 0,
        speculative_compile_ns: 0,
        conservative_record: record(64),
        speculative_record: record(128),
    }
}

#[test]
fn paired_window_keeps_racing_under_threshold() {
    let mut store = AutotuneStore::default();
    let conservative = key(1);
    let speculative = key(2);
    let keys = SpeculativeVariantKeys {
        conservative: &conservative,
        speculative: &speculative,
        adapter_id: "test-adapter",
    };
    let mut window = PairedSpeculationWindow::new();
    let update = window.record_sample(&mut store, keys, sample(100_000, 50_000));
    assert_eq!(update.verdict, SpeculationVerdict::KeepRacing);
    assert_eq!(update.observation.baseline_dispatches, 1);
    assert_eq!(update.observation.speculative_dispatches, 1);
}

#[test]
fn paired_window_adopts_after_sustained_win() {
    let mut store = AutotuneStore::default();
    let conservative = key(3);
    let speculative = key(4);
    let keys = SpeculativeVariantKeys {
        conservative: &conservative,
        speculative: &speculative,
        adapter_id: "test-adapter",
    };
    let mut window = PairedSpeculationWindow::new();
    let mut last = None;
    for _ in 0..8 {
        last = Some(window.record_sample(&mut store, keys, sample(100_000, 50_000)));
    }
    let update = last.expect("Fix: loop records at least one sample");
    assert_eq!(update.verdict, SpeculationVerdict::Adopt);
    assert_eq!(
        update.race_decision.winner,
        SpeculativeVariantKind::Speculative
    );
    assert_eq!(store.len(), 1);
}

#[test]
fn paired_window_rejects_sustained_loss() {
    let mut store = AutotuneStore::default();
    let conservative = key(5);
    let speculative = key(6);
    let keys = SpeculativeVariantKeys {
        conservative: &conservative,
        speculative: &speculative,
        adapter_id: "test-adapter",
    };
    let mut window = PairedSpeculationWindow::new();
    let mut verdict = SpeculationVerdict::KeepRacing;
    for _ in 0..8 {
        verdict = window
            .record_sample(&mut store, keys, sample(50_000, 100_000))
            .verdict;
    }
    assert_eq!(verdict, SpeculationVerdict::Reject);
}

#[test]
fn paired_window_amortizes_speculative_compile_cost() {
    let mut store = AutotuneStore::default();
    let conservative = key(7);
    let speculative = key(8);
    let keys = SpeculativeVariantKeys {
        conservative: &conservative,
        speculative: &speculative,
        adapter_id: "test-adapter",
    };
    let mut window = PairedSpeculationWindow::new();
    let mut update = None;
    for _ in 0..8 {
        let mut s = sample(100_000, 50_000);
        s.speculative_compile_ns = 1_000_000;
        update = Some(window.record_sample(&mut store, keys, s));
    }
    let update = update.expect("Fix: loop records at least one sample");
    assert_eq!(update.verdict, SpeculationVerdict::Reject);
    assert_eq!(update.observation.side_compile_cost_ns, 8_000_000);
}
