//! WHY: budgeted device measurement decides which artifact a compilation ships,
//! and a device time is noisy. Without a stated protocol the same search on the
//! same host selects different artifacts from run to run, which is the defect
//! this suite closes: every rule the protocol states is exercised here, and the
//! classes it partitions (verdicts, throttle states, malformed rules) are closed
//! by exhaustive matches rather than by a list that goes stale.
//!
//! What this does not catch: whether a backend's reported device time is itself
//! trustworthy. That is the backend's contract, proved on hardware.

use vyre_megakernel::measure::{
    improves, CandidateMeasurement, DeviceState, MeasurementEnvironment, MeasurementProtocol,
    MeasurementRecord, ReplacementVerdict, SampleEstimate, ThrottleState,
    MEASUREMENT_PROTOCOL_VERSION,
};
use vyre_megakernel::Digest;

/// WHY: the protocol is charged against the measurement budget the caller
/// authorized. A fitted protocol that spends more launches than the budget would
/// make the recorded work exceed the recorded bound, which the artifact refuses
/// to decode; one that counts nothing would report a measured selection with no
/// sample behind it.
#[test]
fn a_fitted_protocol_spends_its_budget_and_counts_at_least_one_sample() {
    for budget in 0..=64_u32 {
        let fitted = MeasurementProtocol::V1.fitted(budget);
        assert!(
            fitted.launches_per_candidate() <= budget,
            "budget {budget} fitted to {} launches",
            fitted.launches_per_candidate()
        );
        if budget == 0 {
            assert_eq!(fitted.max_rounds, 0, "no budget counts no round");
            continue;
        }
        assert!(
            fitted.max_rounds >= 1 && fitted.repetitions_per_round >= 1,
            "budget {budget} must count a sample: {fitted:?}"
        );
        assert!(
            fitted.min_rounds <= fitted.max_rounds,
            "budget {budget} fitted an unreachable minimum: {fitted:?}"
        );
        fitted
            .validate()
            .expect("a fitted protocol must satisfy its own rules");
    }
}

/// WHY: one launch that lands while something else owns the device reads far
/// slower than the schedule is. A mean or an untrimmed median moves with it, and
/// the selection follows. The trimmed median must not.
#[test]
fn one_stalled_launch_does_not_move_the_estimate() {
    let protocol = MeasurementProtocol::V1;
    let quiet = [100_u64, 100, 100, 100, 100];
    let stalled = [100_u64, 100, 100, 100, 40_000];
    let quiet_estimate =
        SampleEstimate::from_samples(&quiet, &protocol).expect("samples reduce to an estimate");
    let stalled_estimate =
        SampleEstimate::from_samples(&stalled, &protocol).expect("samples reduce to an estimate");

    assert_eq!(
        quiet_estimate.estimate_ns, stalled_estimate.estimate_ns,
        "a stalled launch must not change the estimate"
    );
    assert_eq!(stalled_estimate.trimmed, 1, "the slow end must be dropped");
    assert_eq!(
        stalled_estimate.kept, 4,
        "every sample the trim kept must be counted"
    );
    let mean: u64 = stalled.iter().sum::<u64>() / stalled.len() as u64;
    assert!(
        mean > stalled_estimate.estimate_ns * 4,
        "the fixture must actually be a case where a mean would be wrecked: mean {mean}"
    );

    // The trim drops the slow end, so a stall never carries the estimate past
    // what the device actually achieved on its other launches.
    let spread = [100_u64, 101, 100, 102, 40_000];
    let spread_estimate =
        SampleEstimate::from_samples(&spread, &protocol).expect("samples reduce to an estimate");
    assert!(
        (100..=102).contains(&spread_estimate.estimate_ns),
        "the estimate must stay inside the times the device achieved: {spread_estimate:?}"
    );
}

/// WHY: the stopping rule spends launches until an estimate is precise enough to
/// compare, and no longer. An estimate that reports zero spread when its samples
/// scatter would stop sampling on noise; one that never settles on identical
/// samples would spend the whole budget for nothing.
#[test]
fn uncertainty_tracks_the_spread_and_gates_the_stopping_rule() {
    let protocol = MeasurementProtocol::V1;
    let steady = SampleEstimate::from_samples(&[1_000; 8], &protocol).expect("estimate");
    assert_eq!(steady.uncertainty_ns, 0, "identical samples have no spread");
    assert_eq!(steady.relative_uncertainty_permille(), 0);
    assert!(
        steady.is_settled(&protocol),
        "an estimate with no spread is settled"
    );

    let scattered = SampleEstimate::from_samples(
        &[1_000, 1_400, 600, 1_500, 700, 1_300, 900, 1_100],
        &protocol,
    )
    .expect("estimate");
    assert!(
        scattered.uncertainty_ns > 0,
        "scattered samples must report a spread"
    );
    assert!(
        !scattered.is_settled(&protocol),
        "an estimate whose spread exceeds the target must keep sampling: {scattered:?}"
    );
    assert!(
        scattered.relative_uncertainty_permille() > u32::from(protocol.uncertainty_target_permille),
        "the settled decision must be the recorded target, not a hidden constant"
    );
}

/// WHY: a selection that flips on a difference smaller than the measurement
/// error is a selection made by the device. Both guards have to hold: the fixed
/// equivalence band, and the uncertainty the two estimates actually carry.
#[test]
fn a_difference_inside_the_band_or_inside_the_noise_is_not_an_improvement() {
    let protocol = MeasurementProtocol::V1;
    let incumbent = estimate(100_000, 0);

    assert!(
        !improves(&incumbent, &estimate(99_000, 0), &protocol),
        "1000 ns of 100000 is inside the 20-permille band"
    );
    assert!(
        improves(&incumbent, &estimate(90_000, 0), &protocol),
        "10000 ns of 100000 clears the band and the noise"
    );
    assert!(
        !improves(&incumbent, &estimate(90_000, 12_000), &protocol),
        "a margin smaller than the combined spread is not evidence"
    );
    assert!(
        !improves(&incumbent, &estimate(100_000, 0), &protocol),
        "an identical estimate is not an improvement"
    );
    assert!(
        !improves(&incumbent, &estimate(120_000, 0), &protocol),
        "a slower estimate is not an improvement"
    );
}

/// WHY: this is the rule that makes a re-run reproduce its own artifact. Every
/// verdict variant is reachable, and the exhaustive match closes the class: a new
/// variant fails to compile until someone records what it means here.
#[test]
fn rerunning_the_same_protocol_cannot_silently_replace_the_authenticated_winner() {
    let incumbent = record(MeasurementProtocol::V1, &[(1, 100_000)], 0);

    let noise_only = record(MeasurementProtocol::V1, &[(1, 99_500), (2, 99_000)], 1);
    assert_eq!(
        named(noise_only.verdict_against(&incumbent)),
        "equivalent",
        "a challenger inside the band must leave the incumbent standing"
    );

    let decisive = record(MeasurementProtocol::V1, &[(1, 100_000), (2, 50_000)], 1);
    assert_eq!(
        named(decisive.verdict_against(&incumbent)),
        "replaces",
        "a challenger that clears the band takes the selection"
    );

    let same_winner = record(MeasurementProtocol::V1, &[(1, 140_000)], 0);
    assert_eq!(
        named(same_winner.verdict_against(&incumbent)),
        "equivalent",
        "the same candidate measured again is the same selection"
    );

    let mut recalibrated = MeasurementProtocol::V1;
    recalibrated.version = MEASUREMENT_PROTOCOL_VERSION + 1;
    assert_eq!(
        named(record(recalibrated, &[(2, 50_000)], 0).verdict_against(&incumbent)),
        "incomparable",
        "a different protocol version is a recalibration, not a comparison"
    );

    assert_eq!(
        named(
            record_at(MeasurementProtocol::V1, CALIBRATION + 1, &[(1, 100_000)], 0)
                .verdict_against(&incumbent)
        ),
        "incomparable",
        "a recalibrated fact set prices the candidates differently, so the \
         earlier winner carries no authority even when the same candidate wins"
    );

    assert_eq!(
        named(
            record_at(
                MeasurementProtocol::V1,
                CALIBRATION,
                &[(1, 99_500), (2, 99_000)],
                1
            )
            .verdict_against(&incumbent)
        ),
        "equivalent",
        "an unchanged fact set leaves the noise rule in charge"
    );

    let disjoint = record(MeasurementProtocol::V1, &[(9, 50_000)], 0);
    assert_eq!(
        named(disjoint.verdict_against(&incumbent)),
        "incomparable",
        "a session that never measured the authenticated winner cannot rank it"
    );
}

/// WHY: drift and prediction error are the two figures a later recalibration
/// reads. A sign convention nobody can rely on makes both unusable, and a
/// division by an absent figure would panic in the middle of a compilation.
#[test]
fn drift_and_prediction_error_are_signed_permille_of_the_measured_figure() {
    let slowing = MeasurementEnvironment {
        warmup_launches: 2,
        facts_calibration_version: 0,
        first_round_ns: 1_000,
        last_round_ns: 1_100,
        state: DeviceState::unreported(),
    };
    assert_eq!(slowing.drift_permille(), 100, "a slowing device drifts up");

    let quickening = MeasurementEnvironment {
        last_round_ns: 900,
        ..slowing
    };
    assert_eq!(quickening.drift_permille(), -100);

    let unmeasured = MeasurementEnvironment {
        first_round_ns: 0,
        last_round_ns: 0,
        ..slowing
    };
    assert_eq!(
        unmeasured.drift_permille(),
        0,
        "a session with no counted round reports no drift"
    );

    let pessimistic = candidate(1, 1_000, 1_200);
    assert_eq!(
        pessimistic.prediction_error_permille(),
        200,
        "a model predicting slower than measured reports positive error"
    );
    let optimistic = candidate(1, 1_000, 800);
    assert_eq!(optimistic.prediction_error_permille(), -200);
    let unpriced = candidate(1, 0, 500);
    assert_eq!(
        unpriced.prediction_error_permille(),
        0,
        "an estimate of nothing has no error to report"
    );
}

/// WHY: a zero clock and an unreported clock are different facts, and a reader
/// that confuses them reads a throttled device as an idle one. Every term is
/// checked so adding one without teaching `is_unreported` about it turns this
/// red.
#[test]
fn unreported_device_state_is_distinguishable_from_every_reported_term() {
    assert!(DeviceState::unreported().is_unreported());
    let terms: [(&str, fn(&mut DeviceState)); 5] = [
        ("graphics clock", |state| {
            state.graphics_clock_khz = 1_980_000
        }),
        ("memory clock", |state| state.memory_clock_khz = 9_501_000),
        ("temperature", |state| {
            state.temperature_millicelsius = 45_000;
        }),
        ("power draw", |state| state.power_draw_milliwatts = 118_440),
        ("throttle", |state| state.throttle = ThrottleState::Clear),
    ];
    for (term, report) in terms {
        let mut state = DeviceState::unreported();
        report(&mut state);
        assert!(
            !state.is_unreported(),
            "a device reporting its {term} is not unreported"
        );
    }
    for throttle in [
        ThrottleState::Unreported,
        ThrottleState::Clear,
        ThrottleState::Throttled,
    ] {
        // Exhaustive on purpose: a new throttle state must state whether it means
        // the device is limited before anything can read it.
        let limited = match throttle {
            ThrottleState::Unreported | ThrottleState::Clear => false,
            ThrottleState::Throttled => true,
        };
        assert_eq!(
            limited,
            matches!(throttle, ThrottleState::Throttled),
            "{throttle:?} must state whether the clock is limited"
        );
    }
}

/// WHY: a protocol recorded in an artifact is read back by a later compiler.
/// Every rule it states is a rule the reader relies on, so each one is rejected
/// individually rather than trusting the writer.
#[test]
fn protocol_validation_rejects_every_rule_it_states() {
    for (why, mutate) in [
        (
            "version",
            (|protocol: &mut MeasurementProtocol| protocol.version = 0)
                as fn(&mut MeasurementProtocol),
        ),
        ("repetitions", |protocol| {
            protocol.repetitions_per_round = 0;
        }),
        ("rounds", |protocol| protocol.max_rounds = 0),
        ("inverted rounds", |protocol| {
            protocol.min_rounds = protocol.max_rounds + 1;
        }),
        ("trim share", |protocol| protocol.trim_permille = 900),
        ("zero trim", |protocol| protocol.trim_permille = 0),
        ("uncertainty target", |protocol| {
            protocol.uncertainty_target_permille = 0;
        }),
        ("equivalence band", |protocol| {
            protocol.equivalence_permille = 1_001;
        }),
    ] {
        let mut protocol = MeasurementProtocol::V1;
        mutate(&mut protocol);
        assert!(
            protocol.validate().is_err(),
            "a protocol with a broken {why} must not validate: {protocol:?}"
        );
    }
    MeasurementProtocol::V1
        .validate()
        .expect("the shipped protocol must satisfy its own rules");
}

fn estimate(estimate_ns: u64, uncertainty_ns: u64) -> SampleEstimate {
    SampleEstimate {
        estimate_ns,
        uncertainty_ns,
        kept: 4,
        trimmed: 1,
    }
}

fn candidate(identity: u8, estimate_ns: u64, predicted_ns: u64) -> CandidateMeasurement {
    CandidateMeasurement {
        identity: Digest([identity; 32]),
        analytic_rank: 0,
        predicted_ns,
        samples: vec![estimate_ns; 5],
        estimate: estimate(estimate_ns, 0),
    }
}

/// Calibrated fact-set version every comparable record in this suite shares.
const CALIBRATION: u16 = 7;

/// One record whose candidates are `(identity, estimate_ns)` pairs, ranked
/// against fact set [`CALIBRATION`].
fn record(
    protocol: MeasurementProtocol,
    candidates: &[(u8, u64)],
    winner: u32,
) -> MeasurementRecord {
    record_at(protocol, CALIBRATION, candidates, winner)
}

/// The same record ranked against fact-set version `facts_calibration_version`.
fn record_at(
    protocol: MeasurementProtocol,
    facts_calibration_version: u16,
    candidates: &[(u8, u64)],
    winner: u32,
) -> MeasurementRecord {
    MeasurementRecord {
        protocol,
        environment: MeasurementEnvironment {
            warmup_launches: protocol.warmup_launches,
            facts_calibration_version,
            first_round_ns: 1_000,
            last_round_ns: 1_000,
            state: DeviceState::unreported(),
        },
        rounds: 1,
        candidates: candidates
            .iter()
            .map(|(identity, estimate_ns)| candidate(*identity, *estimate_ns, *estimate_ns))
            .collect(),
        winner,
    }
}

/// Name every verdict. Exhaustive on purpose: a new verdict has to state what it
/// means to a caller before this suite compiles again.
fn named(verdict: ReplacementVerdict) -> &'static str {
    match verdict {
        ReplacementVerdict::Replaces => "replaces",
        ReplacementVerdict::Equivalent => "equivalent",
        ReplacementVerdict::Incomparable => "incomparable",
    }
}
