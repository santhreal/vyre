//! Is a float lowering mode the selected backend does not lower refused, at
//! every dispatch entry point, naming the mode and the backend?
//!
//! WHY: `DispatchConfig::float_lowering` selects the rounding a caller
//! requires. A backend that does not lower the requested mode emitted the mode
//! it does lower, which answers a request for one rounding per operation with
//! contracted arithmetic: a wrong result rather than a slow one, and one the
//! caller cannot see. `GridSyncSplitBackend::require_lowered_float_mode` is the
//! refusal, and the registry wrapper is where every dispatch that carries a
//! `DispatchConfig` passes through, so it is the site the config first reaches
//! compilation.
//!
//! Three closures hold this shut:
//!
//!   - the expected default answer is an exhaustive `match` with no catch-all
//!     arm, so a `FloatLoweringMode` variant added without a recorded default
//!     stops this suite compiling rather than inheriting an answer;
//!   - the refused feature name is compared against
//!     `fp_parity::blocked_contraction_feature`, the one owner of that string,
//!     so the wrapper's copy drifting from it is red;
//!   - the set of entry points exercised below is compared against the set the
//!     wrapper source guards, so a dispatch entry point added without the
//!     refusal is red.
//!
//! Which backend records which decision is a different question, asked over the
//! live registry in `vyre-registry-link/tests/float_lowering_decisions.rs`. A
//! binary that links no concrete driver, which this one is, enumerates an empty
//! registry and would answer it vacuously.

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::sync::LazyLock;

use vyre_driver::{BackendError, BackendRegistration, DispatchConfig, VyreBackend};
use vyre_foundation::fp_parity::{
    approximable_operations, blocked_contraction_feature, FloatLoweringMode,
};
use vyre_foundation::ir::{OpId, Program, UnOp};
use vyre_foundation::operation::TargetId;
use vyre_test_support::strict_float_programs::f32_multiply_add_program;

/// The backend that takes the trait default, so it lowers what the default
/// admits and nothing else.
const REFUSER_ID: &str = "float-lowering-default-fixture";

/// The backend that states it lowers every mode, so the wrapper has something
/// to forward and a blanket refusal is not mistaken for the contract.
const CAPABLE_ID: &str = "float-lowering-capable-fixture";

/// The backend that states it lowers no mode at all, so the refusal for a mode
/// that permits contraction is reachable.
const NOTHING_ID: &str = "float-lowering-nothing-fixture";

struct DefaultLowering;

impl vyre_driver::sealed::Sealed for DefaultLowering {}

impl VyreBackend for DefaultLowering {
    fn id(&self) -> &'static str {
        REFUSER_ID
    }

    fn dispatch_borrowed(
        &self,
        _program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![vec![1]])
    }
}

struct EveryModeLowering;

impl vyre_driver::sealed::Sealed for EveryModeLowering {}

impl VyreBackend for EveryModeLowering {
    fn id(&self) -> &'static str {
        CAPABLE_ID
    }

    fn honors_float_lowering(&self, mode: FloatLoweringMode) -> bool {
        match mode {
            FloatLoweringMode::Contracted | FloatLoweringMode::StrictIeee => true,
        }
    }

    fn dispatch_borrowed(
        &self,
        _program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![vec![2]])
    }
}

struct NoModeLowering;

impl vyre_driver::sealed::Sealed for NoModeLowering {}

impl VyreBackend for NoModeLowering {
    fn id(&self) -> &'static str {
        NOTHING_ID
    }

    fn honors_float_lowering(&self, _mode: FloatLoweringMode) -> bool {
        false
    }

    fn dispatch_borrowed(
        &self,
        _program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![vec![3]])
    }
}

fn no_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(HashSet::new);
    &OPS
}

fn default_lowering() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(DefaultLowering))
}

fn every_mode_lowering() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(EveryModeLowering))
}

fn no_mode_lowering() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(NoModeLowering))
}

/// The registry wrapper around one fixture backend.
///
/// `BackendRegistration::acquire` is the boundary every registry caller comes
/// through, so acquiring a locally built registration applies the same wrapper
/// a linked driver gets. The registration is not submitted to the link-time
/// inventory: this binary's registry is read by sibling suites, and a fixture
/// in it would change what they enumerate.
fn wrapped(
    id: &'static str,
    factory: fn() -> Result<Box<dyn VyreBackend>, BackendError>,
) -> Box<dyn VyreBackend> {
    BackendRegistration {
        id,
        target_id: TargetId::expect_valid(id),
        payload_format: None,
        reference_oracle: false,
        factory,
        supported_ops: no_ops,
        semantic_operations: no_ops,
        target_compiler: None,
        materializer: None,
    }
    .acquire()
    .expect("Fix: a fixture backend factory that returns Ok must acquire through the registry")
}

/// The answer `VyreBackend::honors_float_lowering` gives when a backend states
/// nothing, restated as an exhaustive match with no catch-all arm.
///
/// A variant added to `FloatLoweringMode` makes this match non-exhaustive, so
/// the suite stops compiling until the new mode's default is recorded here and
/// in the trait. Whether `FloatLoweringMode::EVERY` lists every variant is
/// `vyre_foundation::fp_parity`'s own contract, asserted through
/// `roster_index`.
const fn default_admits(mode: FloatLoweringMode) -> bool {
    match mode {
        FloatLoweringMode::Contracted => true,
        FloatLoweringMode::StrictIeee => false,
    }
}

/// A backend that states nothing answers the recorded default for every mode.
#[test]
fn the_trait_default_admits_exactly_the_recorded_mode_set() {
    for &mode in FloatLoweringMode::EVERY {
        assert_eq!(
            DefaultLowering.honors_float_lowering(mode),
            default_admits(mode),
            "Fix: `VyreBackend::honors_float_lowering` and the recorded default disagree about \
             mode `{}`. A backend that states nothing must inherit one answer, not two.",
            mode.cache_label()
        );
    }
}

/// A backend that states it lowers a mode is dispatched, not refused.
///
/// Without this the refusal contract below is satisfied by a wrapper that
/// refuses everything.
#[test]
fn a_mode_the_backend_states_it_lowers_reaches_the_backend() {
    let backend = wrapped(CAPABLE_ID, every_mode_lowering);
    for &mode in FloatLoweringMode::EVERY {
        let mut config = DispatchConfig::default();
        config.float_lowering = mode;
        assert_eq!(
            backend
                .dispatch_borrowed(&f32_multiply_add_program(4, Some(UnOp::Sin)), &[], &config)
                .unwrap_or_else(|error| panic!(
                    "Fix: mode `{}` is stated as lowered and must dispatch: {error}",
                    mode.cache_label()
                )),
            vec![vec![2]],
            "Fix: the wrapper must forward a dispatch whose mode the backend lowers."
        );
    }
}

/// The refusal names the mode and the backend, in both shapes the message has.
///
/// `require_lowered_float_mode` builds one string when the program contains
/// approximable operations and another when it does not, and the expectation
/// here is taken from `fp_parity::blocked_contraction_feature`, which is the
/// one owner of both. A wrapper that formats its own copy and drifts from that
/// owner is what this compares against, so the drift is red rather than
/// invisible.
#[test]
fn the_refusal_names_the_mode_the_backend_and_the_blocked_operations() {
    let backend = wrapped(REFUSER_ID, default_lowering);
    let with_operations = f32_multiply_add_program(4, Some(UnOp::Sin));
    let without_operations = f32_multiply_add_program(4, None);

    assert_eq!(
        approximable_operations(&with_operations),
        vec![String::from("Sin")],
        "Fix: the program built to carry an approximable operation carries none, so the refusal \
         shape it is here to exercise is never built."
    );
    assert!(
        approximable_operations(&without_operations).is_empty(),
        "Fix: the program built to carry no approximable operation carries one, so both cases \
         below exercise the same refusal shape."
    );

    for program in [&with_operations, &without_operations] {
        for &mode in FloatLoweringMode::EVERY {
            if default_admits(mode) {
                continue;
            }
            let mut config = DispatchConfig::default();
            config.float_lowering = mode;
            let expected = blocked_contraction_feature(program, mode).unwrap_or_else(|| {
                panic!(
                    "Fix: mode `{}` blocks contraction, so the one owner of the refused-feature \
                     name must produce one",
                    mode.cache_label()
                )
            });

            match backend.dispatch_borrowed(program, &[], &config) {
                Err(BackendError::UnsupportedFeature { name, backend: id }) => {
                    assert_eq!(
                        name, expected,
                        "Fix: the registry wrapper's refused-feature name and \
                         `fp_parity::blocked_contraction_feature` disagree. Build the name \
                         through that owner so every refusal spells the mode and the blocked \
                         operations one way."
                    );
                    assert!(
                        name.contains(mode.cache_label()),
                        "Fix: the refusal must name the mode; got `{name}`"
                    );
                    assert_eq!(
                        id, REFUSER_ID,
                        "Fix: the refusal must name the backend that does not lower the mode."
                    );
                }
                other => panic!(
                    "Fix: a backend that does not lower `{}` must refuse with \
                     UnsupportedFeature, got {other:?}",
                    mode.cache_label()
                ),
            }
        }
    }

    assert_ne!(
        blocked_contraction_feature(&with_operations, FloatLoweringMode::StrictIeee),
        blocked_contraction_feature(&without_operations, FloatLoweringMode::StrictIeee),
        "Fix: the refusal reads the same with and without approximable operations, so the \
         operation half of the message is never proven."
    );
}

/// A refused mode that permits contraction is named on its own.
///
/// `fp_parity::blocked_contraction_feature` answers nothing for a mode that
/// permits contraction, because such a mode blocks no operation. A backend may
/// still state it does not lower one, and the refusal then names the mode and
/// the backend and no operation set, even where the program carries operations
/// a stricter mode would deny.
#[test]
fn a_refused_mode_that_permits_contraction_is_named_without_operations() {
    let backend = wrapped(NOTHING_ID, no_mode_lowering);
    let program = f32_multiply_add_program(4, Some(UnOp::Sin));
    let permitting: Vec<FloatLoweringMode> = FloatLoweringMode::EVERY
        .iter()
        .copied()
        .filter(|mode| blocked_contraction_feature(&program, *mode).is_none())
        .collect();

    assert!(
        !permitting.is_empty(),
        "Fix: every mode blocks contraction, so the refusal shape this case exercises is \
         unreachable and the fallback it covers is dead code to delete."
    );
    assert!(
        !approximable_operations(&program).is_empty(),
        "Fix: the program carries no approximable operation, so a refusal naming one cannot be \
         told apart from a refusal naming none."
    );

    for mode in permitting {
        let mut config = DispatchConfig::default();
        config.float_lowering = mode;

        match backend.dispatch_borrowed(&program, &[], &config) {
            Err(BackendError::UnsupportedFeature { name, backend: id }) => {
                assert_eq!(
                    name,
                    format!("float lowering mode `{}`", mode.cache_label()),
                    "Fix: a mode that blocks no operation must be refused by name alone."
                );
                assert_eq!(
                    id, NOTHING_ID,
                    "Fix: the refusal must name the backend that does not lower the mode."
                );
            }
            other => panic!(
                "Fix: a backend that states it lowers no mode must refuse `{}` with \
                 UnsupportedFeature, got {other:?}",
                mode.cache_label()
            ),
        }
    }
}

/// One dispatch entry point on the wrapper, as a call that discards its Ok.
struct EntryPoint {
    name: &'static str,
    call: fn(&dyn VyreBackend, &Program, &DispatchConfig) -> Result<(), BackendError>,
}

const ENTRY_POINTS: &[EntryPoint] = &[
    EntryPoint {
        name: "dispatch",
        call: |backend, program, config| backend.dispatch(program, &[], config).map(drop),
    },
    EntryPoint {
        name: "dispatch_borrowed",
        call: |backend, program, config| backend.dispatch_borrowed(program, &[], config).map(drop),
    },
    EntryPoint {
        name: "dispatch_borrowed_timed",
        call: |backend, program, config| {
            backend
                .dispatch_borrowed_timed(program, &[], config)
                .map(drop)
        },
    },
    EntryPoint {
        name: "dispatch_borrowed_into",
        call: |backend, program, config| {
            let mut outputs = Vec::new();
            backend.dispatch_borrowed_into(program, &[], config, &mut outputs)
        },
    },
    EntryPoint {
        name: "dispatch_resident_timed",
        call: |backend, program, config| {
            backend
                .dispatch_resident_timed(program, &[], config)
                .map(drop)
        },
    },
    EntryPoint {
        name: "dispatch_async",
        call: |backend, program, config| backend.dispatch_async(program, &[], config).map(drop),
    },
    EntryPoint {
        name: "dispatch_borrowed_async",
        call: |backend, program, config| {
            backend
                .dispatch_borrowed_async(program, &[], config)
                .map(drop)
        },
    },
    EntryPoint {
        name: "dispatch_with_device_buffers",
        call: |backend, program, config| {
            backend.dispatch_with_device_buffers(program, &[], &mut [], config)
        },
    },
];

/// Every method in the registry wrapper that guards its dispatch with the
/// float-lowering refusal.
///
/// Derived from the wrapper's source rather than listed, so a dispatch entry
/// point added to the wrapper is judged the run after it is written.
fn guarded_entry_points_in_wrapper_source() -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"))
        .join("src/backend/registry/grid_sync_split.rs");
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let header = "impl VyreBackend for GridSyncSplitBackend {";
    let start = source
        .find(header)
        .unwrap_or_else(|| panic!("cannot find `{header}` in {}", path.display()))
        + header.len();
    let body = &source[start..];
    let end = body
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`{header}` block is unterminated in {}", path.display()));

    let mut guarded = BTreeSet::new();
    let mut current: Option<String> = None;
    for line in body[..end].lines() {
        if let Some(rest) = line.strip_prefix("    fn ") {
            current = Some(
                rest.chars()
                    .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
                    .collect(),
            );
        } else if line.contains("self.require_lowered_float_mode(") {
            if let Some(name) = current.take() {
                guarded.insert(name);
            }
        }
    }
    guarded
}

/// Every guarded entry point refuses, and every guarded entry point is here.
#[test]
fn every_wrapper_dispatch_entry_point_refuses_a_mode_the_backend_does_not_lower() {
    let exercised: BTreeSet<String> = ENTRY_POINTS
        .iter()
        .map(|entry| entry.name.to_string())
        .collect();
    let guarded = guarded_entry_points_in_wrapper_source();
    assert!(
        !guarded.is_empty(),
        "Fix: no method in the registry wrapper guards dispatch with the float-lowering refusal, \
         so this suite would pass over an empty set."
    );
    assert_eq!(
        guarded, exercised,
        "Fix: the registry wrapper guards a different set of dispatch entry points than this \
         suite calls. Call the new entry point here, or remove the case for the one that is gone; \
         an entry point nobody calls is a refusal nobody proved."
    );

    let backend = wrapped(REFUSER_ID, default_lowering);
    let program = f32_multiply_add_program(4, Some(UnOp::Sin));
    for &mode in FloatLoweringMode::EVERY {
        if default_admits(mode) {
            continue;
        }
        let mut config = DispatchConfig::default();
        config.float_lowering = mode;
        let expected = blocked_contraction_feature(&program, mode)
            .expect("Fix: a mode that blocks contraction names a blocked feature");
        for entry in ENTRY_POINTS {
            match (entry.call)(backend.as_ref(), &program, &config) {
                Err(BackendError::UnsupportedFeature { name, backend: id }) => {
                    assert_eq!(
                        name,
                        expected,
                        "Fix: `{}` refuses mode `{}` with a different feature name than its \
                         sibling entry points.",
                        entry.name,
                        mode.cache_label()
                    );
                    assert_eq!(
                        id, REFUSER_ID,
                        "Fix: `{}` must name the backend that does not lower the mode.",
                        entry.name
                    );
                }
                other => panic!(
                    "Fix: `{}` accepted or mis-refused mode `{}`, which lets a caller reach \
                     compilation with a mode the backend does not lower: {other:?}",
                    entry.name,
                    mode.cache_label()
                ),
            }
        }
    }
}
