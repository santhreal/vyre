//! Contract for the default `VyreBackend` async dispatch adapters.
//!
//! Guarantees:
//! - `dispatch_async` performs the synchronous dispatch and hands back a ready
//!   handle, so a caller never spin-waits on the default adapter.
//! - A dispatch failure surfaces from `dispatch_async` itself, never deferred
//!   into the handle where a caller could drop it unread.
//! - Repeated calls produce independent handles that each carry their own
//!   outputs.
//! - `PendingDispatch` is object-safe and consumable through both the plain and
//!   the timed await.
//! - `dispatch_borrowed` and `dispatch_resident_async` keep the same semantics
//!   when a backend implements neither.

use std::sync::atomic::{AtomicUsize, Ordering};

use vyre_driver::{
    BackendError, DispatchConfig, PendingDispatch, Resource, TimedDispatchResult, VyreBackend,
};
use vyre_foundation::ir::Program;

/// Echoes its inputs and counts how many times `dispatch` ran.
struct CountingBackend {
    dispatch_calls: AtomicUsize,
}

impl CountingBackend {
    fn new() -> Self {
        Self {
            dispatch_calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.dispatch_calls.load(Ordering::Relaxed)
    }
}

impl vyre_driver::backend::private::Sealed for CountingBackend {}

impl VyreBackend for CountingBackend {
    fn id(&self) -> &'static str {
        "counting"
    }

    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.dispatch_calls.fetch_add(1, Ordering::Relaxed);
        Ok(inputs.to_vec())
    }
}

/// Fails every ordinary dispatch so the error path is observable.
struct FailingBackend;

impl vyre_driver::backend::private::Sealed for FailingBackend {}

impl VyreBackend for FailingBackend {
    fn id(&self) -> &'static str {
        "failing"
    }

    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Err(BackendError::new(
            "injected failure. Fix: this is a test fixture.",
        ))
    }
}

/// Implements only the resident dispatch arm, optionally failing it.
struct ResidentBackend {
    dispatch_calls: AtomicUsize,
    fail: bool,
}

impl ResidentBackend {
    fn new(fail: bool) -> Self {
        Self {
            dispatch_calls: AtomicUsize::new(0),
            fail,
        }
    }

    fn calls(&self) -> usize {
        self.dispatch_calls.load(Ordering::Relaxed)
    }
}

impl vyre_driver::backend::private::Sealed for ResidentBackend {}

impl VyreBackend for ResidentBackend {
    fn id(&self) -> &'static str {
        "resident-counting"
    }

    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Err(BackendError::new(
            "ordinary dispatch is outside this resident test fixture. Fix: call dispatch_resident_async.",
        ))
    }

    fn dispatch_resident_timed(
        &self,
        _program: &Program,
        _resources: &[Resource],
        _config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        self.dispatch_calls.fetch_add(1, Ordering::Relaxed);
        if self.fail {
            return Err(BackendError::new(
                "injected resident failure. Fix: this is a test fixture.",
            ));
        }
        Ok(TimedDispatchResult {
            outputs: vec![vec![10, 11, 12]],
            wall_ns: 7,
            device_ns: Some(5),
            enqueue_ns: Some(1),
            wait_ns: Some(1),
        })
    }
}

#[test]
fn dispatch_async_propagates_error_immediately() {
    let backend = FailingBackend;
    let program = Program::default();
    let result = backend.dispatch_async(&program, &[vec![1, 2, 3]], &DispatchConfig::default());
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!(
            "Fix: dispatch_async must return Err immediately when the synchronous dispatch fails, \
             never defer the failure into the handle."
        ),
    };
    assert!(
        error.to_string().contains("injected failure"),
        "Fix: the async error must surface the underlying dispatch message: {error}"
    );
}

#[test]
fn default_dispatch_async_never_blocks_and_returns_ready_handle() {
    let backend = CountingBackend::new();
    let program = Program::default();
    let pending = backend
        .dispatch_async(&program, &[vec![1, 2, 3]], &DispatchConfig::default())
        .expect("Fix: the default dispatch_async must succeed when dispatch succeeds.");
    assert!(
        pending.is_ready(),
        "Fix: the default dispatch_async must return a ready handle so callers never spin-wait."
    );
    let outputs = pending
        .await_result()
        .expect("Fix: a ready handle's result must be retrievable.");
    assert_eq!(outputs, vec![vec![1, 2, 3]]);
    assert_eq!(
        backend.calls(),
        1,
        "Fix: the default dispatch_async must call dispatch exactly once."
    );
}

#[test]
fn dispatch_async_produces_independent_handles() {
    let backend = CountingBackend::new();
    let program = Program::default();
    let config = DispatchConfig::default();
    let handles = [vec![1u8], vec![2], vec![3]].map(|input| {
        backend
            .dispatch_async(&program, std::slice::from_ref(&input), &config)
            .expect("Fix: every dispatch_async call must produce its own handle.")
    });

    for (index, pending) in handles.into_iter().enumerate() {
        let expected = vec![vec![u8::try_from(index + 1).expect("index fits a byte")]];
        assert_eq!(
            pending
                .await_result()
                .expect("Fix: each independent handle must carry its own result."),
            expected,
            "Fix: handle {index} must return the inputs its own dispatch received."
        );
    }
    assert_eq!(
        backend.calls(),
        3,
        "Fix: three dispatch_async calls must reach dispatch three times."
    );
}

#[test]
fn pending_dispatch_trait_is_object_safe_through_both_awaits() {
    let backend = CountingBackend::new();
    let program = Program::default();
    let config = DispatchConfig::default();

    let untimed: Box<dyn PendingDispatch> = backend
        .dispatch_async(&program, &[vec![4, 5, 6]], &config)
        .expect("Fix: dispatch_async must produce an object-safe PendingDispatch.");
    assert!(untimed.is_ready());
    assert_eq!(
        untimed
            .await_result()
            .expect("Fix: object-safe await_result must succeed."),
        vec![vec![4, 5, 6]]
    );

    let timed_handle: Box<dyn PendingDispatch> = backend
        .dispatch_async(&program, &[vec![7, 8]], &config)
        .expect("Fix: dispatch_async must produce an object-safe PendingDispatch.");
    assert!(timed_handle.is_ready());
    let timed = timed_handle
        .await_timed_result()
        .expect("Fix: object-safe await_timed_result must succeed.");
    assert_eq!(timed.outputs, vec![vec![7, 8]]);
    assert_eq!(
        timed.device_ns, None,
        "Fix: the default async adapter must not invent device timing it never measured."
    );
}

#[test]
fn dispatch_borrowed_async_preserves_semantics() {
    let backend = CountingBackend::new();
    let program = Program::default();
    let inputs: Vec<Vec<u8>> = vec![vec![7, 8, 9]];
    let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    let outputs = backend
        .dispatch_borrowed(&program, &borrowed, &DispatchConfig::default())
        .expect("Fix: dispatch_borrowed must forward to dispatch by default.");
    assert_eq!(outputs, vec![vec![7, 8, 9]]);
}

#[test]
fn default_resident_async_returns_ready_output_without_losing_dispatch_errors() {
    let program = Program::default();
    let backend = ResidentBackend::new(false);
    let pending = backend
        .dispatch_resident_async(&program, &[], &DispatchConfig::default())
        .expect("Fix: default resident async dispatch must preserve a successful resident dispatch.");
    assert!(
        pending.is_ready(),
        "Fix: a backend using the synchronous fallback must return a ready handle."
    );
    assert_eq!(
        pending
            .await_result()
            .expect("Fix: a ready resident result must remain available."),
        vec![vec![10, 11, 12]]
    );
    assert_eq!(backend.calls(), 1);

    let failing = ResidentBackend::new(true);
    let error = match failing.dispatch_resident_async(&program, &[], &DispatchConfig::default()) {
        Ok(_) => panic!("Fix: a resident submission failure must not produce a pending handle."),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("injected resident failure"),
        "Fix: the original resident submission error must remain actionable: {error}"
    );
    assert_eq!(failing.calls(), 1);
}
