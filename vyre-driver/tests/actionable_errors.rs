//! Failure-oriented tests for actionable error contracts.
//!
//! Guarantees:
//! - Every explicitly actionable `BackendError` variant's Display contains "Fix: "
//! - `BackendError::new` preserves complete owner-authored messages verbatim
//! - All stable error codes are unique

use std::collections::HashSet;

use vyre_driver::backend::{BackendError, ErrorCode};

#[test]
fn all_backend_error_variants_contain_fix() {
    let variants = vec![
        BackendError::DeviceOutOfMemory {
            requested: 1024,
            available: 512,
        },
        BackendError::DeviceLost {
            backend: "test".into(),
            device: "gpu0".into(),
            generation: 7,
            message: "device callback reported loss".into(),
        },
        BackendError::UnsupportedFeature {
            name: "subgroup_ops".into(),
            backend: "test".into(),
        },
        BackendError::PoisonedLock {
            lock_error: "mutex poisoned".into(),
        },
        BackendError::KernelCompileFailed {
            backend: "test".into(),
            compiler_message: "type mismatch".into(),
        },
        BackendError::DispatchFailed {
            code: Some(1),
            message: "queue full".into(),
        },
        BackendError::InvalidProgram {
            fix: "Fix: supply a valid program.".into(),
        },
        BackendError::Other("something went wrong. Fix: check logs.".into()),
    ];

    for err in &variants {
        let msg = err.to_string();
        assert!(
            msg.contains("Fix:"),
            "Fix: every BackendError variant must contain 'Fix:' in its message; got: {msg}"
        );
    }
}

#[test]
fn backend_error_new_preserves_messages_without_parsing() {
    let err = BackendError::new("something went wrong");
    assert_eq!(
        err.to_string(),
        "something went wrong",
        "BackendError::new must not inspect or rewrite rendered diagnostic prose"
    );
}

#[test]
fn backend_error_new_preserves_fix_when_present() {
    let err = BackendError::new("something went wrong. Fix: check the adapter limits.");
    let msg = err.to_string();
    assert_eq!(
        msg, "something went wrong. Fix: check the adapter limits.",
        "Fix: BackendError::new must preserve an existing Fix section verbatim"
    );
}

#[test]
fn unsupported_extension_is_actionable() {
    let err = BackendError::unsupported_extension("test-backend", "target-text", "my_ext");
    let msg = err.to_string();
    assert!(
        msg.contains("Fix:"),
        "Fix: unsupported_extension must produce actionable error; got: {msg}"
    );
    assert!(msg.contains("opaque IR extension"));
}

#[test]
fn poisoned_lock_is_actionable() {
    let lock = std::sync::RwLock::new(42);
    let guard = lock.read().unwrap();
    let err = BackendError::poisoned_lock(std::sync::PoisonError::new(guard));
    let msg = err.to_string();
    assert!(
        msg.contains("Fix:"),
        "Fix: poisoned_lock must produce actionable error; got: {msg}"
    );
    assert!(msg.contains("poison"));
}

#[test]
fn error_code_stable_ids_are_unique() {
    let codes = vec![
        ErrorCode::DeviceOutOfMemory,
        ErrorCode::DeviceLost,
        ErrorCode::UnsupportedFeature,
        ErrorCode::PoisonedLock,
        ErrorCode::KernelCompileFailed,
        ErrorCode::DispatchFailed,
        ErrorCode::InvalidProgram,
        ErrorCode::Unknown,
    ];
    let mut ids = HashSet::new();
    for code in codes {
        let id = code.stable_id();
        assert!(
            ids.insert(id),
            "Fix: ErrorCode stable IDs must be unique; duplicate {id}"
        );
    }
}

#[test]
fn backend_error_code_roundtrip() {
    let pairs = vec![
        (
            BackendError::DeviceOutOfMemory {
                requested: 1,
                available: 0,
            },
            ErrorCode::DeviceOutOfMemory,
        ),
        (
            BackendError::DeviceLost {
                backend: "test".into(),
                device: "gpu0".into(),
                generation: 7,
                message: "lost".into(),
            },
            ErrorCode::DeviceLost,
        ),
        (
            BackendError::UnsupportedFeature {
                name: "".into(),
                backend: "".into(),
            },
            ErrorCode::UnsupportedFeature,
        ),
        (
            BackendError::PoisonedLock {
                lock_error: "".into(),
            },
            ErrorCode::PoisonedLock,
        ),
        (
            BackendError::KernelCompileFailed {
                backend: "".into(),
                compiler_message: "".into(),
            },
            ErrorCode::KernelCompileFailed,
        ),
        (
            BackendError::DispatchFailed {
                code: None,
                message: "".into(),
            },
            ErrorCode::DispatchFailed,
        ),
        (
            BackendError::InvalidProgram { fix: "".into() },
            ErrorCode::InvalidProgram,
        ),
        (BackendError::Other("".into()), ErrorCode::Unknown),
    ];
    for (err, expected) in pairs {
        assert_eq!(
            err.code(),
            expected,
            "Fix: BackendError::code must roundtrip for {expected:?}"
        );
    }
}
