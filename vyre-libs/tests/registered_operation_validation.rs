//! WHY: no test validated the program a registered operation builds. Validation
//! is the first thing every dispatch path runs, so an operation whose neutral
//! program the validator refuses is a registration the crate can never execute,
//! and the only signal was a failure inside a consumer's own program. One
//! registered resolver carried a V139 offset for exactly that reason.
//!
//! Two owners answer "what does this program need": `scan_capabilities`, which
//! the registry derives the requirement from, and the validator, which refuses a
//! program whose capabilities the run does not grant. Each program is validated
//! against exactly the capabilities its own registration claims, so a
//! capability the scanner misses is refused rather than granted for free, and a
//! rule that is not capability-sensitive stays a hard failure.
//!
//! Closes: every operation the library catalog registers with a neutral builder,
//! enumerated from the registry at run time, so a new registration is covered by
//! being registered.
//!
//! Does not catch: a program a caller builds with arguments the registration does
//! not supply. The registry holds one neutral builder per operation, and that is
//! the program every gate reads.

use vyre_foundation::program_caps::RequiredCapabilities;
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};
use vyre_libs::operation_catalog::all_entries;

/// The backend surface a registration's declared requirements ask for.
///
/// Only the bits both sides name are carried across. A requirement the registry
/// does not claim stays absent, so the validator refuses the program that needs
/// it instead of the test granting a capability nobody recorded.
fn granted(required: Option<RequiredCapabilities>) -> BackendCapabilities {
    let Some(required) = required else {
        return BackendCapabilities::default();
    };
    BackendCapabilities {
        supports_subgroup_ops: required.subgroup_ops,
        supports_indirect_dispatch: required.indirect_dispatch,
        supports_distributed_collectives: required.distributed_collectives,
        ..BackendCapabilities::default()
    }
}

#[test]
fn every_registered_operation_builds_a_program_the_validator_accepts() {
    let mut validated = 0usize;
    let mut refused: Vec<String> = Vec::new();

    for operation in all_entries() {
        let Some(program) = operation.program() else {
            continue;
        };
        validated += 1;
        let options = ValidationOptions::default()
            .with_backend_capabilities(granted(operation.required_capabilities()));
        let report = validate_with_options(&program, options);
        if report.errors.is_empty() {
            continue;
        }
        let reported = report
            .errors
            .iter()
            .map(|error| format!("{} {}", error.code().as_str(), error.message()))
            .collect::<Vec<_>>()
            .join("; ");
        refused.push(format!("{}: {reported}", operation.id));
    }

    assert!(
        validated > 0,
        "Fix: the library catalog registered no operation with a neutral builder, so this test \
         validated nothing. Restore the registrations before continuing."
    );
    assert!(
        refused.is_empty(),
        "Fix: {} registered operation(s) build a program the validator refuses under the \
         capabilities their own registration declares, so no dispatch path can run them:\n{}",
        refused.len(),
        refused.join("\n")
    );
}
