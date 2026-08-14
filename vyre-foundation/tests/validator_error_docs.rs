//! Coverage contract for the canonical validation-code catalog.

use std::collections::BTreeSet;
use std::fs;

use vyre_foundation::validate::ValidationCode;
use vyre_test_support::monorepo::vyre_workspace_root;

#[test]
fn every_registered_validation_code_is_cataloged() {
    let workspace_root = vyre_workspace_root();
    let markdown = fs::read_to_string(workspace_root.join("docs/error-codes.md"))
        .expect("canonical error-code catalog must be readable");
    let documented = markdown
        .lines()
        .filter_map(|line| {
            let code = line.strip_prefix("| `V")?.split('`').next()?;
            (code.len() == 3 && code.bytes().all(|byte| byte.is_ascii_digit()))
                .then(|| format!("V{code}"))
        })
        .collect::<BTreeSet<_>>();
    let registered = ValidationCode::registered()
        .map(|(code, _)| code.to_string())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        documented, registered,
        "Fix: docs/error-codes.md must contain exactly one row for every live validation rule."
    );
}
