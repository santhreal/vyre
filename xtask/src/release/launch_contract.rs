//! The external actions a launch needs that no gate can perform.
//!
//! Publishing, pushing and verifying the public repository happen outside this
//! repository, so the release gate names them rather than doing them.

use crate::release::repo_boundary;

pub(crate) const PUBLISH_ACTION: &str = "cargo_full publish approved crates in dependency order";
pub(crate) const GIT_PUSH_ACTION: &str = "git push release branch and tags";

/// The actions a launch needs that happen outside this repository.
pub fn required_external_actions() -> [&'static str; 3] {
    [
        PUBLISH_ACTION,
        repo_boundary::verify_public_repo_action(),
        GIT_PUSH_ACTION,
    ]
}
