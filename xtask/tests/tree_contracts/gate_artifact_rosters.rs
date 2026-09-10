//! The declared artifact lists gate descriptors hold.
//!
//! A `GateDescriptor` names the artifacts its gate generates in a `&'static`
//! list, and the descriptor accounting charges a run against exactly that list.
//! The list is typed by hand while the set it describes is derived from the
//! workspace, so the two drift apart in silence: `PUBLIC_API_ARTIFACTS` carried
//! snapshots for four packages that had become `publish = false`, and nothing
//! reported it because a name in the list that matches no package is not read
//! by any gate.
//!
//! Both halves of each list are derived from the tree here, so adding a member
//! or changing a `publish` flag turns this red until the list records it.

use std::collections::BTreeSet;

use xtask::gate_metadata::{PUBLIC_API_ARTIFACTS, TESTING_GUIDE_ARTIFACTS};
use xtask::gates::public_api::roster;
use xtask::gates::scan::Tree;

use super::workspace_sources::workspace_root;

/// Every publishable package has a declared snapshot artifact, and nothing else does.
#[test]
fn the_public_api_artifact_list_is_the_publishable_roster() {
    let tree = Tree::open(&workspace_root()).expect("the workspace manifest is readable");
    let publishable: BTreeSet<String> = roster(&tree)
        .expect("the publishable roster resolves")
        .into_iter()
        .map(|row| format!("docs/public-api/{}.txt", row.package))
        .collect();
    assert!(
        !publishable.is_empty(),
        "the roster is empty, so this contract would hold against any list"
    );
    let declared: BTreeSet<String> = PUBLIC_API_ARTIFACTS
        .iter()
        .map(|path| (*path).to_string())
        .collect();

    let missing: Vec<&String> = publishable.difference(&declared).collect();
    assert!(
        missing.is_empty(),
        "publishable packages with no declared snapshot artifact: {missing:?}. Add them to \
         PUBLIC_API_ARTIFACTS, because the descriptor accounting does not charge the gate for an \
         artifact it never declared."
    );
    let stale: Vec<&String> = declared.difference(&publishable).collect();
    assert!(
        stale.is_empty(),
        "declared snapshot artifacts for packages that are not publishable: {stale:?}. Delete \
         them, because a scoped run is charged against a name no gate writes."
    );
}

/// Every workspace member has a declared testing guide artifact, and nothing else does.
#[test]
fn the_testing_guide_artifact_list_is_the_workspace_roster() {
    let tree = Tree::open(&workspace_root()).expect("the workspace manifest is readable");
    let members: BTreeSet<String> = tree
        .members()
        .expect("the workspace member list resolves")
        .iter()
        .map(|member| {
            let name = member.rsplit('/').next().unwrap_or(member);
            format!("docs/testing/{name}.md")
        })
        .collect();
    assert!(
        !members.is_empty(),
        "the member set is empty, so this contract would hold against any list"
    );
    let declared: BTreeSet<String> = TESTING_GUIDE_ARTIFACTS
        .iter()
        .map(|path| (*path).to_string())
        .collect();

    let missing: Vec<&String> = members.difference(&declared).collect();
    assert!(
        missing.is_empty(),
        "workspace members with no declared testing guide artifact: {missing:?}"
    );
    let stale: Vec<&String> = declared.difference(&members).collect();
    assert!(
        stale.is_empty(),
        "declared testing guide artifacts for directories that are not workspace members: \
         {stale:?}"
    );
}
