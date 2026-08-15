//! Workspace-level platform documentation boundary contract.

use std::fs;
use std::process::Command;

#[test]
fn platform_crate_docs_and_comments_do_not_name_consumers() {
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
    let script = workspace.join("scripts/check_platform_consumer_docs.sh");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .output()
        .expect("platform consumer-doc boundary script should execute");

    assert!(
        output.status.success(),
        "platform consumer-doc boundary failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).as_ref(),
        "",
        "platform consumer-doc boundary must be quiet on success"
    );
}

/// No Rust source file may defer its contract to a document under the
/// repository's documentation directory.
///
/// A comment that says the rule lives elsewhere is not a statement of the
/// rule. It costs a reader a second file to learn what the code promises, and
/// it outlives the file it names, because nothing links the two. That is not
/// hypothetical: the book that held those documents was deleted, and every
/// comment pointing into it became a pointer to nothing on the same commit,
/// with no gate red to show for it. Several of them cited the same document
/// twice in one sentence, which is what a pointer looks like once nobody
/// reads the far end.
///
/// SCOPE IS A RULE, NOT A FILE LIST. Every Rust file tracked by git is
/// scanned, and a hit is a comment line naming a markdown path in the
/// documentation directory. Comments only, deliberately: a path in code is an
/// artifact the program reads or writes, and a generator naming its own
/// output owns that output rather than deferring to it. A path nested under
/// another directory is not a hit either, because the rule is about the
/// published documentation tree and not about the word.
///
/// Breaks if it regresses: a contract moves back into a document nobody
/// ships, and deleting that document silently deletes the contract.
#[test]
fn no_source_file_defers_its_contract_to_a_document() {
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
    let listing = Command::new("git")
        .args(["ls-files", "-z", "*.rs"])
        .current_dir(&workspace)
        .output()
        .expect("git ls-files should run in the workspace");
    assert!(
        listing.status.success(),
        "git ls-files failed:\n{}",
        String::from_utf8_lossy(&listing.stderr)
    );

    let mut deferrals = Vec::new();
    for relative in String::from_utf8_lossy(&listing.stdout).split('\0') {
        if relative.is_empty() {
            continue;
        }
        let Ok(source) = fs::read_to_string(workspace.join(relative)) else {
            continue;
        };
        for (index, line) in source.lines().enumerate() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("//") && !trimmed.starts_with('*') {
                continue;
            }
            for document in documents_named_in(line) {
                deferrals.push(format!("{relative}:{}: {document}", index + 1));
            }
        }
    }

    assert!(
        deferrals.is_empty(),
        "{} comment(s) defer a contract to a document instead of stating it:\n{}\n\
         Fix: state the rule in the source that has to follow it, then delete the pointer.",
        deferrals.len(),
        deferrals.join("\n")
    );
}

/// Every documentation-directory markdown path one comment line names.
///
/// A match must start the path: a directory of the same name nested under
/// another one, such as the release evidence tree, is a different tree and
/// carries no contract.
fn documents_named_in(line: &str) -> Vec<&str> {
    const DIRECTORY: &str = "docs/";
    const SUFFIX: &str = ".md";

    let path_character =
        |character: char| character.is_ascii_alphanumeric() || matches!(character, '.' | '/' | '-' | '_');
    let bytes = line.as_bytes();
    let mut found = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = line[cursor..].find(DIRECTORY) {
        let start = cursor + offset;
        cursor = start + DIRECTORY.len();
        if start > 0 && path_character(char::from(bytes[start - 1])) {
            continue;
        }
        let end = line[start..]
            .find(|character| !path_character(character))
            .map_or(line.len(), |length| start + length);
        let candidate = &line[start..end];
        if candidate.ends_with(SUFFIX) {
            found.push(candidate);
        }
    }
    found
}
