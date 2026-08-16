//! Print every publishable workspace package as a `directory:package` row.
//!
//! Run via `./cargo_full run -p xtask --bin publishable_packages`. A gate reports
//! findings, so a workflow that needs the roster as data reads it here instead.
//! `semver-checks.yml` passes the package column to `cargo-semver-checks`, which
//! takes a comma-separated package list and cannot derive one itself.
//!
//! The roster has one owner: `xtask::gates::public_api::roster`, which the
//! public-API snapshot gate is taken over. A second inventory is how a package
//! ends up covered by one and not the other.

use xtask::gates::public_api::roster;
use xtask::gates::scan::Tree;
use xtask::operator_binary::{help_requested, Usage};

const USAGE: Usage = Usage {
    name: "publishable_packages",
    summary: "Print publishable workspace packages as `directory:package` rows.",
    exit_codes: &[
        (0, "the roster was printed"),
        (1, "the workspace could not be read, or publishes nothing"),
        (2, "command-line arguments are invalid"),
    ],
};

fn main() {
    if help_requested(&USAGE) {
        return;
    }
    let root = xtask::checkout::checkout_root();
    let tree = match Tree::open(&root) {
        Ok(tree) => tree,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    match roster(&tree) {
        Ok(rows) => {
            for row in rows {
                println!("{}:{}", row.directory, row.package);
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
