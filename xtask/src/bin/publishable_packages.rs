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

use xtask::cli::NoArguments;
use xtask::gates::public_api::roster;
use xtask::gates::scan::Tree;

const CLI: NoArguments<'_> = NoArguments {
    binary: "publishable_packages",
    summary: "Print publishable workspace packages as `directory:package` rows.",
    success: "the roster was printed",
    failure: "the workspace could not be read, or publishes nothing",
};

fn main() {
    CLI.accept();
    let root = xtask::checkout::checkout_root();
    let rows = Tree::open(&root)
        .and_then(|tree| roster(&tree))
        .unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        });
    for row in rows {
        println!("{}:{}", row.directory, row.package);
    }
}
