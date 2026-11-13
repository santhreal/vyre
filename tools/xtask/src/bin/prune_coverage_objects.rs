//! Delete the instrumented executables a coverage run did not build.
//!
//! Run via `./cargo_full run -q -p xtask --bin prune_coverage_objects < messages`,
//! where `messages` is the `--message-format=json` output of every
//! `cargo llvm-cov --no-report` invocation of the run. `coverage.yml` runs it
//! between the per-package builds and `cargo llvm-cov report`, so the report
//! reads only executables of this run. The selection is defined in
//! `xtask::coverage_objects`.

use std::io::Read;

use xtask::cli::NoArguments;

const CLI: NoArguments<'_> = NoArguments {
    binary: "prune_coverage_objects",
    summary: "Delete instrumented executables that the cargo JSON messages on stdin do not list.",
    success: "every unlisted executable was deleted",
    failure: "stdin is unreadable, lists no test executable, or a deletion failed",
};

fn main() {
    CLI.accept();
    let mut messages = String::new();
    if let Err(error) = std::io::stdin().read_to_string(&mut messages) {
        eprintln!("Fix: cannot read cargo JSON messages from stdin: {error}");
        std::process::exit(1);
    }
    match xtask::coverage_objects::prune(&messages) {
        Ok(selection) => println!(
            "deleted {} instrumented object(s) no artifact of this run lists; kept {}",
            selection.stale.len(),
            selection.kept
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
