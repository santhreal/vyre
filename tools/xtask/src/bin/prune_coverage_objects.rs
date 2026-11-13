//! Delete the instrumented executables that wrote no profile in a coverage run.
//!
//! Run via `./cargo_full run -q -p xtask --bin prune_coverage_objects` after
//! every `cargo llvm-cov --no-report` invocation of the run and before
//! `cargo llvm-cov report`, so the report reads only executables of this run.
//! `coverage.yml` runs it that way. The selection is defined in
//! `xtask::coverage_objects`.

use xtask::cli::NoArguments;
use xtask::coverage_objects::{delete, llvm_profdata, profiled_build_ids, select, CoverageScope};

const CLI: NoArguments<'_> = NoArguments {
    binary: "prune_coverage_objects",
    summary: "Delete instrumented executables whose build ID no raw profile of the coverage run records.",
    success: "every executable no raw profile records was deleted",
    failure: "the coverage directory holds no raw profile, a profile or executable is unreadable, or a deletion failed",
};

fn main() {
    CLI.accept();
    let root = xtask::checkout::checkout_root();
    let pruned = CoverageScope::of_workspace(&root).and_then(|scope| {
        let ids = profiled_build_ids(&scope.directory, &llvm_profdata()?)?;
        let selection = select(&scope, &ids)?;
        delete(&selection)?;
        Ok(selection)
    });
    match pruned {
        Ok(selection) => println!(
            "deleted {} instrumented executable(s) no raw profile of this run records; kept {}",
            selection.stale.len(),
            selection.kept
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
