//! xtask subcommands that read benchmark and release evidence provenance.
//!
//! A recorded measurement only describes the tree it was taken on. Deciding
//! whether it still does means fingerprinting the source and reading the host
//! devices, which is what `vyre-bench` owns and why this crate links it. The
//! release checks that only read manifests stay in `xtask`.


pub mod bench;
pub mod release;

/// Every subcommand this crate implements, keyed by the name typed on the
/// command line. `xtask` routes to this crate by name and this table resolves
/// it, so the two lists have to agree; the test below is what enforces that.
pub const IMPLEMENTED: &[(&str, fn(&[String]))] = &[
    ("backend-matrix", release::backend_matrix::run),
    ("bench-crossback", bench::bench_crossback::run),
    ("bench-release", bench::bench_release::run),
    ("release-benchmarks", bench::release_benchmarks::run),
    ("release-evidence", release::release_evidence::run),
    ("vyre-release-gate", release::vyre_release_gate::run),
];

/// Run the named subcommand, or report that it is not implemented here.
///
/// `args` is the process argument vector, so the subcommand name is `args[1]`
/// and every subcommand reads its own options from the same slice `xtask` would
/// have passed it.
pub fn dispatch(name: &str, args: &[String]) -> bool {
    match IMPLEMENTED.iter().find(|(row, _)| *row == name) {
        Some((_, run)) => {
            run(args);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xtask::subcommands;

    /// WHY: `xtask` decides which crate owns a subcommand and builds only that
    /// one. A row it assigns here with no entry in `IMPLEMENTED` would fail as
    /// an unknown subcommand after paying for the build, and an entry here that
    /// `xtask` assigns elsewhere is unreachable. Both lists are derived at run
    /// time, so adding a subcommand to one and not the other turns this red.
    #[test]
    fn dispatch_matches_the_subcommands_xtask_assigns_here() {
        let mut assigned = subcommands::owned_by("xtask-evidence");
        let mut implemented: Vec<&str> = IMPLEMENTED.iter().map(|(name, _)| *name).collect();
        assigned.sort_unstable();
        implemented.sort_unstable();
        assert_eq!(assigned, implemented);
    }

    /// WHY: dispatch resolves by linear search, so a duplicated name silently
    /// shadows the second entry and the equality test above would still pass.
    #[test]
    fn every_implemented_name_appears_once() {
        let mut names: Vec<&str> = IMPLEMENTED.iter().map(|(name, _)| *name).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(before, names.len());
    }

    /// WHY: an unknown name must be reported by `xtask`, not swallowed here.
    #[test]
    fn an_unknown_name_is_not_dispatched() {
        assert!(!dispatch("dep-drift", &["xtask".to_string()]));
    }
}
