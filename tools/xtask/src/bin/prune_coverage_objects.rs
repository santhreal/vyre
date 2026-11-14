//! Delete instrumented executables whose build ID no raw profile of the coverage run records.

fn main() {
    let root = xtask::workspace_root();
    if let Err(error) = xtask::coverage_objects_prune(&root) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
