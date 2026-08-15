//! Registry-linked xtask gates.
//!
//! `xtask` builds and runs this binary for the gates it assigns here. It is not
//! meant to be invoked directly, but it accepts the same argument vector so
//! that `cargo run -p xtask-registry -- <name>` works.

fn main() {
    // Sixteen subcommands in this crate are not gates yet. `xtask` cannot route
    // to them, because the parent reaches a delegated crate only through a
    // registered gate, but tests and CI still invoke this binary by name and a
    // name that resolves to nothing is a check that stopped running. They are
    // resolved here until each one is converted and moved into `GATES`.
    let args: Vec<String> = std::env::args().collect();
    if let Some(name) = args.get(1) {
        if !xtask_registry::GATES.iter().any(|gate| gate.name() == name) {
            if let Some((_, run)) = xtask_registry::IMPLEMENTED
                .iter()
                .find(|(implemented, _)| implemented == name)
            {
                run(&args);
                return;
            }
        }
    }
    xtask::delegate::run_delegated_main(
        "xtask-registry",
        "`xtask` assigns these gates here because each one reads the live operation registry.",
        xtask_registry::GATES,
    );
}
