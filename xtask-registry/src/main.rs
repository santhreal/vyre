//! Registry-linked xtask subcommands.
//!
//! `xtask` builds and runs this binary for the subcommands it assigns here. It
//! is not meant to be invoked directly, but it accepts the same argument vector
//! so that `cargo run -p xtask-registry -- <subcommand>` works.

fn main() {
    xtask::delegate::run_delegated_main(
        "xtask-registry",
        "`xtask` assigns these subcommands here because each one reads the live operation registry.",
        xtask_registry::IMPLEMENTED,
    );
}
