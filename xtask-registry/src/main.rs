//! Registry-linked xtask gates.
//!
//! `xtask` builds and runs this binary for the gates it assigns here. It is not
//! meant to be invoked directly, but it accepts the same argument vector so that
//! `cargo run -p xtask-registry -- <gate>` works.

fn main() {
    xtask::delegate::run_delegated_main(
        "xtask-registry",
        "`xtask` assigns these gates here because each one reads the live operation registry.",
        xtask_registry::GATES,
    );
}
