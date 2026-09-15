//! Evidence-provenance xtask subcommands.
//!
//! `xtask` builds and runs this binary for the subcommands it assigns here. It
//! is not meant to be invoked directly, but it accepts the same argument vector
//! so that `cargo run -p xtask-evidence -- <subcommand>` works.

fn main() {
    xtask::delegate::run_delegated_main(
        "xtask-evidence",
        "`xtask` assigns these gates here because each one reads recorded benchmark or release evidence.",
        xtask_evidence::GATES,
    );
}
