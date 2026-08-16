//! Usage and argument handling shared by the operator binaries in this crate.
//!
//! Every operator binary here takes no input but a help flag: it reads the
//! checkout it is run in. So each one printed the same help skeleton and the same
//! argument gate, and the exit codes drifted between copies. One owner means a
//! new binary cannot disagree about what exit 2 means.

/// What a binary prints for `--help`, and what its exit codes mean.
pub struct Usage {
    /// The binary name, as it is invoked and as the error text names it.
    pub name: &'static str,
    /// One line stating what the binary does.
    pub summary: &'static str,
    /// Exit code and what it means, in ascending order.
    pub exit_codes: &'static [(u8, &'static str)],
}

impl Usage {
    /// The help text, so a test reads what an operator reads.
    pub fn render(&self) -> String {
        let mut out = format!("{}\n\nUsage: {}\n\nExit codes:\n", self.summary, self.name);
        for (code, meaning) in self.exit_codes {
            out.push_str(&format!("  {code}  {meaning}\n"));
        }
        out
    }

    /// Write the help text to standard output.
    pub fn print(&self) {
        print!("{}", self.render());
    }
}

/// Whether the caller asked for help, having accepted no other argument.
///
/// An unknown argument exits 2 rather than being ignored: a workflow that passes
/// one is asking for behavior the binary does not have, and running anyway is how
/// a caller believes it got what it asked for.
pub fn help_requested(usage: &Usage) -> bool {
    let mut args = std::env::args().skip(1);
    let Some(argument) = args.next() else {
        return false;
    };
    if matches!(argument.as_str(), "-h" | "--help") && args.next().is_none() {
        usage.print();
        return true;
    }
    eprintln!(
        "Fix: unknown argument `{argument}`. Use `{} --help`.",
        usage.name
    );
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the help text is what an operator reads before running a binary that
    /// touches the checkout, and an exit-code table that omits a code the binary
    /// returns is worse than none. The rendering is the contract.
    #[test]
    fn the_help_text_states_the_invocation_and_every_exit_code() {
        let usage = Usage {
            name: "example_binary",
            summary: "Do the one thing.",
            exit_codes: &[(0, "it worked"), (2, "arguments are invalid")],
        };
        assert_eq!(
            usage.render(),
            "Do the one thing.\n\nUsage: example_binary\n\nExit codes:\n  0  it worked\n  2  \
             arguments are invalid\n"
        );
    }
}
