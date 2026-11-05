# EnforceGate historical reference

**Status: Superseded.** Use `EnforceGate.txt` and
`vyre-driver/src/registry/enforce.rs` for the frozen contract.

pub trait EnforceGate: Send + Sync {
/// Name of this gate  -  appears in verdicts and logs.
fn name(&self) -> &'static str;
/// Evaluate the gate against `program`. Must be pure.
fn evaluate(&self, program: &Program) -> EnforceVerdict;
}
