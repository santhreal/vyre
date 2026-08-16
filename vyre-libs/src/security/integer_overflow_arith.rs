//! `integer_overflow_arith`  -  does this binary op overflow on
//! attacker input? CWE-190 supporting predicate.
//!
//! Per node `n`, write 1 iff `n` is a binary arithmetic node
//! (mul / add / shl) AND at least one operand is reachable from
//! `@http_input_family` AND there is no dominating overflow check.

use vyre_foundation::ir::Program;
use crate::bitset::and::bitset_and;
use crate::bitset::and_not::bitset_and_not;
use crate::bitset::bitset_words;

use crate::security::flow_composition::fuse_security_flow;

pub(crate) const OP_ID: &str = "vyre-libs::security::integer_overflow_arith";

/// Build an overflow-check Program: `arith_set AND attacker_reach`
/// lands in `intermediate`, then that set minus
/// `overflow_check_dominates` lands in `out`.
#[must_use]
pub fn integer_overflow_arith(
    node_count: u32,
    arith_set: &str,
    attacker_reach: &str,
    overflow_check_dominates: &str,
    intermediate: &str,
    out: &str,
) -> Program {
    let words = bitset_words(node_count);
    fuse_security_flow(
        OP_ID,
        &[
            bitset_and(arith_set, attacker_reach, intermediate, words),
            bitset_and_not(intermediate, overflow_check_dominates, out, words),
        ],
        out,
    )
}

/// CPU oracle.
#[must_use]
#[cfg(test)]
pub(crate) fn cpu_ref(
    arith_set: &[u32],
    attacker_reach: &[u32],
    overflow_check_dominates: &[u32],
) -> Vec<u32> {
    let inter = crate::bitset::and::cpu_ref(arith_set, attacker_reach);
    crate::bitset::and_not::cpu_ref(&inter, overflow_check_dominates)
}

/// Soundness marker for [`integer_overflow_arith`].
pub struct IntegerOverflowArith;
impl vyre_spec::soundness::SoundnessTagged for IntegerOverflowArith {
    fn soundness(&self) -> vyre_spec::soundness::Soundness {
        vyre_spec::soundness::Soundness::Exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unguarded_attacker_arith_fires() {
        // arith {0,1,2,3}, attacker {1,2}, no checks.
        assert_eq!(cpu_ref(&[0b1111], &[0b0110], &[0]), vec![0b0110]);
    }

    #[test]
    fn guarded_does_not_fire() {
        assert_eq!(cpu_ref(&[0b1111], &[0b0110], &[0b0010]), vec![0b0100]);
    }

    #[test]
    fn no_attacker_means_no_finding() {
        assert_eq!(cpu_ref(&[0b1111], &[0], &[0]), vec![0]);
    }

    #[test]
    fn no_arith_means_no_finding() {
        assert_eq!(cpu_ref(&[0], &[0xFFFF], &[0]), vec![0]);
    }
}
