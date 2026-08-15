//! Byte grammar of a C integer literal, shared by every kernel that scans one.
//!
//! Three classifications make up the grammar: the radix prefix, the value of a
//! single digit byte, and the trailing type suffix. `gpu_int_literal_scan` and
//! the `#if` evaluator inside `gpu_if_expression` each carried their own copy of
//! all three, and the copies drifted: the evaluator's accumulator wraps where
//! the scanner saturates, so `#if 4294967296` is false through the evaluator and
//! true through the scanner. The classifications live here so a change to the
//! digit table, the radix table or the suffix set reaches both.
//!
//! Every builder takes the binding names its caller uses. A straight-line kernel
//! cannot reuse one name across scopes, so the names, not the shapes, are what
//! differ between the two callers.

use vyre_foundation::ir::{Expr, Node};

/// Digit value standing for "not a digit in any supported radix".
///
/// Larger than the largest valid digit of radix 16, so the caller's
/// `digit < radix` test rejects the byte without a separate validity flag.
pub(super) const NON_DIGIT: u32 = 99;

/// `1` when `byte` is an ASCII decimal digit, else `0`.
pub(super) fn dec_digit_flag(byte: Expr) -> Expr {
    Expr::select(
        Expr::and(
            Expr::ge(byte.clone(), Expr::u32(b'0' as u32)),
            Expr::le(byte, Expr::u32(b'9' as u32)),
        ),
        Expr::u32(1),
        Expr::u32(0),
    )
}

fn byte_in_range_flag(byte: &str, low: u8, high: u8) -> Expr {
    Expr::select(
        Expr::and(
            Expr::ge(Expr::var(byte), Expr::u32(u32::from(low))),
            Expr::le(Expr::var(byte), Expr::u32(u32::from(high))),
        ),
        Expr::u32(1),
        Expr::u32(0),
    )
}

fn prefix_flag(first_byte: &str, second_byte: &str, lower: u8, upper: u8) -> Expr {
    Expr::select(
        Expr::and(
            Expr::eq(Expr::var(first_byte), Expr::u32(b'0' as u32)),
            Expr::or(
                Expr::eq(Expr::var(second_byte), Expr::u32(u32::from(lower))),
                Expr::eq(Expr::var(second_byte), Expr::u32(u32::from(upper))),
            ),
        ),
        Expr::u32(1),
        Expr::u32(0),
    )
}

/// Radix prefix of the literal starting at `first_byte`.
///
/// Binds `is_hex`, `is_bin` and `is_oct`, then `radix_var` (16, 2, 8 or 10) and
/// `skip_var`, the byte count the prefix itself occupies. Octal is the leading-`0`
/// case left once hex and binary are ruled out, so the three flags are ordered
/// and `is_oct` reads the first two.
pub(super) fn push_radix_class(
    parse: &mut Vec<Node>,
    first_byte: &str,
    second_byte: &str,
    radix_var: &str,
    skip_var: &str,
) {
    parse.push(Node::let_bind(
        "is_hex",
        prefix_flag(first_byte, second_byte, b'x', b'X'),
    ));
    parse.push(Node::let_bind(
        "is_bin",
        prefix_flag(first_byte, second_byte, b'b', b'B'),
    ));
    parse.push(Node::let_bind(
        "is_oct",
        Expr::select(
            Expr::and(
                Expr::eq(Expr::var(first_byte), Expr::u32(b'0' as u32)),
                Expr::and(
                    Expr::eq(Expr::var("is_hex"), Expr::u32(0)),
                    Expr::eq(Expr::var("is_bin"), Expr::u32(0)),
                ),
            ),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
    parse.push(Node::let_bind(
        radix_var,
        Expr::select(
            Expr::eq(Expr::var("is_hex"), Expr::u32(1)),
            Expr::u32(16),
            Expr::select(
                Expr::eq(Expr::var("is_bin"), Expr::u32(1)),
                Expr::u32(2),
                Expr::select(
                    Expr::eq(Expr::var("is_oct"), Expr::u32(1)),
                    Expr::u32(8),
                    Expr::u32(10),
                ),
            ),
        ),
    ));
    parse.push(Node::let_bind(
        skip_var,
        Expr::select(
            Expr::or(
                Expr::eq(Expr::var("is_hex"), Expr::u32(1)),
                Expr::eq(Expr::var("is_bin"), Expr::u32(1)),
            ),
            Expr::u32(2),
            Expr::u32(0),
        ),
    ));
}

/// Bindings one digit-value decode leaves in scope.
pub(super) struct DigitValue<'a> {
    /// Byte being decoded.
    pub(super) byte_var: &'a str,
    /// Decimal-digit flag.
    pub(super) dec_var: &'a str,
    /// Lowercase hex-digit flag.
    pub(super) hex_lower_var: &'a str,
    /// Uppercase hex-digit flag.
    pub(super) hex_upper_var: &'a str,
    /// Digit value, or [`NON_DIGIT`] when the byte is no digit at all.
    pub(super) value_var: &'a str,
}

/// Decode one byte to its digit value, radix-independent.
///
/// The value is not range-checked here: the caller compares it against its own
/// radix, which is what rejects `9` in octal and every non-digit byte alike.
pub(super) fn push_digit_value(parse: &mut Vec<Node>, names: &DigitValue<'_>) {
    parse.push(Node::let_bind(
        names.dec_var,
        byte_in_range_flag(names.byte_var, b'0', b'9'),
    ));
    parse.push(Node::let_bind(
        names.hex_lower_var,
        byte_in_range_flag(names.byte_var, b'a', b'f'),
    ));
    parse.push(Node::let_bind(
        names.hex_upper_var,
        byte_in_range_flag(names.byte_var, b'A', b'F'),
    ));
    parse.push(Node::let_bind(
        names.value_var,
        Expr::select(
            Expr::eq(Expr::var(names.dec_var), Expr::u32(1)),
            Expr::sub(Expr::var(names.byte_var), Expr::u32(b'0' as u32)),
            Expr::select(
                Expr::eq(Expr::var(names.hex_lower_var), Expr::u32(1)),
                Expr::add(
                    Expr::sub(Expr::var(names.byte_var), Expr::u32(b'a' as u32)),
                    Expr::u32(10),
                ),
                Expr::select(
                    Expr::eq(Expr::var(names.hex_upper_var), Expr::u32(1)),
                    Expr::add(
                        Expr::sub(Expr::var(names.byte_var), Expr::u32(b'A' as u32)),
                        Expr::u32(10),
                    ),
                    Expr::u32(NON_DIGIT),
                ),
            ),
        ),
    ));
}

/// Fold one digit into the running value, saturating at `u32::MAX`.
///
/// Saturation, not wrapping, is what the CPU reference does: it accumulates in
/// `u64` with `saturating_mul`/`saturating_add`. A literal above `u32::MAX` must
/// stay truthy in a `#if`, and wrapping can carry it to zero, which flips the
/// conditional. A digit separator contributes no digit and leaves the value
/// alone.
///
/// Binds `sat_limit` and `would_overflow` in the caller's scope.
pub(super) fn push_saturating_digit_accumulate(
    parse: &mut Vec<Node>,
    value_var: &str,
    radix_var: &str,
    digit_var: &str,
    separator_var: &str,
) {
    parse.push(Node::let_bind(
        "sat_limit",
        Expr::div(
            Expr::sub(Expr::u32(u32::MAX), Expr::var(digit_var)),
            Expr::var(radix_var),
        ),
    ));
    parse.push(Node::let_bind(
        "would_overflow",
        Expr::select(
            Expr::gt(Expr::var(value_var), Expr::var("sat_limit")),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
    parse.push(Node::assign(
        value_var,
        Expr::select(
            Expr::eq(Expr::var(separator_var), Expr::u32(1)),
            Expr::var(value_var),
            Expr::select(
                Expr::eq(Expr::var("would_overflow"), Expr::u32(1)),
                Expr::u32(u32::MAX),
                Expr::add(
                    Expr::mul(Expr::var(value_var), Expr::var(radix_var)),
                    Expr::var(digit_var),
                ),
            ),
        ),
    ));
}

/// Bindings one type-suffix classification leaves in scope.
pub(super) struct SuffixClass<'a> {
    /// First suffix byte.
    pub(super) byte_var: &'a str,
    /// Byte after it, needed because `wb` is the only two-byte suffix.
    pub(super) next_byte_var: &'a str,
    /// `u`/`U`/`l`/`L` flag.
    pub(super) single_var: &'a str,
    /// `z`/`Z` flag.
    pub(super) z_var: &'a str,
    /// `wb`/`WB` flag.
    pub(super) wb_var: &'a str,
}

/// Classify the bytes at the cursor as a C integer type suffix.
///
/// The supported set is `u`, `U`, `l`, `L`, `z`, `Z`, `wb` and `WB`. A suffix
/// carries no value for a preprocessor conditional, so both callers consume it
/// and drop it; adding one here reaches both.
pub(super) fn push_suffix_class(parse: &mut Vec<Node>, names: &SuffixClass<'_>) {
    parse.push(Node::let_bind(
        names.single_var,
        Expr::select(
            Expr::or(
                Expr::or(
                    Expr::eq(Expr::var(names.byte_var), Expr::u32(b'u' as u32)),
                    Expr::eq(Expr::var(names.byte_var), Expr::u32(b'U' as u32)),
                ),
                Expr::or(
                    Expr::eq(Expr::var(names.byte_var), Expr::u32(b'l' as u32)),
                    Expr::eq(Expr::var(names.byte_var), Expr::u32(b'L' as u32)),
                ),
            ),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
    parse.push(Node::let_bind(
        names.z_var,
        Expr::select(
            Expr::or(
                Expr::eq(Expr::var(names.byte_var), Expr::u32(b'z' as u32)),
                Expr::eq(Expr::var(names.byte_var), Expr::u32(b'Z' as u32)),
            ),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
    parse.push(Node::let_bind(
        names.wb_var,
        Expr::select(
            Expr::and(
                Expr::or(
                    Expr::eq(Expr::var(names.byte_var), Expr::u32(b'w' as u32)),
                    Expr::eq(Expr::var(names.byte_var), Expr::u32(b'W' as u32)),
                ),
                Expr::or(
                    Expr::eq(Expr::var(names.next_byte_var), Expr::u32(b'b' as u32)),
                    Expr::eq(Expr::var(names.next_byte_var), Expr::u32(b'B' as u32)),
                ),
            ),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
}

/// `1` when [`push_suffix_class`] matched any suffix.
pub(super) fn suffix_matched_expr(names: &SuffixClass<'_>) -> Expr {
    Expr::or(
        Expr::or(
            Expr::eq(Expr::var(names.single_var), Expr::u32(1)),
            Expr::eq(Expr::var(names.z_var), Expr::u32(1)),
        ),
        Expr::eq(Expr::var(names.wb_var), Expr::u32(1)),
    )
}

/// Bytes the matched suffix occupies: 2 for `wb`/`WB`, 1 for the rest.
pub(super) fn suffix_advance_expr(names: &SuffixClass<'_>) -> Expr {
    Expr::select(
        Expr::eq(Expr::var(names.wb_var), Expr::u32(1)),
        Expr::u32(2),
        Expr::u32(1),
    )
}
