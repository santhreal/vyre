//! IR expressions over one haystack byte.
//!
//! Every C lexer builder reads the source through these: a masked word
//! load, an ASCII literal, an equality test, and the character classes the
//! C11 grammar splits on. They build `Expr` only; nothing here emits a
//! token.

use vyre_foundation::ir::Expr;

pub(super) fn byte_load(buffer: &str, index: Expr) -> Expr {
    Expr::bitand(Expr::load(buffer, index), Expr::u32(0xFF))
}

pub(super) fn ascii(byte: u8) -> Expr {
    Expr::u32(u32::from(byte))
}

pub(super) fn byte_eq(value: Expr, byte: u8) -> Expr {
    Expr::eq(value, ascii(byte))
}

pub(super) fn byte_at_or_zero(haystack: &str, index: Expr, haystack_len: u32) -> Expr {
    Expr::select(
        Expr::lt(index.clone(), Expr::u32(haystack_len)),
        byte_load(haystack, index),
        Expr::u32(0),
    )
}

pub(super) fn byte_between(value: Expr, low: u8, high: u8) -> Expr {
    Expr::and(
        Expr::ge(value.clone(), ascii(low)),
        Expr::le(value, ascii(high)),
    )
}

pub(super) fn is_alpha(value: Expr) -> Expr {
    Expr::or(
        byte_between(value.clone(), b'a', b'z'),
        byte_between(value, b'A', b'Z'),
    )
}

pub(super) fn is_digit(value: Expr) -> Expr {
    byte_between(value, b'0', b'9')
}

pub(super) fn is_octal_digit(value: Expr) -> Expr {
    byte_between(value, b'0', b'7')
}

pub(super) fn is_hex_digit(value: Expr) -> Expr {
    Expr::or(
        is_digit(value.clone()),
        Expr::or(
            byte_between(value.clone(), b'a', b'f'),
            byte_between(value, b'A', b'F'),
        ),
    )
}

pub(super) fn has_hex_digits_after(
    haystack: &str,
    escape_pos: Expr,
    digits: u32,
    haystack_len: u32,
) -> Expr {
    let mut expr = Expr::bool(true);
    for offset in 1..=digits {
        expr = Expr::and(
            expr,
            is_hex_digit(byte_at_or_zero(
                haystack,
                Expr::add(escape_pos.clone(), Expr::u32(offset)),
                haystack_len,
            )),
        );
    }
    expr
}

pub(super) fn is_valid_escape_byte(
    haystack: &str,
    escape_pos: Expr,
    escaped_byte: Expr,
    haystack_len: u32,
) -> Expr {
    let simple_escape = [
        b'\'', b'"', b'?', b'\\', b'a', b'b', b'e', b'f', b'n', b'r', b't', b'v', b'\n', b'\r',
    ]
    .into_iter()
    .fold(Expr::bool(false), |acc, byte| {
        Expr::or(acc, byte_eq(escaped_byte.clone(), byte))
    });

    Expr::or(
        simple_escape,
        Expr::or(
            is_octal_digit(escaped_byte.clone()),
            Expr::or(
                Expr::and(
                    Expr::or(
                        byte_eq(escaped_byte.clone(), b'x'),
                        byte_eq(escaped_byte.clone(), b'X'),
                    ),
                    has_hex_digits_after(haystack, escape_pos.clone(), 1, haystack_len),
                ),
                Expr::or(
                    Expr::and(
                        byte_eq(escaped_byte.clone(), b'u'),
                        has_hex_digits_after(haystack, escape_pos.clone(), 4, haystack_len),
                    ),
                    Expr::and(
                        byte_eq(escaped_byte, b'U'),
                        has_hex_digits_after(haystack, escape_pos, 8, haystack_len),
                    ),
                ),
            ),
        ),
    )
}

pub(super) fn is_ident_start(value: Expr) -> Expr {
    Expr::or(is_alpha(value.clone()), byte_eq(value, b'_'))
}

pub(super) fn is_ident_continue(value: Expr) -> Expr {
    Expr::or(is_ident_start(value.clone()), is_digit(value))
}
