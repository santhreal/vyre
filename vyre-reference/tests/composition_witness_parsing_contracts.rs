//! Integration contract tests for the sequential LR parser reference witness.

use vyre_reference::composition_witness::{
    is_structural_whitespace_witness, line_splice_classify_witness,
    line_splice_classify_witness_into, parse_lr_witness, whitespace_classify_word_witness,
    whitespace_classify_word_witness_into, LrProduction, ParseLrWitnessError,
};

const fn pack_shift(s: u32) -> u32 {
    (1 << 30) | (s & 0x3FFF_FFFF)
}

const fn pack_reduce(p: u32) -> u32 {
    (2 << 30) | (p & 0x3FFF_FFFF)
}

const fn pack_accept() -> u32 {
    3 << 30
}

#[test]
fn test_parse_lr_witness_simple_accept() {
    // Grammar: S -> a
    // State 0: on 'a' (token 0) -> shift 1, on EOF (token 1) -> error
    // State 1: on EOF (token 1) -> reduce 1
    // State 2: on EOF (token 1) -> accept
    // Goto: State 0 on S (nt 0) -> State 2
    let action = vec![
        pack_shift(1),
        0, // State 0: token 0 -> shift 1, token 1 -> error
        0,
        pack_reduce(1), // State 1: token 0 -> error, token 1 -> reduce 1
        0,
        pack_accept(), // State 2: token 0 -> error, token 1 -> accept
    ];
    let goto = vec![
        2,        // State 0 on nt 0 -> 2
        u32::MAX, // State 1
        u32::MAX, // State 2
    ];
    let prods = vec![
        LrProduction { lhs: 0, rhs_len: 1 }, // Prod 0 (start)
        LrProduction { lhs: 0, rhs_len: 1 }, // Prod 1: S -> a
    ];

    let tokens = [0u32, 1u32]; // 'a', EOF
    let res = parse_lr_witness(&action, &goto, &prods, 2, 1, &tokens);
    assert_eq!(res.expect("valid parse"), vec![1]);
}

#[test]
fn test_parse_lr_witness_unexpected_token_error() {
    let action = vec![
        0, 0, // State 0: all error
    ];
    let goto = vec![u32::MAX];
    let prods = vec![LrProduction { lhs: 0, rhs_len: 0 }];

    let tokens = [0u32];
    let err = parse_lr_witness(&action, &goto, &prods, 2, 1, &tokens).unwrap_err();
    assert_eq!(
        err,
        ParseLrWitnessError::UnexpectedToken {
            state: 0,
            token: 0,
            pos: 0
        }
    );
    assert_eq!(
        err.to_string(),
        "LR unexpected token: state=0 token=0 pos=0. Fix: validate token stream against grammar or extend action table."
    );
}

#[test]
fn test_parse_lr_witness_invalid_production_error() {
    let action = vec![
        pack_reduce(99),
        0, // State 0: reduce 99 (invalid)
    ];
    let goto = vec![u32::MAX];
    let prods = vec![LrProduction { lhs: 0, rhs_len: 0 }];

    let tokens = [0u32];
    let err = parse_lr_witness(&action, &goto, &prods, 2, 1, &tokens).unwrap_err();
    assert_eq!(err, ParseLrWitnessError::InvalidProduction { prod_id: 99 });
    assert_eq!(
        err.to_string(),
        "LR invalid production id 99 in action table. Fix: rebuild tables so every reduce action references a valid production."
    );
}

#[test]
fn test_parse_lr_witness_stack_underflow_error() {
    let action = vec![
        pack_reduce(1),
        0, // State 0: reduce 1 with rhs_len = 5 (stack only has 1 element)
    ];
    let goto = vec![u32::MAX];
    let prods = vec![
        LrProduction { lhs: 0, rhs_len: 0 },
        LrProduction { lhs: 0, rhs_len: 5 },
    ];

    let tokens = [0u32];
    let err = parse_lr_witness(&action, &goto, &prods, 2, 1, &tokens).unwrap_err();
    assert_eq!(err, ParseLrWitnessError::StackUnderflow);
    assert_eq!(
        err.to_string(),
        "LR stack underflow on reduce. Fix: verify push/pop balance in grammar and that goto table matches."
    );
}

#[test]
fn test_parse_lr_witness_no_goto_error() {
    let action = vec![pack_shift(1), 0, pack_reduce(1), 0];
    let goto = vec![
        u32::MAX, // State 0: no goto for nt 0
        u32::MAX,
    ];
    let prods = vec![
        LrProduction { lhs: 0, rhs_len: 0 },
        LrProduction { lhs: 0, rhs_len: 1 },
    ];

    let tokens = [0u32, 0u32];
    let err = parse_lr_witness(&action, &goto, &prods, 2, 1, &tokens).unwrap_err();
    assert_eq!(
        err,
        ParseLrWitnessError::NoGoto {
            state: 0,
            nonterminal: 0
        }
    );
    assert_eq!(
        err.to_string(),
        "LR missing goto: state=0 nt=0. Fix: regenerate goto table from closure sets."
    );
}

#[test]
fn test_parse_lr_witness_out_of_range_token() {
    let action = vec![0, 0];
    let goto = vec![u32::MAX];
    let prods = vec![LrProduction { lhs: 0, rhs_len: 0 }];

    let tokens = [100u32]; // Token 100 exceeds num_tokens = 2
    let err = parse_lr_witness(&action, &goto, &prods, 2, 1, &tokens).unwrap_err();
    assert_eq!(
        err,
        ParseLrWitnessError::UnexpectedToken {
            state: 0,
            token: 100,
            pos: 0
        }
    );
}

#[test]
fn test_line_splice_classify_witness_known_answers() {
    assert!(line_splice_classify_witness(b"").is_empty());
    assert_eq!(line_splice_classify_witness(b"a\\\nb"), vec![1, 0, 0, 1]);
    assert_eq!(
        line_splice_classify_witness(b"a\\\r\nb"),
        vec![1, 0, 0, 0, 1]
    );
    assert_eq!(line_splice_classify_witness(b"a\\\rb"), vec![1, 0, 0, 1]);
    assert_eq!(line_splice_classify_witness(b"a\\ b"), vec![1, 1, 1, 1]);

    let mut out = Vec::new();
    line_splice_classify_witness_into(b"a\\\nB", &mut out);
    assert_eq!(out, vec![1, 0, 0, 1]);
}

#[test]
fn test_whitespace_classify_word_witness_known_answers() {
    assert!(is_structural_whitespace_witness(b' '));
    assert!(is_structural_whitespace_witness(b'\t'));
    assert!(is_structural_whitespace_witness(b'\n'));
    assert!(is_structural_whitespace_witness(b'\r'));
    assert!(!is_structural_whitespace_witness(b'a'));

    // pack_bytes_le: (b0) | (b1 << 8) | (b2 << 16) | (b3 << 24)
    let word1 =
        (b'a' as u32) | ((b'b' as u32) << 8) | ((b'c' as u32) << 16) | ((b'd' as u32) << 24);
    assert_eq!(whitespace_classify_word_witness(&[word1]), vec![0]);

    let word2 =
        (b' ' as u32) | ((b'\t' as u32) << 8) | ((b'\n' as u32) << 16) | ((b'\r' as u32) << 24);
    assert_eq!(whitespace_classify_word_witness(&[word2]), vec![0b1111]);

    let word3 =
        (b'a' as u32) | ((b' ' as u32) << 8) | ((b'b' as u32) << 16) | ((b'\t' as u32) << 24);
    assert_eq!(whitespace_classify_word_witness(&[word3]), vec![0b1010]);

    let mut out = Vec::new();
    whitespace_classify_word_witness_into(&[word3], &mut out);
    assert_eq!(out, vec![0b1010]);
}
