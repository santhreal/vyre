use super::*;

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_0() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_1() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_2() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_3() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_4() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_5() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_6() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_7() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "empty segment_offsets")]
fn test_reduce_segment_reduce_adv_8() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 0];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
fn test_reduce_segment_reduce_adv_9() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_10() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_11() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_12() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_13() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_14() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_15() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_16() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_17() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 1 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_18() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 31 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_19() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 31 + 1);
}

#[test]
fn test_reduce_segment_reduce_adv_20() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    let res = result.expect("Fix: cpu_ref must not panic on adversarial input");
    assert!(res.len() <= 0 + 31 + 1);
}

#[test]
#[should_panic(expected = "malformed segment 0")]
fn test_reduce_segment_reduce_adv_21() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 31];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "malformed segment 0")]
fn test_reduce_segment_reduce_adv_22() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 31];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "malformed segment 0")]
fn test_reduce_segment_reduce_adv_23() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 31];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "malformed segment 0")]
fn test_reduce_segment_reduce_adv_24() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 31];
    let _ = cpu_ref(&input, &segment_offsets);
}

#[test]
#[should_panic(expected = "malformed segment 0")]
fn test_reduce_segment_reduce_adv_25() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 31];
    let _ = cpu_ref(&input, &segment_offsets);
}
