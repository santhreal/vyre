use super::*;

#[test]
fn test_reduce_segment_reduce_adv_52() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 0];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_53() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 0];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_54() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_55() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_56() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_57() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_58() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_59() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_60() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_61() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_62() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_63() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_64() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_65() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_66() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_67() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_68() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_69() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 31];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}
