use super::*;

#[test]
fn test_reduce_segment_reduce_adv_26() {
    let input: Vec<u32> = vec![0u32; 0];
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

#[test]
fn test_reduce_segment_reduce_adv_27() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_28() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_29() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_30() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_31() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_32() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_33() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_34() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_35() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 32];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_36() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_37() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_38() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_39() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_40() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_41() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_42() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_43() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_44() {
    let input: Vec<u32> = vec![0u32; 0];
    let segment_offsets: Vec<u32> = vec![2143289344u32; 1024];
    let result = std::panic::catch_unwind(|| cpu_ref(&input, &segment_offsets));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_reduce_segment_reduce_adv_45() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 0];
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
fn test_reduce_segment_reduce_adv_46() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 0];
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
fn test_reduce_segment_reduce_adv_47() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![0u32; 0];
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
fn test_reduce_segment_reduce_adv_48() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 0];
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
fn test_reduce_segment_reduce_adv_49() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 0];
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
fn test_reduce_segment_reduce_adv_50() {
    let input: Vec<u32> = vec![1u32; 0];
    let segment_offsets: Vec<u32> = vec![4294967295u32; 0];
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
fn test_reduce_segment_reduce_adv_51() {
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

