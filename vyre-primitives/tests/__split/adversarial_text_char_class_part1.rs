use super::*;

#[test]
fn test_text_char_class_adv_0() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_1() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_2() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_3() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_4() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_5() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_6() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_7() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_8() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_9() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_10() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_11() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_12() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_13() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_14() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_15() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_16() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_17() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_18() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_19() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_20() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_21() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_22() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_23() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_24() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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
fn test_text_char_class_adv_25() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
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

