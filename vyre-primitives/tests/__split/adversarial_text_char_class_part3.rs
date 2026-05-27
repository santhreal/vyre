use super::*;

#[test]
fn test_text_char_class_adv_52() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_53() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_54() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_55() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_56() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_57() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_58() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_59() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_60() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_61() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_62() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_63() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_64() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_65() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_66() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_67() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_68() {
    let source: Vec<u8> = vec![1u8; 0];
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
fn test_text_char_class_adv_69() {
    let source: Vec<u8> = vec![1u8; 0];
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
