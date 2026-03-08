/// Truncate a string to `max` bytes, appending "..." if truncated.
/// Handles multi-byte UTF-8 safely by finding the nearest char boundary.
pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_string_unchanged() {
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn exact_length_unchanged() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn long_string_truncated_with_ellipsis() {
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn empty_string() {
        assert_eq!(truncate("", 5), "");
    }

    #[test]
    fn zero_max() {
        assert_eq!(truncate("abc", 0), "...");
    }

    #[test]
    fn multibyte_utf8_safe() {
        let result = truncate("héllo", 2);
        assert!(result.ends_with("..."));
        // 'h' is 1 byte, 'é' is 2 bytes, so at max=2 we can fit "hé" (3 bytes > 2),
        // boundary backs up to 1, giving "h..."
        assert_eq!(result, "h...");
    }

    #[test]
    fn multibyte_cjk() {
        let result = truncate("日本語テスト", 4);
        // '日' is 3 bytes, next char starts at byte 3, byte 4 is mid-char, backs to 3
        assert!(result.ends_with("..."));
        assert_eq!(result, "日...");
    }
}
