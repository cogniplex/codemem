use super::*;

#[test]
fn csv_escape_plain_text() {
    assert_eq!(csv_escape("hello world"), "hello world");
}

#[test]
fn csv_escape_with_commas() {
    assert_eq!(csv_escape("hello,world"), "\"hello,world\"");
}

#[test]
fn csv_escape_with_quotes() {
    assert_eq!(csv_escape("he said \"hi\""), "\"he said \"\"hi\"\"\"");
}

#[test]
fn csv_escape_with_newlines() {
    assert_eq!(csv_escape("line1\nline2"), "\"line1\nline2\"");
}

#[test]
fn csv_escape_combined() {
    assert_eq!(csv_escape("a,b\n\"c\""), "\"a,b\n\"\"c\"\"\"");
}
