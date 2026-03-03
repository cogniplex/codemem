use super::*;

#[test]
fn new_file_is_changed() {
    let detector = ChangeDetector::new();
    assert!(detector.is_changed("foo.rs", b"fn main() {}"));
}

#[test]
fn same_content_not_changed() {
    let mut detector = ChangeDetector::new();
    let content = b"fn main() {}";
    detector.update_hash("foo.rs", content);
    assert!(!detector.is_changed("foo.rs", content));
}

#[test]
fn different_content_is_changed() {
    let mut detector = ChangeDetector::new();
    detector.update_hash("foo.rs", b"fn main() {}");
    assert!(detector.is_changed("foo.rs", b"fn main() { println!(\"hi\"); }"));
}

#[test]
fn remove_hash_makes_changed() {
    let mut detector = ChangeDetector::new();
    detector.update_hash("foo.rs", b"content");
    assert!(!detector.is_changed("foo.rs", b"content"));
    detector.remove_hash("foo.rs");
    assert!(detector.is_changed("foo.rs", b"content"));
}

#[test]
fn tracked_count() {
    let mut detector = ChangeDetector::new();
    assert_eq!(detector.tracked_count(), 0);
    detector.update_hash("a.rs", b"a");
    detector.update_hash("b.rs", b"b");
    assert_eq!(detector.tracked_count(), 2);
}
