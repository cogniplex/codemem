use super::*;

#[test]
fn print_check_ok_does_not_panic() {
    print_check("test", true, "");
}

#[test]
fn print_check_fail_with_detail_does_not_panic() {
    print_check("test", false, "some detail");
}
