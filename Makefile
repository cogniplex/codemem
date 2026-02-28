.PHONY: build release test bench check clippy fmt clean install serve

build:
	cargo build --workspace

release:
	cargo build --release

test:
	cargo test --workspace

bench:
	cargo bench

check:
	cargo check --workspace

clippy:
	cargo clippy --workspace -- -D warnings

fmt:
	cargo fmt --all

clean:
	cargo clean

install:
	cargo install --path crates/codemem-cli

serve:
	cargo run -- serve
