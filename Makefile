.PHONY: build release test bench check clippy fmt clean install serve ui-build ui-lint

build:
	cargo build --workspace

release: ui-build
	cargo build --release --features ui

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

install: ui-build
	cargo install --path crates/codemem --features ui

serve:
	cargo run -- serve

ui-build:
	cd ui && bun install && bun run build
	rm -rf crates/codemem/ui-dist
	cp -r ui/dist crates/codemem/ui-dist

ui-lint:
	cd ui && bun run tsc --noEmit && bun run eslint .
