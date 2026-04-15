# Changelog

## [0.17.2](https://github.com/cogniplex/codemem/compare/v0.17.1...v0.17.2) (2026-04-15)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.19.0 to 0.20.0
    * codemem-engine bumped from 0.19.0 to 0.20.0

## [0.17.1](https://github.com/cogniplex/codemem/compare/v0.17.0...v0.17.1) (2026-04-13)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.18.0 to 0.19.0
    * codemem-engine bumped from 0.18.0 to 0.19.0

## [0.17.0](https://github.com/cogniplex/codemem/compare/v0.16.0...v0.17.0) (2026-04-13)


### Features

* CLI UX overhaul — mcp subcommand group ([#72](https://github.com/cogniplex/codemem/issues/72)) ([e5fd074](https://github.com/cogniplex/codemem/commit/e5fd07401c2dda67e764f49da516d375f04abca7))
* UI overhaul — 3-page layout, graph code viewer ([#78](https://github.com/cogniplex/codemem/issues/78)) ([6e22545](https://github.com/cogniplex/codemem/commit/6e225452aff3ef85e6e9cfbfc45c2ca0c909a071))


### Bug Fixes

* code-mapper pkg: fallback + auto-allow codemem MCP tools ([#73](https://github.com/cogniplex/codemem/issues/73)) ([0cd7e9c](https://github.com/cogniplex/codemem/commit/0cd7e9cfc5f065041c269b5a3d3fbc8d1588c207))
* filter orphan edges + resolve SCIP indexer paths ([#74](https://github.com/cogniplex/codemem/issues/74)) ([c3e69a2](https://github.com/cogniplex/codemem/commit/c3e69a28b7269051d9cc2d131faf6998020be7a0))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.17.0 to 0.18.0
    * codemem-engine bumped from 0.17.0 to 0.18.0

## [0.16.0](https://github.com/cogniplex/codemem/compare/v0.15.0...v0.16.0) (2026-04-07)


### Features

* JinaBERT support, F16 default, configurable embedding model ([#68](https://github.com/cogniplex/codemem/issues/68)) ([48f423d](https://github.com/cogniplex/codemem/commit/48f423d4f85ee0174276491ea1ace35bac37d214))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.16.0 to 0.17.0
    * codemem-engine bumped from 0.16.0 to 0.17.0

## [0.15.0](https://github.com/cogniplex/codemem/compare/v0.14.1...v0.15.0) (2026-03-27)


### Features

* graph intelligence — SCIP noise filtering, test impact, co-change detection ([#63](https://github.com/cogniplex/codemem/issues/63)) ([5723bc6](https://github.com/cogniplex/codemem/commit/5723bc6df45b63be8bd6c2684513ee0b18c1b26a))


### Bug Fixes

* namespace-scoped PageRank to prevent cross-project score pollution ([#61](https://github.com/cogniplex/codemem/issues/61)) ([5316a89](https://github.com/cogniplex/codemem/commit/5316a892f26dc49ff3d067b55faf9507ab57a320))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.15.0 to 0.16.0
    * codemem-engine bumped from 0.15.0 to 0.16.0

## [0.14.1](https://github.com/cogniplex/codemem/compare/v0.14.0...v0.14.1) (2026-03-18)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.14.0 to 0.15.0
    * codemem-engine bumped from 0.14.0 to 0.15.0

## [0.14.0](https://github.com/cogniplex/codemem/compare/v0.13.0...v0.14.0) (2026-03-18)


### Features

* temporal graph layer — commit/PR nodes, ModifiedBy edges, temporal queries ([#52](https://github.com/cogniplex/codemem/issues/52)) ([3679b22](https://github.com/cogniplex/codemem/commit/3679b2221bf1ac7c8bcbd179e35d9726908e494b))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.13.0 to 0.14.0
    * codemem-engine bumped from 0.13.0 to 0.14.0

## [0.13.0](https://github.com/cogniplex/codemem/compare/v0.12.0...v0.13.0) (2026-03-16)


### Features

* diff-aware review pipeline with blast radius analysis ([#46](https://github.com/cogniplex/codemem/issues/46)) ([9285d57](https://github.com/cogniplex/codemem/commit/9285d57f8b1cf2ae8491e1cfb7ea34ec010272ad))
* memory expiration with opportunistic cleanup ([#41](https://github.com/cogniplex/codemem/issues/41)) ([cf8e995](https://github.com/cogniplex/codemem/commit/cf8e995df43f908ee77dea90520d3483b5a059ca))
* scope context with repo/branch/user awareness ([#45](https://github.com/cogniplex/codemem/issues/45)) ([d6ec98f](https://github.com/cogniplex/codemem/commit/d6ec98f7f20a4246a0b1ac63b33164710525a244))


### Bug Fixes

* 12 correctness, safety, and pipeline fixes from code review ([#47](https://github.com/cogniplex/codemem/issues/47)) ([71d628f](https://github.com/cogniplex/codemem/commit/71d628f3f76c342c0087e107b1faad2682222068))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.12.0 to 0.13.0
    * codemem-engine bumped from 0.12.0 to 0.13.0

## [0.12.0](https://github.com/cogniplex/codemem/compare/v0.11.0...v0.12.0) (2026-03-13)


### Features

* LSP enrichment + cross-repo linking pipeline ([#33](https://github.com/cogniplex/codemem/issues/33)) ([a74bde5](https://github.com/cogniplex/codemem/commit/a74bde595b567f6c79c58ff19c134f580258348a))
* replace LSP enrichment with SCIP integration ([#39](https://github.com/cogniplex/codemem/issues/39)) ([b50dd29](https://github.com/cogniplex/codemem/commit/b50dd29f0c0c12718b1191d6eaf178937d41c33f))


### Bug Fixes

* namespace-scoped file hashes, configurable embed batch size, hook storage ([#35](https://github.com/cogniplex/codemem/issues/35)) ([1c5bb9c](https://github.com/cogniplex/codemem/commit/1c5bb9c30ee9cce6868e60fb7a9400c2474a5d1c))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.11.0 to 0.12.0
    * codemem-engine bumped from 0.11.0 to 0.12.0

## [0.11.0](https://github.com/cogniplex/codemem/compare/v0.10.4...v0.11.0) (2026-03-11)


### Features

* configurable embedding model, dtype, and batch size ([#31](https://github.com/cogniplex/codemem/issues/31)) ([6dfbfce](https://github.com/cogniplex/codemem/commit/6dfbfce0377b5e46c0a2de8907ac11a53f19e490))
* enhance agent prompts with typed relationships and top-down processing ([#25](https://github.com/cogniplex/codemem/issues/25)) ([867bbac](https://github.com/cogniplex/codemem/commit/867bbace136a31550941671f56adce33b51cd574))
* incremental re-indexing with symbol-level diff ([#26](https://github.com/cogniplex/codemem/issues/26)) ([872b10f](https://github.com/cogniplex/codemem/commit/872b10f05cbe35d44e04f87767dcd18c5f5c8ba7))


### Bug Fixes

* Claude Code hooks spec compliance (issue [#27](https://github.com/cogniplex/codemem/issues/27)) ([#29](https://github.com/cogniplex/codemem/issues/29)) ([dafc4e8](https://github.com/cogniplex/codemem/commit/dafc4e865ced97ddbcb4c7d98d0e9b10de723519))


### Performance Improvements

* lazy init for vector/BM25/embeddings in CodememEngine ([#28](https://github.com/cogniplex/codemem/issues/28)) ([823cbc1](https://github.com/cogniplex/codemem/commit/823cbc1ed798fc0988c07e63510a51eddc6c0fb6))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.10.1 to 0.11.0
    * codemem-engine bumped from 0.10.3 to 0.11.0

## [0.10.4](https://github.com/cogniplex/codemem/compare/v0.10.3...v0.10.4) (2026-03-09)


### Bug Fixes

* use directory basename as namespace instead of full path ([#23](https://github.com/cogniplex/codemem/issues/23)) ([99cb532](https://github.com/cogniplex/codemem/commit/99cb53275d675f0f9edaea5ae708f900cb819355))

## [0.10.3](https://github.com/cogniplex/codemem/compare/v0.10.2...v0.10.3) (2026-03-09)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-engine bumped from 0.10.2 to 0.10.3

## [0.10.2](https://github.com/cogniplex/codemem/compare/v0.10.1...v0.10.2) (2026-03-09)


### Bug Fixes

* cargo install codemem broken on crates.io ([#17](https://github.com/cogniplex/codemem/issues/17)) ([13e5e78](https://github.com/cogniplex/codemem/commit/13e5e781ead4298ec0d2078071ac3f03f487a48e))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-engine bumped from 0.10.1 to 0.10.2

## [0.10.1](https://github.com/cogniplex/codemem/compare/v0.10.0...v0.10.1) (2026-03-09)


### Refactoring

* fix bugs, unify flows, add safe accessors, define constants ([#16](https://github.com/cogniplex/codemem/issues/16)) ([1554c3f](https://github.com/cogniplex/codemem/commit/1554c3f64f0c2e456cbe5f17548a813f62a9a4f4))
* tier 1 quick wins — dead code, wiring, visibility, dedup ([#13](https://github.com/cogniplex/codemem/issues/13)) ([56a469a](https://github.com/cogniplex/codemem/commit/56a469a25ad57a00cfeee6714b76e8c582f33ace))
* tier 2 quick wins — constructors, dedup helpers, storage ergonomics ([#14](https://github.com/cogniplex/codemem/issues/14)) ([e7243b7](https://github.com/cogniplex/codemem/commit/e7243b7f79af93662fcf644b31ea091dd863d0b4))
* tier 3 — domain logic to engine, drop binary storage/embeddings deps ([#15](https://github.com/cogniplex/codemem/issues/15)) ([a92b846](https://github.com/cogniplex/codemem/commit/a92b8463e2660b0318c01fa03f18f9ac864ddc39))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.10.0 to 0.10.1
    * codemem-engine bumped from 0.10.0 to 0.10.1

## [0.10.0](https://github.com/cogniplex/codemem/compare/v0.9.0...v0.10.0) (2026-03-08)


### Features

* session continuity and persistence pipeline improvements ([#10](https://github.com/cogniplex/codemem/issues/10)) ([970b6f8](https://github.com/cogniplex/codemem/commit/970b6f899dccee81e84143ad6a8ac96f8965307c))


### Bug Fixes

* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.9.0 to 0.10.0
    * codemem-storage bumped from 0.9.0 to 0.10.0
    * codemem-embeddings bumped from 0.8.1 to 0.9.0
    * codemem-engine bumped from 0.9.0 to 0.10.0

## [0.9.0](https://github.com/cogniplex/codemem/compare/v0.8.0...v0.9.0) (2026-03-08)


### Features

* **engine:** respect configured vector dimensions ([9a4d18d](https://github.com/cogniplex/codemem/commit/9a4d18decd256d3549d86cd289bda7c9453043b8))
* **graph:** tag-based auto-linking and memory-neighbor scoring ([1403797](https://github.com/cogniplex/codemem/commit/1403797ebca46847e617e8d9a2faa9da31565e5a))
* track per-session metadata in MCP HTTP transport ([95c5f06](https://github.com/cogniplex/codemem/commit/95c5f062ff1c1d90b2cc18494533928d48e95339))
* **ui:** replace Sigma.js with react-force-graph-2d, overhaul PointCloud ([56ba13a](https://github.com/cogniplex/codemem/commit/56ba13afb044a9ef4a1cd27d4261e00695d3700a))
* wire up all 14 enrichment types in enrich_codebase tool ([c389e3f](https://github.com/cogniplex/codemem/commit/c389e3f11d16b4e1bf5054be6768ed4157a1f4f4))


### Bug Fixes

* **api:** route memory CRUD through engine pipeline ([e21a4a7](https://github.com/cogniplex/codemem/commit/e21a4a7ea94071e6f8795f8da174bbd0aaadb8bc))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* create ~/.codemem directory in serve command before opening DB ([1353b75](https://github.com/cogniplex/codemem/commit/1353b75d00f235ca235888f0336660aefd6eb3a2))
* implement MCP SSE server-push for GET /mcp endpoint ([85a8c3f](https://github.com/cogniplex/codemem/commit/85a8c3f9bd024ec1504d0502fd6fa53ddd4c33f0))
* propagate config changes to running engine in API routes ([a1c2f86](https://github.com/cogniplex/codemem/commit/a1c2f8632699a3a95852422b5e35b61a05c1e4f4))
* use full persist pipeline in store_pattern_memory, remove dead code ([a262533](https://github.com/cogniplex/codemem/commit/a2625331cb876c869cd2b921621c8d38b4c77f81))


### Refactoring

* consolidate enrichment dispatch into run_enrichments() ([77b5921](https://github.com/cogniplex/codemem/commit/77b592168f3dacd54fec38f33a3c37437bbed69e))
* encapsulate CodememEngine fields behind accessor methods ([5d64df7](https://github.com/cogniplex/codemem/commit/5d64df7335bf76aa4d919b673159f9cf6aae52f7))
* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))
* remove legacy MCP tool aliases and update to canonical names ([f03809d](https://github.com/cogniplex/codemem/commit/f03809dc5b611469282ef6f72941d842d43e5e62))
* rewrite code-mapper as multi-agent team ([fa09a04](https://github.com/cogniplex/codemem/commit/fa09a04ba2cdb84719d6a8d64627a1c9e04fe9f6))
* split large test files into focused modules ([faf8ff4](https://github.com/cogniplex/codemem/commit/faf8ff43ee23a93e4db9359961f0fd6593b69cbc))
* split monolithic engine files into focused modules ([be79b09](https://github.com/cogniplex/codemem/commit/be79b097cf36fa443f7c6dd1961417187422e83a))


### Tests

* add 201 new tests across engine, API, CLI, and MCP layers ([b5e26a8](https://github.com/cogniplex/codemem/commit/b5e26a82f12d0135c10141f22e7aeb9b24882d78))
* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
* add embedding provider and CLI lifecycle test coverage ([1793bdf](https://github.com/cogniplex/codemem/commit/1793bdf218e21320d763881bab9e1c2eedac33f2))
* **cli:** expand CLI command test coverage with extracted testable functions ([938ee04](https://github.com/cogniplex/codemem/commit/938ee04c3590a119ec37677c490b3720e3b1e9d1))


### Miscellaneous

* fix .gitignore to exclude ui-dist build artifacts ([a78ac22](https://github.com/cogniplex/codemem/commit/a78ac22f09a0e99375eefcca46ac61f70ec97a3c))
