# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| latest  | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in codemem, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email security@cogniplex.dev with:

- A description of the vulnerability
- Steps to reproduce it
- The potential impact
- Any suggested fix (optional)

We will acknowledge your report within 48 hours and aim to release a fix within 7 days for critical issues.

## Scope

codemem stores code analysis data locally in SQLite databases. Key areas of concern:

- **Local data storage**: Memory databases at `~/.codemem/` contain code snippets and embeddings
- **MCP server**: The REST/MCP server binds to localhost by default
- **Embedding providers**: External providers (Ollama, OpenAI) send code snippets over the network; the default Candle provider runs locally

## Best Practices

- Do not expose the MCP server to untrusted networks
- Review stored memories if working with sensitive codebases (`codemem recall` to inspect)
- Use the local Candle embedding provider if code must not leave your machine
