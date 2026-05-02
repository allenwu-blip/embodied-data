# Security policy

`embodied-data` is a small CLI tool that reads dataset files (HDF5, parquet, mp4) and writes new dataset files. The threat surface is the parsing and re-encoding paths, not network or auth code.

## Supported versions

Only the latest minor release line gets active fixes. Earlier versions get advisories but no backports.

| Version | Status |
| --- | --- |
| 0.3.x  | Supported (current) |
| 0.2.x  | Advisory only |
| 0.1.x  | Advisory only |
| < 0.1  | Unsupported |

## Reporting a vulnerability

Email `cnwt20060606@outlook.com` directly with subject prefix `[security]`. **Do not file a public GitHub issue or PR for vulnerabilities.**

In your report, please include:

- Affected version (`embodied-data version` output if possible)
- Reproduction steps — minimal command, dataset shape, expected vs actual behavior
- Suggested mitigation, if you have one
- Whether the issue is being exploited in the wild (best-effort answer is fine)

## Response timeline

This is a solo OSS project. Best-effort response within **7 days** for an initial reply. If you don't hear back within that window, feel free to send a polite follow-up to the same address — assume the first email got buried, not ignored.

For coordinated disclosure: please give us **30 days** before any public write-up, unless the issue is already being actively exploited. We'll work with you on a credit line in the release notes if you want one.

## Scope

In-scope reports:

- Code execution via a crafted input dataset (HDF5, parquet, mp4) processed by `convert` / `validate` / `preview` / `inspect`.
- Secret or credential exposure (e.g., HF tokens written to disk, log output, error messages).
- Dependency CVEs that affect us through one of the pinned versions in `uv.lock`.
- Re-encoding bugs that produce dataset corruption silently (data integrity is part of the security model — the whole point of the tool is trustworthy output).

Out of scope:

- Denial-of-service via deliberately oversized fixtures (e.g., a 100 GB fake h5). The tool's job is to convert — running out of disk on a giant input is expected.
- Issues in transitive dev dependencies that don't affect runtime (lint tools, test runners).
- Bugs in upstream LeRobot / AgiBot World data formats themselves — please report those upstream.
- Social engineering, physical access, or anything outside the code path of `embodied-data` itself.
