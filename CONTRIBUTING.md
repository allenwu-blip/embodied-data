# Contributing to embodied-data

Thanks for stopping by. **embodied-data** is a solo open-source project
(maintained by Allen Wu, UMich Robotics) that converts between AgiBot World
and LeRobot v3 dataset formats. Issues, PRs, and questions are all welcome
— just be aware that response time is measured in days, not minutes.

## Quick start

```bash
git clone https://github.com/allenwu-blip/embodied-data
cd embodied-data
uv sync
uv run pytest -q
uv run embodied-data --help
```

`uv sync` installs runtime + dev dependencies (declared via
`dependency-groups` in `pyproject.toml`). Python 3.12+ is required.

## Project layout

- `src/embodied_data/convert/` — format converters (AgiBot → LeRobot v3,
  sim-only and Beta/Alpha pipelines).
- `src/embodied_data/validate/` — schema and dataset validators.
- `tests/` — pytest suite (unit + integration).
- `docs/schema/` — schema reference docs (AgiBot, LeRobot v3, mapping).
- `CHANGELOG.md` — Keep a Changelog format, user-visible changes.
- `README.md` — install, quickstart, supported pipelines.

## Running tests

Full suite:

```bash
uv run pytest -q
```

Single test by node id:

```bash
uv run pytest -q tests/path/to/test_file.py::test_name
```

**Beta video fixture skip pattern.** Some Beta-pipeline tests are
`skipif`-guarded on a real video fixture and will skip by default — that's
expected. To run them locally, fetch the fixture once:

```bash
uv run python scripts/fetch_beta_video_fixture.py
```

After that the previously-skipped tests will execute on subsequent runs.

## Linting / formatting

```bash
uv run ruff check .
uv run ruff format --check .
```

CI runs both. `ruff check` covers lint rules; `ruff format --check`
verifies formatting. Running just one is **not** enough — please run both
before pushing. If `ruff format --check` complains, apply with
`uv run ruff format .`.

## Commit messages

This repo uses [Conventional Commits](https://www.conventionalcommits.org/).
Prefixes in active use:

- `feat:` — new user-visible feature
- `fix:` — bug fix
- `chore:` — tooling, deps, repo hygiene
- `docs:` — documentation only
- `test:` — tests only
- `refactor:` — code change that doesn't alter behavior

Keep the summary line under ~72 characters. Use the body (after a blank
line) to explain *why* the change was made — the diff already shows
*what*.

## Pull requests

- Push directly to `main` is permitted for the maintainer; everyone else
  please open a PR.
- Open as **draft** while work is in progress; mark **ready for review**
  once CI is green and you've self-reviewed the diff.
- **Squash merge** is the default — keep PRs small and focused so the
  squashed commit message stays meaningful.
- Reference the issue you're closing in the PR description (`Closes #N`).

## CHANGELOG

Any user-visible change (new feature, bug fix, breaking change, CLI
behavior change) needs an entry in `CHANGELOG.md` under the
`## [Unreleased]` section, in the same PR as the change. The file follows
[Keep a Changelog](https://keepachangelog.com/) — group entries under
`Added` / `Changed` / `Fixed` / `Removed` / `Deprecated` / `Security` as
appropriate. Internal-only refactors don't need a changelog entry.

## Reporting bugs / feature requests / questions

- **Bug reports** and **feature requests** — use the GitHub issue
  templates in `.github/ISSUE_TEMPLATE/` (also surfaced when you click
  *New issue* on the repo).
- **Questions / open-ended discussion** — please use the GitHub
  Discussions board rather than filing an issue.
- **Security issues** — see `SECURITY.md`. Do not file a public issue.
