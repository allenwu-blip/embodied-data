# Release checklist — tests-green → PyPI live

Seven steps. Each step lists the command and what to verify before moving on.
Steps marked **[PUBLISH]** require Allen's explicit approval — do NOT execute
them on his behalf.

1. **Bump version + commit**
   `vim pyproject.toml` (set `version = "0.1.0"`) → `git commit -am "chore: release v0.1.0"`
   Verify: `grep '^version' pyproject.toml` shows the new version; working tree clean.

2. **Push, wait for CI green**
   `git push origin main`
   Verify: GitHub Actions `CI` run on the head commit is green (lint + format + 50 tests + version-drift step prints `tag=<none> match=true` since no tag yet).

3. **Build + twine check locally**
   `uv build && uv run --with twine python -m twine check dist/*`
   Verify: `dist/` contains exactly one wheel and one sdist for the new version; both report `PASSED`.

4. **Tag locally — DO NOT PUSH**
   `git tag -a v0.1.0 -m "v0.1.0"`
   Verify: `git tag -l 'v*'` lists `v0.1.0`; `python scripts/check_version.py` prints `match=true`.

5. **[PUBLISH] Push tag** (after Allen approves)
   `git push origin v0.1.0`
   Verify: tag visible at `https://github.com/allenwu-blip/embodied-data/releases/tag/v0.1.0` (auto-created as draft) and CI re-runs on the tag ref.

6. **CI re-runs on tag — verify drift guard**
   No command. Just watch the tag-triggered CI run.
   Verify: the `Version-tag drift check` step ran in `--strict` mode and printed `version=0.1.0 tag=v0.1.0 match=true`. Release notes draft from `docs/release-v0.1.0.md` is ready to paste.

7. **[PUBLISH] Upload to PyPI** (after Allen approves)
   `uv run --with twine python -m twine upload dist/*` (or `gh release create v0.1.0 --notes-file docs/release-v0.1.0.md dist/*` to publish the GitHub release in the same shot)
   Verify: `pip install embodied-data==0.1.0` from a clean venv resolves; `embodied-data --help` runs.

## Rollback

If a bad release reaches PyPI, do **not** delete — yank instead:
`uv run --with twine python -m twine yank embodied-data==0.1.0 --reason "<reason>"`
Then bump to `0.1.1` and restart at step 1.
