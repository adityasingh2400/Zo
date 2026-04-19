# SHIP — push workflow

One repo, one branch, one remote. No middle stages, no atomic-unit theatre.

## TL;DR

```bash
git add -A
git commit -m "<what you did>"
git push
```

That's it. `git push` lands on `zo/main` because `main` tracks `zo/main`.

## The setup (already done)

- **Remote:** `zo` → https://github.com/adityasingh2400/Zo.git (the only remote)
- **Branch:** `main` (the only branch; tracks `zo/main`)
- **Origin (`chicken` / ychackathon):** removed. Do not re-add.
- **`zo-main` local branch:** removed. There is no `zo-main`, only `main`.

## What NOT to do

- Do **not** add `origin` pointing at `chicken`/`ychackathon` again.
- Do **not** create a `zo-main` branch. The branch is `main`.
- Do **not** push to anything in `/Users/aditya/Desktop/ychackathon/` — that
  directory is dead weight from before the migration. Treat it as read-only
  archive; better, ignore it entirely.
- Do **not** split work into "secret-migration-friendly" atomic commits.
  Migration is over. One commit per logical chunk is fine; one commit per
  whole feature is also fine.

## Commit cadence (hackathon-mode)

- Land working commits to `main` directly. No PRs, no feature branches.
- Tag at every working state per the build plan
  (`v0.1-cartesia-green`, `v0.2-audio-first-green`, …,
  `v0.7-FREEZE`). Push tags with `git push --tags` once they're set.
- If something breaks production: `git revert <bad-sha> && git push`.
  Don't force-push.

## Recovery

If the local clone gets weird, the canonical state is on github:

```bash
cd /Users/aditya/Desktop/Zo
git fetch zo
git reset --hard zo/main      # only do this if local work is throwaway
```

Local backup tag from the migration cleanup: `pre-zo-cleanup` (kept locally,
not pushed). Recover via `git checkout pre-zo-cleanup` if needed.

## Verifying you're pushing to the right place

```bash
git remote -v
# expected:
# zo  https://github.com/adityasingh2400/Zo.git (fetch)
# zo  https://github.com/adityasingh2400/Zo.git (push)

git branch -vv
# expected:
# * main <sha> [zo/main] <last commit msg>
```

If `git remote -v` shows anything other than `zo`, run:

```bash
git remote remove origin   # if origin came back somehow
```
