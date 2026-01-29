# Push to GitHub Without Secrets in History

GitHub blocked your push because **Hugging Face tokens** were committed in the past (in notebooks and `upload_model_to_hub.py`). Those commits are still in your history, so the push is rejected.

Do these two things:

---

## 1. Revoke the exposed token (important)

The token that was committed should be treated as **compromised**.

1. Go to https://huggingface.co/settings/tokens  
2. **Revoke** the token that was in the repo  
3. Create a **new** token and use it only via environment variable: `HF_TOKEN` (e.g. in `.env` or your shell). Never commit it.

---

## 2. Push with a fresh history (one commit, no secrets)

This creates a new `main` with **only the current files** (one commit). Old commits (and the secrets) are no longer in the branch you push.

Run in your repo root (e.g. `whisper-finetune`):

```bash
# Create a new branch with no history (orphan)
git checkout --orphan temp-main

# Stage all current files
git add -A

# One commit with current state (no secrets in this tree)
git commit -m "Initial commit: whisper-finetune (clean history)"

# Replace main with this branch
git branch -D main
git branch -m main

# Push to your new remote (force, because history changed)
git push -u origin main --force
```

After this, `origin/main` will have a single commit. GitHub will not see any secrets in history.

**Note:** Anyone who had cloned the old repo will have a different history; they can re-clone or `git fetch origin && git reset --hard origin/main`.

---

## 3. Use HF token via env only

The script `src/whisper_finetune/scripts/upload_model_to_hub.py` now uses:

- `HF_TOKEN` from the environment (e.g. `export HF_TOKEN=hf_...` or in `.env`).

Do not put tokens in notebooks or configs. Use `.env` (and add `.env` to `.gitignore`) or your shell profile.
