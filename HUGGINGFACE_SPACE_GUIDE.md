# Hugging Face Space Submission Guide (OpenEnv RL Challenge)

This guide is tailored for this repository and aligned with the hackathon validator rules.

## 1) Required Project Layout
- Keep `inference.py` in the repository root.
- Keep `Dockerfile` in the repository root.
- Keep `requirements.txt` in the repository root.

## 2) LLM Client Requirement
- Use `from openai import OpenAI` for all LLM calls.
- Do not use alternate SDKs.
- Do not use direct HTTP calls for model inference.

## 3) Required Environment Variables
Set these as Space Variables/Secrets:

- `API_BASE_URL`
  - Purpose: OpenAI-compatible endpoint
  - Default: `https://api.openai.com/v1` (already in code)
- `MODEL_NAME`
  - Purpose: model id used by inference
  - Default: `gpt-4.1-mini` (already in code)
- `HF_TOKEN`
  - Purpose: API token used by the OpenAI client in this project
  - Required: yes, no default

## 4) Required Output Format
Your container output must follow this exact structure:

- One `[START]` line at the start of the episode
- One `[STEP]` line after every `env.step()`
- One `[END]` line after close, always, even on exceptions

Field rules:
- `reward` and `rewards` must use 2 decimal places
- `done` and `success` must be lowercase: `true` or `false`
- `error` must be raw `last_action_error` or `null`
- Every record must be one single line

## 5) Space Operational Rules (Critical)
Before submission:
- Stop unrelated spaces.
- Keep only your main submission space running.
- Wait until build completes and state is `Running`.
- Verify logs are being produced by `inference.py`.

If Space is `Building` or `Stopped` at submission time, validation can fail.

## 6) Resource Constraints
Your code must run within:
- 2 vCPU
- 8 GB RAM

Tips:
- Keep dependencies minimal.
- Avoid loading large local models in the container.
- Use remote API inference through configured endpoint.

## 7) Suggested Space Setup
1. Create a Docker Space.
2. Push this repository as-is.
3. In Space settings, add:
   - Variable: `API_BASE_URL`
   - Variable: `MODEL_NAME`
   - Secret: `HF_TOKEN`
   - Optional variable: `TASK_NAME` (`task1_easy` default)
   - Optional variable: `BENCHMARK` (`smart-irrigation` default)
   - Optional variable: `SEED` (`42` default)
4. Rebuild Space.
5. Confirm status is `Running`.
6. Check logs contain `[START]`, `[STEP]`, `[END]` lines.

## 8) Fast Pre-Submission Checklist
- [ ] `inference.py` exists at root
- [ ] Uses `OpenAI` client
- [ ] `API_BASE_URL` has default
- [ ] `MODEL_NAME` has default
- [ ] `HF_TOKEN` is required
- [ ] Output format uses `[START]`, `[STEP]`, `[END]`
- [ ] Rewards are 2-decimal formatted
- [ ] Booleans are lowercase
- [ ] Space status is `Running`
- [ ] Only primary space is active

## 9) Resubmission
If validation fails:
1. Fix the issue.
2. Rebuild/restart the same Space.
3. Confirm `Running`.
4. Submit again.

There is no penalty for resubmitting.
