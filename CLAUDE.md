# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

**DKDiscord** is a small pipeline that mines a **Door Kickers 2: Task Force North**
Discord channel for useful gameplay knowledge (tips, keybinds, loadouts, tactics)
and turns it into readable text files. The game lacks comprehensive in-game hints,
so the community's Discord chat is the de-facto knowledge base.

Flow: **scrape/export Discord messages → clean to JSON → feed to an LLM → extract tips.**

## Files

| File | Purpose |
|------|---------|
| `Main.py` | Scrapes a Discord channel via Playwright (browser automation) into `discord_channel.json`. Per the in-code note, Discord's anti-automation blocks this — prefer **DiscordChatExporter** to produce the raw JSON export. |
| `JSON_cleaner.py` | Reduces a DiscordChatExporter JSON export to just `id` / `timestamp` / `username` / `content` → `cleaned_discord.json`. |
| `GeminiTest.py` | **Gemini path.** Chunks cleaned messages and prompts Gemini to extract strict, specific gameplay tips as JSON → `doorkickers2_tips.txt`. |
| `contextcounter.py` | **OpenAI path.** Token-aware chunking (`tiktoken`) + `gpt-4o` for categorized tip extraction. |
| `tips_extraction.py` | **OpenAI path.** Similar to above using `gpt-5-nano`, produces an executive summary + categorized topics. |
| `discord_channel.json` | Output of `Main.py` (currently empty). |

## Models in use

- **Gemini** (`GeminiTest.py`): `gemini-3.5-flash` — the newest Flash model
  (Google I/O, May 2026), set via `MODEL_NAME`. Uses the `google.generativeai` SDK.
- **OpenAI** (`contextcounter.py`, `tips_extraction.py`): `gpt-4o` / `gpt-5-nano`.

When upgrading the Gemini model, change `MODEL_NAME` in `GeminiTest.py` and
re-check `RATE_LIMIT_DELAY_SECONDS` against your account's RPM (delay ≈ 60 / RPM).

## Setup & running

There is no dependency manifest yet. The scripts expect:

```bash
pip install google-generativeai openai tiktoken tqdm playwright playwright-stealth
```

- **Gemini extraction:** set your API key in `GeminiTest.py` (`API_KEY`), point
  `INPUT_JSON_FILE` at a cleaned JSON file, then `python GeminiTest.py`.
- **OpenAI extraction:** set `OPENAI_API_KEY` in your environment, then run
  `python contextcounter.py` or `python tips_extraction.py` (adjust the input
  filename inside `main()`).

## Conventions & gotchas

- **API keys:** `GeminiTest.py` hardcodes `API_KEY` (placeholder `"INSERT API KEY"`).
  Prefer an environment variable (`os.environ.get("GEMINI_API_KEY")`); never commit
  a real key. OpenAI scripts already read the key from the environment.
- The LLM prompts deliberately demand *specific, generally-applicable* tips and
  reject vague/off-topic chatter. Preserve that strictness when editing prompts.
- Input JSON is expected as `{"messages": [...]}` with `timestamp`/`username`/`content`.
- Some comments/prints are in Norwegian — that's expected, not a bug.
