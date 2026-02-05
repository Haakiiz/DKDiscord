# Agents

## Project context
- This repo is a set of scripts for Discord chat export and LLM analysis. There is no running bot service here yet.
- Primary scripts: `Main.py` (Playwright export, unreliable), `JSON_cleaner.py` (trim export), `ask_chat.py` (Q and A), `tips_extraction.py` (summaries), `contextcounter.py` (chunk and summarize), `GeminiTest.py` (Gemini prototype).

## Working guidelines
- Prefer DiscordChatExporter JSON as input. Many scripts expect a top-level `messages` list.
- Outputs go to `outputs/` by default for `ask_chat.py`, or the repo root for `tips_extraction.py` and `contextcounter.py`. Do not change output paths unless asked.
- Avoid committing or hardcoding API keys. Use environment variables instead.
- Avoid editing `venv/` or large data files unless explicitly requested.

## Runbook
- Q and A over an export:
  `python ask_chat.py path\to\export.json --question "Your question"`
- Summarize into reports:
  `python tips_extraction.py path\to\export.json --merge`
