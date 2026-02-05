# DnDDiscordBot

Utilities for exporting and analyzing Discord chat logs, with a focus on game-related research and summaries. The repo is script-first; there is no standalone bot runtime here yet.

## Quick start
1. Create and activate a virtual environment.
2. Install deps:
   `pip install openai tiktoken tqdm playwright playwright-stealth google-generativeai`
3. Set `OPENAI_API_KEY` for OpenAI-based scripts.
4. If you plan to use Playwright, run `playwright install` once.

## Scripts
- `Main.py`: Experimental Playwright scraper that scrolls a Discord channel and saves `discord_channel.json`. It is likely blocked by automation detection. Prefer DiscordChatExporter.
- `JSON_cleaner.py`: Trims a DiscordChatExporter JSON into `cleaned_discord.json` with id, timestamp, username, and content.
- `ask_chat.py`: Asks a question over a chat export and writes a markdown report to `outputs/`.
- `tips_extraction.py`: Summarizes a chat export into chunk reports (and optional merged report). Supports `--provider openai|grok`. Writes `dk2_chunk_summaries_YYYY-MM-DD.txt` and `dk2_final_report_YYYY-MM-DD.txt`.
- `contextcounter.py`: Chunks a cleaned export and writes `llm_outputs_YYYY-MM-DD.txt`.
- `GeminiTest.py`: Gemini-based extractor. Requires `GEMINI_API_KEY`.
- `llm_choice.py`: Small CLI that prompts for Gemini or Grok and runs a single prompt.

## Data flow
1. Export a channel with DiscordChatExporter.
2. (Optional) Run `JSON_cleaner.py` to produce `cleaned_discord.json`.
3. Run `ask_chat.py` or `tips_extraction.py` to analyze.
4. Results land in `outputs/` or the repo root, depending on script flags.

## Environment variables
- `OPENAI_API_KEY`: Required for OpenAI scripts.
- `GEMINI_API_KEY`: Required for `GeminiTest.py` and Gemini runs in `llm_choice.py`.
- `GEMINI_MODEL`: Optional override for Gemini in `llm_choice.py`.
- `XAI_API_KEY`: Required for Grok in `llm_choice.py` and `tips_extraction.py` (when using `--provider grok`).
- `ASK_MODEL`, `ASK_MERGE_MODEL`, `ASK_REFINE_MODEL`, `ASK_CHUNK_TOKEN_LIMIT`, `ASK_SAFETY_TOKENS`, `ASK_MAX_OUTPUT_TOKENS`, `ASK_RATE_DELAY`.
- `DK2_PROVIDER`, `DK2_MODEL`, `DK2_GROK_MODEL`, `DK2_MAX_WORKERS`, `DK2_PREFER_API`.

## Notes
- Chat exports may contain sensitive data. Keep them out of git.
- Large exports can be slow; consider using `--rate-delay` in `ask_chat.py`.
