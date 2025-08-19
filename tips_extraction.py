import argparse
import concurrent.futures
import datetime as dt
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import tiktoken
from tqdm import tqdm
from openai import OpenAI

# -------------------------
# Config / Defaults
# -------------------------
DEFAULT_MODEL = "gpt-5-mini"         # higher quality than nano for summarization
ENCODING_NAME = "o200k_base"         # works for 4o/5-series models with 200k ctx
CONTEXT_WINDOW = 200_000             # token context of the model (adjust if needed)
MAX_OUTPUT_TOKENS = 4_096            # how much we want back per chunk
SAFETY_MARGIN_TOKENS = 1_000         # buffer for formatting/misc
MAX_WORKERS = 6                      # keep sensible to avoid rate limits
CHUNK_MAX_CHARS = 240_000            # secondary guard
RETRY_MAX = 5

# -------------------------
# Prompt(s)
# -------------------------
BASE_SYSTEM_PROMPT = """You are a precise, structured research analyst."""

BASE_TASK_PROMPT = """Developer: You have access to the JSON export of messages from a Discord channel focused on the game "Door Kickers 2: Task Force North."

Begin with a concise checklist (3–7 bullets) outlining your approach to analyzing and summarizing the chat data.

Please analyze the chat data and generate the following, presented in clear natural language with distinct headers:

# Executive Summary
Provide a concise overview of recent activity, overarching themes, and the general tone of the conversations.

# Categorized Discussion Topics
Summarize significant discussion topics, each under its own subheader. For each, include:
- Main topic or discussion theme
- Critical points or highlights relating to this topic
- Number of unique users who participated
- Excerpts from notable messages (with author and timestamp)

Requested categories include:
- Game updates
- Notable in-game strategies or tips
- Reveals or statements from developers
- News or insights about the game's development
- Major or extended discussion threads

If you do not find relevant topics or discussions in the chat data, state under Executive Summary: "No relevant discussions found in the provided chat log." and omit the Categorized Discussion Topics section.
"""

MERGE_PROMPT = """You will receive several partial reports that each summarize a different chunk of the same Discord chat.
Combine them into ONE coherent final report with the SAME sections and rules as the original task.
De-duplicate overlapping points, reconcile inconsistent counts, and keep excerpts representative but brief.
If across all chunks there are no relevant discussions, output only:
"Executive Summary: No relevant discussions found in the provided chat log."
"""

# -------------------------
# Tokenizer
# -------------------------
encoder = tiktoken.get_encoding(ENCODING_NAME)

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])

# -------------------------
# Chunking
# -------------------------
def compute_chunk_budget() -> int:
    # Leave space for: system + instructions + output + safety.
    reserved = count_tokens(BASE_TASK_PROMPT) + SAFETY_MARGIN_TOKENS + MAX_OUTPUT_TOKENS
    return max(1024, CONTEXT_WINDOW - reserved)

def chunk_messages_by_tokens(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    chunk_token_limit = compute_chunk_budget()
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    tks = 0
    chars = 0

    for msg in tqdm(messages, desc="Chunking messages", unit="msg"):
        # CLEAN BUG: Only strip & replace newlines. (Your original replace('', ' ') exploded the text.)
        content = (msg.get("content", "") or "").strip().replace("\n", " ")
        line = f"{msg.get('timestamp','N/A')} - {msg.get('username','N/A')}: {content}"
        tok = count_tokens(line)
        ch = len(line)

        # If a single message is massive, trim it so it still fits (rare but safe).
        if tok > chunk_token_limit:
            # Keep some headroom for numbering/formatting in the prompt
            line = truncate_by_tokens(line, chunk_token_limit - 128)
            tok = count_tokens(line)
            ch = len(line)

        # If adding this message exceeds budget, start a new chunk
        if (tks + tok > chunk_token_limit or chars + ch > CHUNK_MAX_CHARS) and cur:
            chunks.append(cur)
            cur = []
            tks = 0
            chars = 0

        cur.append({
            "timestamp": msg.get("timestamp", "N/A"),
            "username": msg.get("username", "N/A"),
            "content": content
        })
        tks += tok
        chars += ch

    if cur:
        chunks.append(cur)

    logging.info(f"Chunks created: {len(chunks)} (budget ~{chunk_token_limit} tokens per chunk)")
    return chunks

def build_user_content_from_chunk(chunk: List[Dict[str, Any]]) -> str:
    # Keep it plain text for the "user" message
    lines = []
    for idx, msg in enumerate(chunk, start=1):
        lines.append(f"{idx}. {msg['timestamp']} - {msg['username']}: {msg['content']}")
    return "\n".join(lines)

# -------------------------
# LLM Calls
# -------------------------
client = OpenAI()

def call_llm(input_messages: List[Dict[str, str]], model: str) -> str:
    for attempt in range(RETRY_MAX):
        try:
            resp = client.responses.create(
                model=model,
                input=input_messages,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2,
            )
            return getattr(resp, "output_text", "").strip()
        except Exception as e:
            # crude but robust backoff with jitter
            if attempt < RETRY_MAX - 1:
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                logging.warning(f"LLM error ({e}). Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                logging.error(f"Failed after {RETRY_MAX} attempts: {e}")
                return f"Error: {e}"

def summarize_chunk(chunk: List[Dict[str, Any]], chunk_index: int, model: str) -> Tuple[int, str]:
    user_content = build_user_content_from_chunk(chunk)
    input_messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": BASE_TASK_PROMPT.strip()},
        {"role": "user", "content": user_content}
    ]
    logging.info(f"Summarizing chunk {chunk_index} ({len(chunk)} messages)...")
    out = call_llm(input_messages, model=model)
    return chunk_index, out

def merge_summaries(partials: List[str], model: str) -> str:
    # If it would be too long, merge in multiple passes (rare). Simple single-pass first:
    body = "\n\n".join([f"=== Partial Report {i+1} ===\n{p}" for i, p in enumerate(partials)])
    # If it’s still too long, trim the body to fit the budget
    budget = CONTEXT_WINDOW - (count_tokens(MERGE_PROMPT) + SAFETY_MARGIN_TOKENS + MAX_OUTPUT_TOKENS)
    body = truncate_by_tokens(body, max(1024, budget))

    input_messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": MERGE_PROMPT.strip()},
        {"role": "user", "content": body}
    ]
    logging.info("Merging partial reports into final report...")
    return call_llm(input_messages, model=model)

# -------------------------
# IO / Main
# -------------------------
def load_messages(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected 'messages' to be a list in the JSON file.")
    return messages

def write_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    logging.info(f"Wrote: {path}")

def main():
    parser = argparse.ArgumentParser(description="Summarize a Discord channel export with an LLM.")
    parser.add_argument("input", help="Path to JSON export with a top-level 'messages' array.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default {DEFAULT_MODEL})")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Concurrent chunk calls.")
    parser.add_argument("--merge", action="store_true", help="Also merge chunk summaries into one final report.")
    parser.add_argument("--out-dir", default=".", help="Output directory (default current).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = load_messages(in_path)
    logging.info(f"Loaded {len(messages)} messages from {in_path.name}")

    chunks = chunk_messages_by_tokens(messages)
    logging.info(f"Created {len(chunks)} chunks.")

    # Summarize chunks
    results: List[str] = ["" for _ in range(len(chunks))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(summarize_chunk, chunk, i+1, args.model): i
            for i, chunk in enumerate(chunks)
        }
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="LLM Calls", unit="chunk"):
            idx = futs[fut]
            try:
                chunk_index, text = fut.result()
                results[chunk_index - 1] = text or ""
                logging.info(f"Chunk {chunk_index} done.")
            except Exception as e:
                logging.exception(f"Error processing chunk {idx+1}: {e}")
                results[idx] = f"Error: {e}"

    today = dt.datetime.now().strftime("%Y-%m-%d")
    chunks_out = out_dir / f"dk2_chunk_summaries_{today}.md"
    write_file(chunks_out, "\n\n".join([f"## Chunk {i+1}\n\n{t}" for i, t in enumerate(results)]))

    if args.merge:
        final_report = merge_summaries(results, model=args.model)
        final_out = out_dir / f"dk2_final_report_{today}.md"
        write_file(final_out, final_report)

    logging.info("All done.")

if __name__ == "__main__":
    main()
