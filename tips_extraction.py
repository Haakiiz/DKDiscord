import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import tiktoken
from tqdm import tqdm
from openai import OpenAI

# =========================
# Config / Defaults
# =========================
DEFAULT_MODEL = os.getenv("DK2_MODEL", "gpt-4o-mini")  # safer default; override with --model or env
DEFAULT_GROK_MODEL = os.getenv("DK2_GROK_MODEL", "grok-4-1-fast-non-reasoning")
DEFAULT_PROVIDER = os.getenv("DK2_PROVIDER", "openai")
ENCODING_NAME = "o200k_base"
CONTEXT_WINDOW = 128_000
GROK_CONTEXT_WINDOW = 2_000_000
MAX_OUTPUT_TOKENS = 2048
SAFETY_MARGIN_TOKENS = 800
MAX_WORKERS = int(os.getenv("DK2_MAX_WORKERS", "6"))
CHUNK_MAX_CHARS = 240_000
RETRY_MAX = 5

# =========================
# Prompts
# =========================
BASE_SYSTEM_PROMPT = "You are a precise, structured research analyst."

BASE_TASK_PROMPT = (
    "Developer: You have access to the JSON export of messages from a Discord channel focused on the game \"Door Kickers 2: Task Force North.\"\n\n"
    "Begin with a concise checklist (3–7 bullets) outlining your approach to analyzing and summarizing the chat data.\n\n"
    "Please analyze the chat data and generate the following, presented in clear natural language with distinct headers:\n\n"
    "# Executive Summary\n"
    "Provide a concise overview of recent activity, overarching themes, and the general tone of the conversations.\n\n"
    "# Categorized Discussion Topics\n"
    "Summarize significant discussion topics, each under its own subheader. For each, include:\n"
    "- Main topic or discussion theme\n"
    "- Critical points or highlights relating to this topic\n"
    "- Number of unique users who participated\n"
    "- Excerpts from notable messages (with author and timestamp)\n\n"
    "Requested categories include:\n"
    "- Game updates\n"
    "- Notable in-game strategies or tips\n"
    "- Reveals or statements from developers\n"
    "- News or insights about the game's development\n"
    "- Major or extended discussion threads\n\n"
    "CRITICAL: Never output the sentence 'No relevant discussions found in the provided chat log.' If a slice is thin or noisy, still write an Executive Summary describing activity level and themes. For each requested category, either summarize what you find OR write 'Not observed in this slice.'\n"
)

MERGE_PROMPT = (
    "You will receive several partial reports that each summarize a different chunk of the same Discord chat.\n"
    "Combine them into ONE coherent final report with the SAME sections and rules as the original task.\n"
    "De-duplicate overlapping points, reconcile inconsistent counts, and keep excerpts representative but brief.\n"
    "For each requested category, either merge findings across chunks or write 'Not observed across all chunks.' if never seen.\n"
    "Always include an Executive Summary describing overall activity and themes across all chunks.\n"
)

# =========================
# Tokenizer helpers
# =========================
encoder = tiktoken.get_encoding(ENCODING_NAME)

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])

# =========================
# JSON → enriched text line
# =========================

def pick_username(msg: Dict[str, Any]) -> str:
    if msg.get("username"):
        return str(msg["username"])  # some exporters
    author = msg.get("author") or {}
    return author.get("nickname") or author.get("name") or author.get("id") or "N/A"


def summarize_mentions(msg: Dict[str, Any]) -> str:
    items = []
    for m in msg.get("mentions") or []:
        items.append(m.get("nickname") or m.get("name") or m.get("id") or "?")
    return ", ".join(items)


def summarize_reactions(msg: Dict[str, Any]) -> str:
    parts = []
    for r in (msg.get("reactions") or [])[:10]:  # cap to avoid bloat
        em = r.get("emoji") or {}
        nm = em.get("name") or em.get("code") or em.get("id") or "?"
        parts.append(f"{nm}:{r.get('count', 0)}")
    return ", ".join(parts)


def summarize_attachments(msg: Dict[str, Any]) -> str:
    atts = msg.get("attachments") or []
    if not atts:
        return ""
    names = []
    for a in atts[:3]:
        nm = a.get("fileName") or a.get("url") or "attachment"
        names.append(nm.split("/")[-1])
    rest = len(atts) - len(names)
    base = ", ".join(names)
    if rest > 0:
        base += f" (+{rest})"
    return base


def summarize_embeds(msg: Dict[str, Any]) -> str:
    emb = msg.get("embeds") or []
    if not emb:
        return ""
    titles = []
    for e in emb[:3]:
        t = e.get("title") or e.get("provider") or e.get("url") or "embed"
        titles.append(str(t)[:80])
    rest = len(emb) - len(titles)
    base = ", ".join(titles)
    if rest > 0:
        base += f" (+{rest})"
    return base


def summarize_reference(msg: Dict[str, Any]) -> str:
    ref = msg.get("reference")
    if not ref:
        return ""
    return ref.get("messageId") or "?"


def build_line_from_message(msg: Dict[str, Any]) -> str:
    timestamp = msg.get("timestamp", "N/A")
    username = pick_username(msg)
    mtype = msg.get("type") or "Default"
    content = (msg.get("content") or "").replace("\n", " ").strip()

    # Build extras
    extras = []
    ref = summarize_reference(msg)
    if ref:
        extras.append(f"reply_to[{ref}]")
    men = summarize_mentions(msg)
    if men:
        extras.append(f"mentions[{men}]")
    att = summarize_attachments(msg)
    if att:
        extras.append(f"attachments[{att}]")
    emb = summarize_embeds(msg)
    if emb:
        extras.append(f"embeds[{emb}]")
    reacts = summarize_reactions(msg)
    if reacts:
        extras.append(f"reactions[{reacts}]")
    if msg.get("isPinned"):
        extras.append("pinned")
    if mtype:
        extras.append(f"type[{mtype}]")

    extra_str = (" | " + " | ".join(extras)) if extras else ""
    if not content:
        content = "(no-text)"
    return f"{timestamp} - {username}: {content}{extra_str}"

# =========================
# Chunking
# =========================

def compute_chunk_budget(context_window: int) -> int:
    reserved = count_tokens(BASE_TASK_PROMPT) + count_tokens(BASE_SYSTEM_PROMPT) + SAFETY_MARGIN_TOKENS + MAX_OUTPUT_TOKENS
    return max(1024, context_window - reserved)


def chunk_lines(lines: List[str], context_window: int) -> List[List[str]]:
    chunk_token_limit = compute_chunk_budget(context_window)
    chunks: List[List[str]] = []
    cur: List[str] = []
    tks = 0
    chars = 0

    for line in tqdm(lines, desc="Chunking", unit="ln"):
        tok = count_tokens(line)
        ch = len(line)
        if tok > chunk_token_limit:
            keep = max(256, chunk_token_limit - 128)
            line = truncate_by_tokens(line, keep)
            tok = count_tokens(line)
            ch = len(line)
        if (tks + tok > chunk_token_limit or chars + ch > CHUNK_MAX_CHARS) and cur:
            chunks.append(cur)
            cur = []
            tks = 0
            chars = 0
        cur.append(line)
        tks += tok
        chars += ch

    if cur:
        chunks.append(cur)

    logging.info(f"Chunks created: {len(chunks)} (budget ~{chunk_token_limit} tokens per chunk)")
    return chunks


def build_user_content_from_chunk(chunk_lines: List[str]) -> str:
    first_ts = chunk_lines[0].split(" ", 1)[0] if chunk_lines else "N/A"
    last_ts = chunk_lines[-1].split(" ", 1)[0] if chunk_lines else "N/A"
    header = (
        f"Chunk span: {first_ts} → {last_ts}\n"
        f"Messages: {len(chunk_lines)}\n\n"
    )
    body = "\n".join(f"{i+1}. {line}" for i, line in enumerate(chunk_lines))
    return header + body

# =========================
# OpenAI calls with robust fallback
# =========================
client = OpenAI()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _responses_api(prompt: str, model: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return (getattr(resp, "output_text", "") or "").strip()


def _chat_api(user_content_1: str, user_content_2: str, model: str) -> str:
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content_1},
            {"role": "user", "content": user_content_2},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return (chat.choices[0].message.content or "").strip()


def call_llm(
    prompt_for_responses: str,
    user_content_for_chat: Tuple[str, str],
    model: str,
    prefer: str,
    allow_chat: bool,
) -> str:
    last_err = None

    order = [prefer]
    if allow_chat:
        order = [prefer, "chat" if prefer == "responses" else "responses"]
    for api in order:
        for attempt in range(RETRY_MAX):
            try:
                if api == "responses":
                    out = _responses_api(prompt_for_responses, model)
                else:
                    out = _chat_api(user_content_for_chat[0], user_content_for_chat[1], model)
                if out:
                    return out
                else:
                    logging.warning(f"{api} returned empty text (attempt {attempt+1}/{RETRY_MAX}).")
                    # fall through to retry
            except Exception as e:
                last_err = e
                if attempt < RETRY_MAX - 1:
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    logging.warning(f"{api} error: {e}. Retrying in {delay:.1f}s…")
                    time.sleep(delay)
                else:
                    logging.error(f"{api} failed after {RETRY_MAX} attempts: {e}")
                    break
    return f"(model returned no text; last error: {last_err})"

# =========================
# Summarize + Merge
# =========================

def summarize_chunk(
    chunk_lines: List[str],
    chunk_index: int,
    model: str,
    prefer_api: str,
    allow_chat: bool,
) -> Tuple[int, str]:
    user_slice = build_user_content_from_chunk(chunk_lines)
    prompt = BASE_TASK_PROMPT + "\n\n=== CHAT SLICE ===\n" + user_slice
    logging.info(f"Summarizing chunk {chunk_index} — chars={len(prompt):,} tokens≈{count_tokens(prompt):,}")

    out = call_llm(
        prompt_for_responses=prompt,
        user_content_for_chat=(BASE_TASK_PROMPT, "=== CHAT SLICE ===\n" + user_slice),
        model=model,
        prefer=prefer_api,
        allow_chat=allow_chat,
    )
    return chunk_index, out


def merge_summaries(partials: List[str], model: str, prefer_api: str, allow_chat: bool, context_window: int) -> str:
    body = "\n\n".join([f"=== Partial Report {i+1} ===\n{p}" for i, p in enumerate(partials)])
    budget = context_window - (count_tokens(MERGE_PROMPT) + count_tokens(BASE_SYSTEM_PROMPT) + SAFETY_MARGIN_TOKENS + MAX_OUTPUT_TOKENS)
    body = truncate_by_tokens(body, max(1024, budget))

    prompt = MERGE_PROMPT + "\n\n" + body
    logging.info(f"Merging {len(partials)} partials — chars={len(prompt):,} tokens≈{count_tokens(prompt):,}")

    out = call_llm(
        prompt_for_responses=prompt,
        user_content_for_chat=(MERGE_PROMPT, body),
        model=model,
        prefer=prefer_api,
        allow_chat=allow_chat,
    )
    return out

# =========================
# IO / Main
# =========================

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
    parser = argparse.ArgumentParser(description="Summarize a Discord channel export with robust OpenAI fallbacks (TXT outputs).")
    parser.add_argument("input", help="Path to JSON export with a top-level 'messages' array.")
    parser.add_argument("--provider", choices=["openai", "grok"], default=DEFAULT_PROVIDER, help=f"LLM provider (default {DEFAULT_PROVIDER})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default {DEFAULT_MODEL})")
    parser.add_argument("--grok-model", default=DEFAULT_GROK_MODEL, help=f"Grok model (default {DEFAULT_GROK_MODEL})")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Concurrent chunk calls (default %(default)s).")
    parser.add_argument("--merge", action="store_true", help="Also merge chunk summaries into one final report.")
    parser.add_argument("--out-dir", default=".", help="Output directory (default current).")
    parser.add_argument("--prefer-api", choices=["responses", "chat"], default=os.getenv("DK2_PREFER_API", "chat"), help="Primary API to use; we will auto-fallback to the other one if it returns empty.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(Path(".env"))

    provider = args.provider
    allow_chat = provider == "openai"
    if provider == "grok" and args.prefer_api != "responses":
        logging.warning("Grok provider forces prefer-api=responses.")
        args.prefer_api = "responses"
    if provider == "grok":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise SystemExit("XAI_API_KEY is not set.")
        global client
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        model = args.grok_model
        context_window = GROK_CONTEXT_WINDOW
    else:
        model = args.model
        context_window = CONTEXT_WINDOW

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = load_messages(in_path)
    logging.info(f"Loaded {len(messages)} messages from {in_path.name}")

    # Build enriched lines for ALL messages (never skip)
    lines: List[str] = [build_line_from_message(m) for m in messages]

    # Chunk
    chunks = chunk_lines(lines, context_window)

    results: List[str] = ["" for _ in range(len(chunks))]
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(summarize_chunk, chunk, i + 1, model, args.prefer_api, allow_chat): i
            for i, chunk in enumerate(chunks)
        }
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="LLM Calls", unit="chunk"):
            idx = futs[fut]
            try:
                chunk_index, text = fut.result()
                results[chunk_index - 1] = text or "(empty)"
                logging.info(f"Chunk {chunk_index} done.")
            except Exception as e:
                logging.exception(f"Error processing chunk {idx+1}: {e}")
                results[idx] = f"Error: {e}"

    today = dt.datetime.now().strftime("%Y-%m-%d")
    chunks_out = out_dir / f"dk2_chunk_summaries_{today}.txt"
    write_file(chunks_out, "\n\n".join([f"=== Chunk {i+1} ===\n{t}" for i, t in enumerate(results)]))

    if args.merge:
        final_report = merge_summaries(
            results,
            model=model,
            prefer_api=args.prefer_api,
            allow_chat=allow_chat,
            context_window=context_window,
        )
        final_out = out_dir / f"dk2_final_report_{today}.txt"
        write_file(final_out, final_report)

    logging.info("All done.")


if __name__ == "__main__":
    main()
