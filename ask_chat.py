import argparse
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
from openai import OpenAI

DEFAULT_MODEL = os.getenv("ASK_MODEL", "gpt-4o-mini")
DEFAULT_MERGE_MODEL = os.getenv("ASK_MERGE_MODEL", DEFAULT_MODEL)
DEFAULT_REFINE_MODEL = os.getenv("ASK_REFINE_MODEL", DEFAULT_MODEL)

ENCODING = tiktoken.get_encoding("o200k_base")
CHUNK_LIMIT = int(os.getenv("ASK_CHUNK_TOKEN_LIMIT", "46000"))
SAFETY_MARGIN = int(os.getenv("ASK_SAFETY_TOKENS", "8000"))
MAX_OUTPUT_TOKENS = int(os.getenv("ASK_MAX_OUTPUT_TOKENS", "2000"))

client = OpenAI()

def extract_output_text(response: Any) -> str:
    """Return plain text from an OpenAI responses API result if available."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []

    output = getattr(response, "output", None)
    if output:
        for item in output:
            item_content = getattr(item, "content", None)
            if item_content is None and isinstance(item, dict):
                item_content = item.get("content")
            if not item_content:
                continue
            for part in item_content:
                part_type = getattr(part, "type", None)
                if part_type is None and isinstance(part, dict):
                    part_type = part.get("type")
                if part_type in {"output_text", "text"}:
                    part_text = getattr(part, "text", None)
                    if part_text is None and isinstance(part, dict):
                        part_text = part.get("text")
                    if isinstance(part_text, str):
                        chunks.append(part_text)

    if not chunks and hasattr(response, "choices"):
        for choice in getattr(response, "choices", []):
            message = getattr(choice, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for row in content:
                        if isinstance(row, dict) and row.get("type") == "text" and row.get("text"):
                            chunks.append(row["text"])
            text_attr = getattr(choice, "text", None)
            if isinstance(text_attr, str):
                chunks.append(text_attr)

    return "".join(chunks).strip()


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def load_messages(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    raise ValueError("Input JSON must be a list or contain a 'messages' list.")


def extract_text(msg: Dict[str, Any]) -> str:
    ts = str(
        msg.get("timestamp")
        or msg.get("createdAt")
        or msg.get("id")
        or "N/A"
    )
    username = msg.get("username")
    if not username and isinstance(msg.get("author"), dict):
        author = msg["author"]
        username = (
            author.get("nickname")
            or author.get("name")
            or author.get("id")
        )
    username = str(username or "Unknown")
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    content = str(content).replace("\n", " ").strip()

    extras = []
    reactions = msg.get("reactions")
    if reactions:
        parts = []
        for reaction in reactions[:5]:
            emoji = reaction.get("emoji") or {}
            label = (
                emoji.get("name")
                or emoji.get("id")
                or "?"
            )
            count = reaction.get("count")
            parts.append(f"{label}:{count}")
        extras.append(f"Reactions={', '.join(parts)}")
    attachments = msg.get("attachments")
    if attachments:
        names = []
        for attachment in attachments[:3]:
            name = (
                attachment.get("fileName")
                or attachment.get("filename")
                or attachment.get("url")
            )
            if name:
                names.append(name.split("/")[-1])
        if names:
            extras.append(f"Attachments={', '.join(names)}")

    trailer = f" ({'; '.join(extras)})" if extras else ""
    return f"[{ts}] {username}: {content}{trailer}".strip()


def chunk_messages(messages: List[Dict[str, Any]]) -> List[List[str]]:
    chunks: List[List[str]] = []
    current: List[str] = []
    token_budget = max(1024, CHUNK_LIMIT - SAFETY_MARGIN)
    tokens = 0

    for msg in messages:
        line = extract_text(msg)
        line_tokens = count_tokens(line) + 4
        if current and tokens + line_tokens > token_budget:
            chunks.append(current)
            current = []
            tokens = 0
        current.append(line)
        tokens += line_tokens

    if current:
        chunks.append(current)
    return chunks


def call_model(prompt: str, model: str) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    text_out = extract_output_text(response)
    if text_out:
        return text_out
    raise RuntimeError(f"Model {model} returned no text output.")


def refine_question(question: str, model: str) -> str:
    prompt = (
        "You help researchers query large Discord chat exports. "
        "Improve the user's question so it is precise, scoped to the dataset, and answerable from chat history. "
        "Return only the refined question without commentary. "
        f"Original question: {question}"
    )
    try:
        refined = call_model(prompt, model)
    except Exception as exc:
        print(f"Warning: question refinement failed ({exc}). Using original question.")
        return question
    return refined or question


def answer_chunk(question: str, chunk: List[str], model: str) -> str:
    chat_excerpt = "\n".join(chunk)
    prompt = (
        "You receive a slice of Discord chat history. "
        "Answer the research question using only this slice. "
        "Quote relevant messages with timestamps and usernames when available. "
        "If nothing is relevant, reply with 'No relevant information in this slice.'\n\n"
        f"Question: {question}\n\n"
        f"Chat slice:\n{chat_excerpt}"
    )
    try:
        return call_model(prompt, model)
    except Exception as exc:
        return f"Error calling model: {exc}"


def merge_answers(question: str, partials: List[str], model: str) -> str:
    if not partials:
        return "No analysis was produced."
    if len(partials) == 1:
        return partials[0]
    body = "\n\n".join(
        f"=== Slice {idx + 1} ===\n{text}" for idx, text in enumerate(partials)
    )
    prompt = (
        "You will combine per-slice answers into a single coherent answer. "
        "Remove duplicates, reconcile conflicts, and keep citations to timestamps and usernames when provided. "
        "If all slices reported no relevant information, state that clearly.\n\n"
        f"Research question: {question}\n\n"
        f"Slice answers:\n{body}"
    )
    try:
        return call_model(prompt, model)
    except Exception as exc:
        return f"Error merging slice answers: {exc}\n\n{body}"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "question"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask adaptable questions over a Discord chat export using an LLM."
    )
    parser.add_argument("input", help="Path to chat log JSON.")
    parser.add_argument("--question", help="Initial question to ask.")
    parser.add_argument(
        "--refine",
        dest="refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable question refinement (default: on).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for answering slices (default {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--merge-model",
        default=DEFAULT_MERGE_MODEL,
        help="Model for merging slice answers.",
    )
    parser.add_argument(
        "--refine-model",
        default=DEFAULT_REFINE_MODEL,
        help="Model used for question refinement.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for saving responses.",
    )
    parser.add_argument(
        "--rate-delay",
        type=float,
        default=float(os.getenv("ASK_RATE_DELAY", "0")),
        help="Seconds to sleep between slice calls.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find {in_path}")

    question = args.question or input("Enter your baseline question: ").strip()
    if not question:
        raise ValueError("A question is required to continue.")

    if args.refine:
        print("Refining question...")
        question = refine_question(question, args.refine_model)
        print(f"Refined question: {question}")

    messages = load_messages(in_path)
    print(f"Loaded {len(messages)} messages.")

    chunks = chunk_messages(messages)
    print(f"Created {len(chunks)} chunk(s).")

    partials: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Analyzing chunk {idx}/{len(chunks)}...")
        partial = answer_chunk(question, chunk, args.model)
        partials.append(partial or "No response from model.")
        if args.rate_delay > 0:
            time.sleep(args.rate_delay)

    final_answer = merge_answers(question, partials, args.merge_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(question)[:60]
    out_file = out_dir / f"qa_{timestamp}_{slug}.md"

    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"# Question\n{question}\n\n")
        f.write("## Final Answer\n")
        f.write(final_answer + "\n\n")
        f.write("## Slice Answers\n")
        for idx, partial in enumerate(partials, start=1):
            f.write(f"### Slice {idx}\n{partial}\n\n")

    print(f"Analysis saved to {out_file}")


if __name__ == "__main__":
    main()
