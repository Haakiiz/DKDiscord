import argparse
import os
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

try:
    import google.generativeai as genai
except ImportError:  # Gemini is optional unless chosen.
    genai = None


DEFAULT_GROK_MODEL = "grok-4-1-fast-non-reasoning"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not content:
                continue
            for part in content:
                ptype = getattr(part, "type", None)
                if ptype is None and isinstance(part, dict):
                    ptype = part.get("type")
                if ptype in {"output_text", "text"}:
                    ptext = getattr(part, "text", None)
                    if ptext is None and isinstance(part, dict):
                        ptext = part.get("text")
                    if isinstance(ptext, str):
                        chunks.append(ptext)
    return "".join(chunks).strip()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return input("Enter your prompt: ").strip()


def choose_provider(args: argparse.Namespace) -> str:
    if args.provider:
        return args.provider
    raw = input("Choose provider [gemini/grok] (default: gemini): ").strip().lower()
    return raw or "gemini"


def run_grok(prompt: str, model: str, max_output_tokens: int) -> str:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise SystemExit("XAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    text = extract_output_text(response)
    if not text:
        raise RuntimeError("Grok returned no text output.")
    return text


def run_gemini(prompt: str, model: str) -> str:
    if genai is None:
        raise SystemExit("google-generativeai is not installed.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    response = client.generate_content(prompt)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError("Gemini returned no text output.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a prompt with Gemini or Grok.")
    parser.add_argument("--provider", choices=["gemini", "grok"], help="LLM provider to use.")
    parser.add_argument("--prompt", help="Prompt text.")
    parser.add_argument("--prompt-file", help="Path to a file containing the prompt.")
    parser.add_argument("--out", help="Write output to this file.")
    parser.add_argument("--grok-model", default=DEFAULT_GROK_MODEL, help="Grok model name.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument("--max-output-tokens", type=int, default=2000, help="Max output tokens for Grok.")
    args = parser.parse_args()

    prompt = read_prompt(args)
    if not prompt:
        raise SystemExit("Prompt cannot be empty.")

    provider = choose_provider(args)
    if provider == "grok":
        output = run_grok(prompt, args.grok_model, args.max_output_tokens)
    elif provider == "gemini":
        output = run_gemini(prompt, args.gemini_model)
    else:
        raise SystemExit(f"Unknown provider: {provider}")

    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
