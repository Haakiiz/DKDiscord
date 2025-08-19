import json
import tiktoken
from tqdm import tqdm
import concurrent.futures
import time
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI()
MAX_TOKENS = 128000
encoder = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    return len(encoder.encode(text))

BASE_PROMPT = """Developer: You have access to the JSON export of messages from a Discord channel focused on the game "Door Kickers 2: Task Force North."

Begin with a concise checklist (3-7 bullets) outlining your approach to analyzing and summarizing the chat data.

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

If you do not find relevant topics or discussions in the chat data, state under Executive Summary: "No relevant discussions found in the provided chat log." and omit the Categorized Discussion Topics section."""

RESERVED_TOKENS_FOR_PROMPT = count_tokens(BASE_PROMPT)
CHUNK_TOKEN_LIMIT = MAX_TOKENS - RESERVED_TOKENS_FOR_PROMPT

def chunk_messages_by_tokens(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    MAX_CHAR_LENGTH = 240000
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_chars = 0

    for msg in tqdm(messages, desc="Chunking messages by tokens", unit="msg"):
        content = f"{msg.get('timestamp', 'N/A')} - {msg.get('username', 'N/A')}: {msg.get('content', '').strip().replace('', ' ')}"
        token_count = count_tokens(content)
        char_count = len(content)


        if (current_tokens + token_count > CHUNK_TOKEN_LIMIT or current_chars + char_count > MAX_CHAR_LENGTH) and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
            current_chars = 0

        current_chunk.append(msg)
        current_tokens += token_count
        current_chars += char_count

    print(f"detta er token count {current_tokens}")
    print(f"Detta er char_count: {current_chars}")

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def build_prompt_from_chunk(chunk):
    prompt = BASE_PROMPT + "\n"
    for idx, msg in enumerate(chunk, start=1):
        timestamp = msg.get("timestamp", "N/A")
        username = msg.get("username", "N/A")
        content = msg.get("content", "").strip().replace("\n", " ")
        prompt += f"{idx}. {timestamp} - {username}: {content}\n"
    return prompt

def call_llm_for_chunk(chunk, chunk_index, max_retries=5):
    prompt = build_prompt_from_chunk(chunk)
    print(f"Chunk {chunk_index}: Calling LLM (prompt length: {len(prompt)} characters)...")

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model="gpt-5-nano",
                input=prompt
            )
            output = getattr(response, "output_text", "")
            return chunk_index, output

        except Exception as e:
            error_message = str(e)
            if "rate_limit_exceeded" in error_message and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Rate limit hit. Retrying chunk {chunk_index} in {wait_time}s (attempt {attempt+1})...")
                time.sleep(wait_time)
            else:
                return chunk_index, f"Error: {e}"

def main():
    with open("Door Kickers - dev_q_and_a 01.03.2025-19.08.2025].json", "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected 'messages' to be a list in the JSON file.")

    print(f"Read {len(messages)} messages from cleaned_discord.json")

    chunks = chunk_messages_by_tokens(messages)
    print(f"Created {len(chunks)} chunks based on token limits.")

    results = [None] * len(chunks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(call_llm_for_chunk, chunk, idx): idx
            for idx, chunk in enumerate(chunks, start=1)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="LLM Calls", unit="chunk"):
            idx = futures[future]
            try:
                chunk_index, output_text = future.result()
                results[chunk_index - 1] = output_text
                print(f"Chunk {chunk_index} processed.")
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")
                results[idx - 1] = f"Error: {e}"

    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    output_filename = f"llm_outputs_{today}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        for idx, output in enumerate(results, start=1):
            output = output or ""
            f.write(f"=== Chunk {idx} ===\n")
            f.write(output + "\n\n")

    print(f"LLM processing complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
