import json
import tiktoken
from tqdm import tqdm
import concurrent.futures
import time
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI()
MAX_TOKENS = 128000
encoder = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(encoder.encode(text))

BASE_PROMPT = """Extract and summarize actionable insights from a large JSON file containing Discord chat logs about a game lacking comprehensive in-game hints, focusing on specified categories of interest.

# Steps

1. **Data Loading**: Access and load the JSON file containing the Discord chat logs.
2. **Categorization**:
   - Identify relevant messages that fit under the following categories:
     - **Gameplay Tips & Tricks**: Strategies, mechanics, and insights not immediately obvious or documented.
     - **Keyboard Shortcuts & Controls**: Essential key combinations and shortcuts not officially documented.
     - **Optimal Loadouts & Character Builds**: Endorsed loadouts and character builds, including specific recommendations.
     - **Hidden or Secret Features**: Game elements discovered through experimentation.
3. **Information Extraction**:
   - Extract pertinent information from every message that fits any of the above categories.
4. **Insight Summarization**:
   - Summarize each identified message into concise, actionable insights, maintaining focus on relevancy and practicality.
5. **Formatting**:
   - Present information as a numbered list, formatted as: `[number]. [timestamp] - [username]: [message content]`

# Output Format

- First, put the relevant category along with a very short description in the title with markdown formatting (e.g. ## Keyboard Shortcuts & Controls - Shortcut for throwing grenades)
- Format: `[number]. [timestamp] - [username]: [message content]`

# Examples

**Example Start**
## Keyboard Shortcuts & Controls - Dodge rolls
1. [1630225600] - User123: Discovering you can dodge roll using Ctrl + Shift + Z boosts survival against enemy waves significantly.
2. [1630225700] - GameDev99: If you equip the Shadow Cloak and Light Bow, your stealth multipliers make it easy to clear the Dark Forest map.

(Note: Real examples will be longer and contain game-specific insights; ensure all content pertains to the categories specified.)

**Example End**

# Notes

- Focus strictly on extracting relevant information from the content, ignoring messages that do not contribute to the specified categories.
- Exclude general commentary, off-topic discussions, and irrelevant chat data.
- Ensure that the output remains free of metadata and extraneous information beyond the required format."""

RESERVED_TOKENS_FOR_PROMPT = count_tokens(BASE_PROMPT)
CHUNK_TOKEN_LIMIT = MAX_TOKENS - RESERVED_TOKENS_FOR_PROMPT

def chunk_messages_by_tokens(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    MAX_CHAR_LENGTH = 240000
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_chars = 0

    for msg in tqdm(messages, desc="Chunking messages by tokens", unit="msg"):
        content = f"{msg.get('timestamp', 'N/A')} - {msg.get('username', 'N/A')}: {msg.get('content', '').strip().replace('\\n', ' ')}"
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
                model="gpt-4o",
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
    with open("cleaned_discord.json", "r", encoding="utf-8") as f:
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
