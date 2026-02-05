# -*- coding: utf-8 -*-
import json
import os
import time
import google.generativeai as genai
import math # Import math for ceil

# --- Configuration ---
INPUT_JSON_FILE = "1.2 Generalchat_cleaned_discord.json"
OUTPUT_TXT_FILE = "doorkickers2_tips.txt"

# Use environment variables for secrets.
# Set GEMINI_API_KEY in your shell or .env file.
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY is not set. Export it before running this script.")

# Rate limit: 10 requests per minute -> 60 seconds / 10 requests = 6 seconds per request minimum.
# Add a small buffer.
RATE_LIMIT_DELAY_SECONDS = 6.1

# --- Gemini Configuration ---
try:
    genai.configure(api_key=API_KEY)
    # Using the specific model requested by the user
    MODEL_NAME = 'models/gemini-1.5-flash-latest' # Using stable flash model name
    # Note: 'gemini-2.0-flash-thinking-exp-01-21' seems to be causing issues or might be deprecated/renamed.
    # Let's try the standard 'gemini-1.5-flash-latest' which has good performance and limits.
    # If you specifically need the experimental one and it exists under a different name, adjust MODEL_NAME.
    # Standard Flash Limits (Free Tier): RPM: 15, TPM: 1,000,000
    # Adjust RATE_LIMIT_DELAY_SECONDS if using a model with different RPM.
    # For 15 RPM: 60 / 15 = 4 seconds. Let's use 4.1
    RATE_LIMIT_DELAY_SECONDS = 4.1 # Adjusted for 15 RPM of gemini-1.5-flash

    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Gemini API configured successfully using model: {MODEL_NAME}")
    print(f"Adjusted Rate Limit Delay for {MODEL_NAME}: {RATE_LIMIT_DELAY_SECONDS} seconds")

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure your API key is correct and valid for the specified model.")
    # If the error is about the model name, try finding the current correct name in Google AI Studio documentation.
    exit() # Exit if API configuration fails

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Helper Functions ---

def load_messages(filepath):
    """Loads messages from the specified JSON file."""
    try:
        # Explicitly open with utf-8-sig to handle potential BOM
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        if 'messages' in data and isinstance(data['messages'], list):
            print(f"Successfully loaded {len(data['messages'])} messages from {filepath}")
            return data['messages']
        else:
            print(f"Error: JSON file {filepath} does not contain a 'messages' list.")
            return None
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        # Provide more context on JSON decoding errors
        print(f"Error decoding JSON from {filepath}. Malformed JSON likely near character {e.pos}: {e.msg}")
        # Optional: Try to read a few lines around the error position if possible (advanced)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

def format_chunk_for_prompt(message_chunk):
    """Formats a list of message dictionaries into a string for the LLM prompt."""
    formatted_lines = []
    for msg in message_chunk:
        content = str(msg.get('content', '')) if msg.get('content') is not None else ''
        # Basic sanitization: replace potential prompt-breaking sequences if necessary (e.g., triple backticks)
        # content = content.replace("```", "` ` `") # Example basic sanitization
        username = str(msg.get('username', 'Unknown'))
        timestamp = str(msg.get('timestamp', 'N/A'))
        formatted_lines.append(f"[{timestamp}] {username}: {content}")
    return "\n".join(formatted_lines)


def extract_json_from_response(response_text):
    """Attempts to extract a JSON list from the Gemini response text."""
    json_string = None
    try:
        # Priority 1: Check for ```json ... ```
        json_marker_start = "```json"
        json_marker_end = "```"
        start_index = response_text.find(json_marker_start)
        if start_index != -1:
            end_index = response_text.find(json_marker_end, start_index + len(json_marker_start))
            if end_index != -1:
                json_string = response_text[start_index + len(json_marker_start):end_index].strip()

        # Priority 2: Check for ``` ... ``` if not found above
        if json_string is None:
            code_marker = "```"
            start_index = response_text.find(code_marker)
            if start_index != -1:
                end_index = response_text.find(code_marker, start_index + len(code_marker))
                if end_index != -1:
                    potential_json = response_text[start_index + len(code_marker):end_index].strip()
                    # Basic check if it looks like JSON list
                    if potential_json.startswith('[') and potential_json.endswith(']'):
                        json_string = potential_json

        # Priority 3: Look for plain [ ... ] if not found above
        if json_string is None:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = response_text[json_start:json_end+1].strip()
            # Handle simple '[]' case
            elif response_text.strip() == '[]':
                json_string = '[]'

        if json_string is not None:
            # Before parsing, try to fix common JSON issues like trailing commas
            # json_string = json_string.replace(r',]', ']')
            # json_string = json_string.replace(r',}', '}')
            extracted_data = json.loads(json_string)
            if isinstance(extracted_data, list):
                return extracted_data
            else:
                print(f"Warning: Extracted JSON is not a list. Content: {extracted_data}")
                return []
        else:
            print("Warning: Could not find JSON list structure ('[]', '```json [...]```', or '``` [...] ```') in response.")
            print(f"Response sample: {response_text[:300]}...") # Print more for debug
            return []

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to decode JSON from extracted string: {e}")
        context_snippet = json_string if json_string else response_text
        print(f"Problematic JSON string/response snippet (approx): {context_snippet[:300]}...")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during JSON extraction: {e}")
        return []


def analyze_chunk_with_gemini(chunk_text, chunk_number, total_chunks):
    """Sends a chunk of text to Gemini for analysis and extracts tips."""
    # Retry logic parameters
    max_retries = 2
    retry_delay_base = 5 # seconds

    for attempt in range(max_retries + 1):
        try:
            print(f"Sending chunk {chunk_number}/{total_chunks} to Gemini ({MODEL_NAME}) (Attempt {attempt+1}/{max_retries+1})...")
            response = model.generate_content(
                prompt.format(chunk_number=chunk_number, total_chunks=total_chunks, chunk_text=chunk_text), # Use .format for prompt
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    # response_mime_type="application/json" # Can try enabling this if model consistently outputs JSON
                     temperature=0.3 # Lower temperature might help focus on extraction
                )
            )

            # --- Rate Limiting Delay ---
            # Moved after successful call attempt, before processing response
            print(f"API call attempt complete. Waiting {RATE_LIMIT_DELAY_SECONDS} seconds...")
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
            # -------------------------

            response_text = ""
            # Proper check for content and blocked status
            if response.parts:
                response_text = response.text
            elif response.prompt_feedback.block_reason:
                 print(f"Warning: Prompt blocked for chunk {chunk_number}. Reason: {response.prompt_feedback.block_reason}")
                 if response.prompt_feedback.safety_ratings:
                    print(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                 return [] # Blocked, no retry needed for this specific error
            else:
                 # Check candidates for other finish reasons if parts are empty
                 finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                 print(f"Warning: Received response with no content parts for chunk {chunk_number}. Finish Reason: {finish_reason}")
                 if response.candidates and response.candidates[0].safety_ratings:
                      print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
                 # Consider retrying only if finish_reason is inconclusive or suggests temporary issue
                 if finish_reason not in ["STOP", "MAX_TOKENS", "SAFETY", "RECITATION"]: # Example: Retry on UNKNOWN/UNSPECIFIED
                      if attempt < max_retries:
                           wait_time = retry_delay_base * (2 ** attempt) # Exponential backoff
                           print(f"Retrying after {wait_time} seconds...")
                           time.sleep(wait_time)
                           continue # Go to next attempt
                      else:
                           print("Max retries reached for empty response.")
                           return []
                 else:
                    return [] # Don't retry if finished normally, blocked, or max tokens

            # If we got here, the API call was successful (or seemed so)
            extracted_tips = extract_json_from_response(response_text)
            return extracted_tips # Return successful result

        except (genai.api_core.exceptions.ResourceExhausted, genai.api_core.exceptions.ServiceUnavailable, genai.api_core.exceptions.DeadlineExceeded, genai.api_core.exceptions.InternalServerError) as e:
            print(f"\nAPI Error (Attempt {attempt+1}/{max_retries+1}) for chunk {chunk_number}: {type(e).__name__} - {e}")
            if isinstance(e, genai.api_core.exceptions.ResourceExhausted):
                 print("Quota/Rate Limit Error: Check usage limits. The script includes delays but limits might be stricter than documented or burst limits exceeded.")
                 # Check if the API suggests a retry delay
                 retry_info = getattr(e, 'retry', None) or getattr(e, 'metadata', None) # Try finding retry info
                 suggested_delay = 20 # Default wait for quota issues
                 if retry_info and hasattr(retry_info, 'retry_delay'):
                     try:
                         suggested_delay = max(suggested_delay, int(retry_info.retry_delay.total_seconds()))
                     except: pass # Ignore errors parsing suggested delay
                 print(f"API suggests waiting, or using default wait of {suggested_delay}s")
                 wait_time = suggested_delay
            else:
                 # Exponential backoff for other retryable errors
                 wait_time = retry_delay_base * (2 ** attempt)

            if attempt < max_retries:
                print(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Skipping this chunk.")
                return [] # Failed after retries

        except (gena
                .api_core.exceptions.InvalidArgument) as e:
             print(f"\nAPI InvalidArgument Error for chunk {chunk_number}: {e}")
             print("This often relates to safety settings, invalid model name, or issues with the prompt/data format.")
             print("No retry will be attempted for this chunk.")
             return [] # Do not retry on InvalidArgument

        except Exception as e:
            # Catch any other unexpected errors
            print(f"\nAn Unexpected Error occurred (Attempt {attempt+1}/{max_retries+1}) for chunk {chunk_number}: {type(e).__name__} - {e}")
            if attempt < max_retries:
                 wait_time = retry_delay_base * (2 ** attempt) # Exponential backoff
                 print(f"Retrying after {wait_time} seconds...")
                 time.sleep(wait_time)
            else:
                 print("Max retries reached for unexpected error. Skipping this chunk.")
                 return [] # Failed after retries

    return [] # Should not be reached if loop logic is correct, but safety return


# Define prompt as a template string outside the function
prompt_template = """
Analyze the following Discord chat log excerpt about the game Doorkickers 2. This is chunk {chunk_number}/{total_chunks}.
Your task is to identify and extract SPECIFIC, DETAILED, and GENERALLY APPLICABLE gameplay tips. Focus ONLY on:
1.  In-game shortcuts or keybinds (e.g., "Ctrl+F for silent mode").
2.  Optimal or effective weapon/equipment loadouts and configurations (e.g., "Use slugs in shotguns for better range because...", "Suppressors reduce noise significantly but slightly increase recoil", "AP ammo is crucial against armored enemies in X situation").
3.  Actionable tactical tips, techniques, or principles that apply broadly across different maps or scenarios (e.g., "When breaching, place the first trooper further back to cover the door while the second plants the charge.", "Use grenades/flashbangs to flush enemies from behind hard cover before entry.", "Slicing the pie technique explained...").

IGNORE ABSOLUTELY EVERYTHING ELSE, including but not limited to:
*   Vague or generic advice (e.g., "use cover", "be careful", "aim well", "communicate").
*   Map-specific callouts unless they clearly illustrate a general tactic applicable elsewhere.
*   Casual conversation, banter, memes, greetings, off-topic discussions, or personal opinions not directly tied to concrete gameplay mechanics or strategies.
*   Questions without clear answers providing a specific, actionable tip.
*   Simple statements of fact about the game unless they constitute a non-obvious tip (e.g., "The game has guns" is not a tip, but "AP ammo penetrates level III armor but not level IV" could be).
*   Complaints, bug reports, or feature requests.
*   Discussions about mods unless the tip is about a core game mechanic reflected in the mod.
*   Meta-commentary about the Discord server or community.
*   Role-playing or storytelling.

For each valid tip you find, extract the core advice, the username of the person who stated it, and the timestamp of the message containing the tip. If a tip is built over a couple of immediately consecutive messages by the same user, you can consolidate it, but primarily focus on self-contained tips within single messages. Use the timestamp/username from the key message providing the actual tip.

Format your response ONLY as a valid JSON list. Each item in the list must be a JSON object with the following keys:
*   "tip": (string) The extracted gameplay tip or advice, stated clearly and concisely.
*   "username": (string) The username of the message author.
*   "timestamp": (string) The timestamp of the message (ISO 8601 format).

Example of valid JSON objects:
[
  {{
    "tip": "Slugs extend shotgun range significantly, making them viable for medium distances.",
    "username": "RangeRover",
    "timestamp": "2025-01-18T12:00:00.000+01:00"
  }},
  {{
    "tip": "Hold SHIFT while drawing a path to make troopers move at maximum speed (sprint).",
    "username": "SpeedyG",
    "timestamp": "2025-01-19T15:22:10.543+01:00"
  }}
]

If no relevant tips meeting the strict criteria are found in the provided text excerpt, return an empty JSON list: []

Chat Log Excerpt (Chunk {chunk_number}/{total_chunks}):
--- START ---
{chunk_text}
--- END ---

JSON Response:
"""

def save_tips(tips_list, filepath):
    """Formats and saves the extracted tips to a text file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Extracted Doorkickers 2 Tips and Tricks\n")
            f.write("=========================================\n\n")

            if not tips_list:
                f.write("No specific gameplay tips meeting the criteria were found in the chat log analysis.\n")
                print(f"No valid tips found/extracted. Output file '{filepath}' created with a note.")
                return

            f.write("## Gameplay Tips, Shortcuts, and Loadouts\n\n")
            tip_count = 0
            processed_tips = set() # Use a set to store normalized tips to check for duplicates

            for i, tip_info in enumerate(tips_list):
                if isinstance(tip_info, dict) and 'tip' in tip_info and 'username' in tip_info and 'timestamp' in tip_info:
                    tip = tip_info.get('tip', '').strip() # Default to empty string
                    user = tip_info.get('username', '').strip()
                    ts = tip_info.get('timestamp', '').strip()

                    # Filter out empty or clearly invalid tips more strictly
                    if tip and user and ts and tip != 'N/A' and user != 'Unknown' and ts != 'N/A':
                        normalized_tip = ' '.join(tip.lower().split())
                        if normalized_tip not in processed_tips and len(normalized_tip) > 5: # Add basic length check
                            tip_count += 1
                            f.write(f"{tip_count}. Tip: {tip}\n")
                            f.write(f"   Source: {user} ({ts})\n\n")
                            processed_tips.add(normalized_tip)
                        elif normalized_tip in processed_tips:
                             print(f"Skipping duplicate tip: '{tip[:60]}...'")
                        else:
                            print(f"Skipping potentially invalid short/normalized tip: '{tip[:60]}...'")
                    else:
                        print(f"Skipping invalid/incomplete tip entry: {tip_info}")
                else:
                     print(f"Skipping malformed tip entry at index {i}: {tip_info}")

        if tip_count > 0:
            print(f"Successfully saved {tip_count} unique, valid tips to {filepath}")
        else:
            print(f"Although processing occurred, no unique, valid tips were extracted or saved. Check warnings above.")
            if 'f' in locals() and not f.closed:
                 f.write("\n(Analysis completed, but no unique, valid tips matching the criteria were extracted.)\n")

    except IOError as e:
         print(f"Error writing tips to file {filepath}: {e}. Check permissions or path.")
    except Exception as e:
        print(f"An unexpected error occurred while saving tips to {filepath}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Doorkickers 2 Tip Extraction...")

    # 1. Load Messages
    messages = load_messages(INPUT_JSON_FILE)

    if messages:
        total_messages = len(messages)
        all_extracted_tips = []

        # Reduced chunk size significantly
        chunk_size = 3000

        # Calculate total chunks needed
        num_chunks = math.ceil(total_messages / chunk_size)

        print(f"Total messages: {total_messages}")
        print(f"Using Model: {MODEL_NAME} | Chunk Size: {chunk_size} | Number of Chunks: {num_chunks}")
        print(f"Rate Limit Delay: {RATE_LIMIT_DELAY_SECONDS} seconds between API calls.")


        # 2. Process in Chunks
        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, total_messages)
            chunk = messages[start_index:end_index]
            current_chunk_number = i + 1

            print(f"\n--- Processing Chunk {current_chunk_number}/{num_chunks} (Messages {start_index+1}-{end_index}) ---")

            if not chunk:
                print("Chunk is empty, skipping.")
                continue

            valid_chunk_messages = [msg for msg in chunk if msg.get('content')]
            if not valid_chunk_messages:
                print("Chunk contains no messages with content, skipping.")
                continue

            # Format the prompt using the template
            prompt = prompt_template

            formatted_chunk = format_chunk_for_prompt(valid_chunk_messages)

            # 3. Analyze with Gemini (includes rate limiting delay and basic retry)
            extracted_chunk_tips = analyze_chunk_with_gemini(formatted_chunk, current_chunk_number, num_chunks)

            if extracted_chunk_tips:
                if isinstance(extracted_chunk_tips, list):
                    valid_tips_in_chunk = [tip for tip in extracted_chunk_tips if isinstance(tip, dict)]
                    print(f"Extracted {len(valid_tips_in_chunk)} potential tip objects in chunk {current_chunk_number}.")
                    all_extracted_tips.extend(valid_tips_in_chunk)
                else:
                    print(f"Warning: Gemini response for chunk {current_chunk_number} was not a list as expected. Skipping.")
            else:
                print(f"No relevant tips extracted or found in chunk {current_chunk_number}.")

        # 4. Save Results
        print("\nFinished processing all chunks.")
        save_tips(all_extracted_tips, OUTPUT_TXT_FILE)

    else:
        print(f"Could not load messages from {INPUT_JSON_FILE}. Exiting.")

    print("Script finished.")
