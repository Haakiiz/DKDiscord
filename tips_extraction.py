import json
from openai import OpenAI
import os


client = OpenAI()


def read_cleaned_json(file_path):
    """Read the cleaned JSON file and return the list of messages."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def construct_prompt(messages):
    """
    Constructs a prompt that tells the LLM to produce a numbered list of messages.
    Each item will include the timestamp, username, and content.
    """
    prompt = (
        "You are given a large JSON file containing Discord chat logs primarily consisting of player interactions, discussions, and insights about a game that lacks comprehensive in-game hints beyond basic tutorials. Many useful gameplay tips, keyboard shortcuts, optimized loadouts, hidden features, and tricks are discussed by both developers and experienced gamers within these conversations. "
        "Your task is to systematically extract and summarize actionable insights from the JSON file, specifically focusing on the following categories: "
        "1. Gameplay Tips & Tricks: Non-obvious strategies improving gameplay efficiency or effectiveness; hidden or undocumented mechanics discovered by the community or developers. "
        "2. Keyboard Shortcuts & Controls: Specific key combinations and shortcuts not clearly documented officially (e.g., 'Pressing Shift + 1 on go-codes triggers them automatically when all soldiers are positioned correctly.'). "
        "3. Optimal Loadouts & Character Builds: Community-approved or developer-recommended gear, equipment, abilities, or skill trees, including situational or map-specific recommendations. "
        "4. Hidden or Secret Features: Undocumented game elements discovered through player experimentation or developer insights. "
        "Format your extraction clearly, categorizing each insight under the respective sections above. Ensure extracted information is concise, actionable, and practical for players seeking to improve their gameplay experience. Avoid general commentary or off-topic discussions.\n\n"
        "Each message contains an ID, timestamp, username, and message content. Please produce a readable numbered list of the messages, one per line, in the following format:\n\n"
        "1. [timestamp] - [username]: [message content]\n\n"
        "Do not include any extra commentary or metadata. Only output the numbered list.\n\n"
        "Here are the messages:\n"
    )

    for idx, msg in enumerate(messages, 1):
        timestamp = msg.get("timestamp", "N/A")
        username = msg.get("username", "N/A")
        content = msg.get("content", "").strip().replace("\n", " ")
        prompt += f"{idx}. {timestamp} - {username}: {content}\n"
    return prompt


def call_openai(prompt):
    """
    Calls the OpenAI ChatCompletion API (using GPT-4) with the provided prompt,
    and returns the model's output text.
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "You are an assistant that formats data into a clear numbered list."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )
    return response.output_text


def main():
    input_file = "cleaned_discord.json"  # The cleaned JSON file from earlier
    output_file = "llm_output.txt"  # The file where the LLM output will be saved

    messages = read_cleaned_json(input_file)
    if not messages:
        print("No messages found in the input file.")
        return

    prompt = construct_prompt(messages)
    print("Prompt constructed. Calling OpenAI API...")

    output_text = call_openai(prompt)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"LLM output saved to {output_file}.")


if __name__ == "__main__":
    main()
