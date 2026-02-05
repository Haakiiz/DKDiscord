import json


def clean_discord_json(input_file, output_file):
    # Load the exported JSON from DiscordChatExporter
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare a list to hold cleaned message data.
    cleaned_messages = []

    # Iterate over the messages
    for msg in data.get("messages", []):
        # Extract only the desired fields.
        cleaned_message = {
            "id": msg.get("id"),
            "timestamp": msg.get("timestamp"),
            "username": msg.get("author", {}).get("name"),
            "content": msg.get("content")
        }
        cleaned_messages.append(cleaned_message)

    # Optionally, you can output just the messages array
    output_data = {"messages": cleaned_messages}

    # Write the cleaned JSON to a new file.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"Cleaned data saved to {output_file} (total messages: {len(cleaned_messages)})")


if __name__ == "__main__":
    input_file = "Door Kickers - dev_q_and_a 01.03.2025-19.08.2025].json"  # Replace with your exported JSON filename
    output_file = "cleaned_discord.json"  # The file to write the cleaned data to
    clean_discord_json(input_file, output_file)
