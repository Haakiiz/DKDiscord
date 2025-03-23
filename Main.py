from playwright.sync_api import sync_playwright
import time
import json


def scroll_to_top(page, scroll_delay=2, scroll_amount=-2000):
    """
    Scrolls upward through the chat history until no more new messages are loaded.
    """
    prev_height = None
    while True:
        page.mouse.wheel(0, scroll_amount)
        time.sleep(scroll_delay)
        current_height = page.evaluate('''() => {
            // Using the new selector for the scroll container
            const el = document.querySelector('ol[class^="scrollerInner__"]');
            return el ? el.scrollHeight : 0;
        }''')
        if prev_height == current_height:
            print("Reached the top of the channel history.")
            break
        prev_height = current_height



def extract_messages(page):
    """
    Extracts messages from the Discord channel by querying the DOM.
    Returns a list of dictionaries with message details.
    """
    messages = page.evaluate('''() => {
        // Select all message content divs by id prefix
        const msgNodes = document.querySelectorAll('div[id^="message-content-"]');
        const msgs = [];
        msgNodes.forEach(node => {
            // Get the message content id and extract the unique suffix.
            const messageContentId = node.getAttribute("id");
            const suffix = messageContentId.replace("message-content-", "");

            // Combine text from all nested span elements in the content div.
            let content = "";
            const spans = node.querySelectorAll("span");
            spans.forEach(span => {
                content += span.innerText;
            });

            // Extract the username using the corresponding message-username id.
            const usernameElem = document.getElementById("message-username-" + suffix);
            const author = usernameElem ? usernameElem.innerText.trim() : "Unknown";

            // Extract the timestamp using the corresponding message-timestamp id.
            const timestampElem = document.getElementById("message-timestamp-" + suffix);
            const timestamp = timestampElem ? timestampElem.getAttribute("datetime") : null;

            msgs.push({
                "id": messageContentId,
                "author": author,
                "timestamp": timestamp,
                "content": content.trim()
            });
        });
        return msgs;
    }''')
    return messages


def main():
    output_file = "discord_channel.json"

    with sync_playwright() as p:
        # Launch in non-headless mode so you can log in manually.
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Navigate to Discord. Adjust the URL if you have a direct link to the channel.
        discord_url = "https://discord.com/channels/@me"
        print(f"Navigating to {discord_url}")
        page.goto(discord_url)

        input("Please log in and navigate to the desired channel. Press Enter once you're ready...")

        print("Scrolling through chat history. This may take a while if there are many messages...")
        scroll_to_top(page)

        print("Extracting messages from the loaded chat...")
        messages = extract_messages(page)

        # Save the structured messages as JSON for LLM processing
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4)
        print(f"Chat log with {len(messages)} messages saved to {output_file}.")

        browser.close()


if __name__ == "__main__":
    main()
