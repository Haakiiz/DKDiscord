from playwright.sync_api import sync_playwright
import time
import json
from datetime import datetime



def get_oldest_timestamp(page):
    """
    Returns the oldest timestamp (as ISO string) from the loaded messages on the page.
    If no timestamps are found, returns None.
    """
    oldest = page.evaluate('''() => {
        const times = Array.from(document.querySelectorAll('time[datetime]')).map(el => el.getAttribute("datetime"));
        if (times.length === 0) return null;
        let minDate = new Date(times[0]);
        times.forEach(t => {
            const d = new Date(t);
            if (d < minDate) minDate = d;
        });
        return minDate.toISOString();
    }''')
    return oldest


def scroll_to_top(page, scroll_delay=2, scroll_amount=-2000, cutoff_date=None):
    """
    Scrolls upward through the chat history until no more new messages are loaded
    or until the oldest message is older than the cutoff_date (if provided).
    Uses the outer container with role="group" and a class starting with "scroller__".

    Args:
        page: The Playwright page instance.
        scroll_delay: Seconds to wait between scrolls.
        scroll_amount: The pixel amount to scroll per iteration.
        cutoff_date: A datetime object. Scrolling will stop when messages older than this date are loaded.
    """
    prev_scroll = None
    while True:
        # Scroll the outer container by the given amount (negative scroll_amount scrolls up)
        page.evaluate('''(amount) => {
            const scroller = document.querySelector('div[role="group"][class^="scroller__"]');
            if (scroller) {
                scroller.scrollBy(0, amount);
            }
        }''', scroll_amount)
        time.sleep(scroll_delay)

        # Check the current scroll position
        current_scroll = page.evaluate('''() => {
            const scroller = document.querySelector('div[role="group"][class^="scroller__"]');
            return scroller ? scroller.scrollTop : 0;
        }''')
        print("Current scroll position:", current_scroll)

        # If cutoff_date is provided, check the oldest message timestamp on the page.
        if cutoff_date:
            oldest_iso = get_oldest_timestamp(page)
            if oldest_iso:
                oldest_dt = datetime.fromisoformat(oldest_iso.replace("Z", "+00:00"))
                print("Oldest loaded message date:", oldest_dt.isoformat())
                # If the oldest loaded message is earlier than (or equal to) the cutoff, stop scrolling.
                if oldest_dt <= cutoff_date:
                    print("Reached the cutoff date. Stopping scroll.")
                    break

        # If scroll position hasn't changed or we've reached the top (scrollTop==0), stop scrolling.
        if current_scroll == prev_scroll or current_scroll == 0:
            print("Reached the top of the channel history.")
            break
        prev_scroll = current_scroll


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

    # Ask the user for an optional cutoff date in YYYY-MM-DD format.
    cutoff_input = input("Enter cutoff date (YYYY-MM-DD) to stop scrolling at older messages, or leave blank for full history: ").strip()
    cutoff_date = None
    if cutoff_input:
        try:
            # Assume UTC for simplicity.
            cutoff_date = datetime.fromisoformat(cutoff_input)
            print(f"Cutoff date set to: {cutoff_date.isoformat()}")
        except Exception as e:
            print("Invalid date format. Continuing without a cutoff.")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir="my-user-data-dir",
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            ]
        )

        page = context.new_page()
        page.goto("https://discord.com/channels/@me")

        input("Please log in manually and complete any CAPTCHA. Then press Enter here...")

        # Your scrolling and extraction code goes here.
        print("Scrolling through chat history. This may take a while if there are many messages...")
        scroll_to_top(page, cutoff_date=cutoff_date)

        print("Extracting messages from the loaded chat...")
        messages = extract_messages(page)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4)
        print(f"Chat log with {len(messages)} messages saved to {output_file}.")

        context.close()

if __name__ == "__main__":
    main()
