import time
import json
from datetime import datetime
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

"""BARE BRUK DISCORDCHATEXPORTER SOM DU KAN LASTE NED. DENNA HER FUNKA IKKE PGA BLOKKERT AV AUTOMATIONDETECTION!!"""

def transform_cookie(cookie):
    """
    Transforms a cookie from the exported format into the format expected by Playwright.
    """
    new_cookie = {
        "name": cookie["name"],
        "value": cookie["value"],
        "domain": cookie["domain"],
        "path": cookie["path"],
        "httpOnly": cookie["httpOnly"],
        "secure": cookie["secure"],
    }

    same_site = cookie.get("sameSite", "").lower()
    if same_site == "lax":
        new_cookie["sameSite"] = "Lax"
    elif same_site == "strict":
        new_cookie["sameSite"] = "Strict"
    elif same_site == "no_restriction":
        new_cookie["sameSite"] = "None"
    else:
        new_cookie["sameSite"] = "Lax"

    if not cookie.get("session", False) and "expirationDate" in cookie:
        new_cookie["expires"] = int(cookie["expirationDate"])
    return new_cookie


def get_oldest_timestamp(page):
    """
    Returns the oldest timestamp (as ISO string) from the loaded messages on the page.
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
    """
    prev_scroll = None
    while True:
        page.evaluate('''(amount) => {
            const scroller = document.querySelector('div[role="group"][class^="scroller__"]');
            if (scroller) {
                scroller.scrollBy(0, amount);
            }
        }''', scroll_amount)
        time.sleep(scroll_delay)

        current_scroll = page.evaluate('''() => {
            const scroller = document.querySelector('div[role="group"][class^="scroller__"]');
            return scroller ? scroller.scrollTop : 0;
        }''')
        print("Current scroll position:", current_scroll)

        if cutoff_date:
            oldest_iso = get_oldest_timestamp(page)
            if oldest_iso:
                oldest_dt = datetime.fromisoformat(oldest_iso.replace("Z", "+00:00"))
                print("Oldest loaded message date:", oldest_dt.isoformat())
                if oldest_dt <= cutoff_date:
                    print("Reached the cutoff date. Stopping scroll.")
                    break

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
        const msgNodes = document.querySelectorAll('div[id^="message-content-"]');
        const msgs = [];
        msgNodes.forEach(node => {
            const messageContentId = node.getAttribute("id");
            const suffix = messageContentId.replace("message-content-", "");
            let content = "";
            const spans = node.querySelectorAll("span");
            spans.forEach(span => {
                content += span.innerText;
            });
            const usernameElem = document.getElementById("message-username-" + suffix);
            const author = usernameElem ? usernameElem.innerText.trim() : "Unknown";
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
    cookie_file = "discord_cookies.json"  # Ensure this file exists in the same directory as your script.

    cutoff_input = input(
        "Enter cutoff date (YYYY-MM-DD) to stop scrolling at older messages, or leave blank for full history: ").strip()
    cutoff_date = None
    if cutoff_input:
        try:
            cutoff_date = datetime.fromisoformat(cutoff_input)
            print(f"Cutoff date set to: {cutoff_date.isoformat()}")
        except Exception as e:
            print("Invalid date format. Continuing without a cutoff.")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir="my-user-data-dir",
            headless=False,
            args=["--disable-blink-features=AutomationControlled",
                  "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"]
        )

        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                raw_cookies = json.load(f)
            cookies = [transform_cookie(cookie) for cookie in raw_cookies]
            context.add_cookies(cookies)
            print(f"Loaded and transformed {len(cookies)} cookies from {cookie_file}.")
        except Exception as e:
            print("Failed to load cookies. Please ensure the cookie file exists and is in the correct format.")
            context.close()
            return

        page = context.new_page()
        page.goto("https://discord.com/channels/@me")

        # Wait for the page to load completely
        page.wait_for_load_state("networkidle")

        # Now apply stealth modifications after the page has loaded
        stealth_sync(page)

        # Optionally force a reload if needed (uncomment the next two lines if you want to try a reload)
        # time.sleep(5)
        # page.reload()

        input("Check if you're logged in. Press Enter once confirmed...")

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
