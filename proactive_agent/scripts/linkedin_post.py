#!/usr/bin/env python3
"""
📱 LinkedIn Post Generator
========================
Generates a shareable post for your autonomous agent project.
"""

import os
from pathlib import Path
from datetime import datetime


PROJECT_LINK = "https://github.com/907-bot/Autonomous-browser-agent"

POST_TEMPLATE = """🤖 I just built an Ultimate Proactive Agent that lives in your system tray!

Ever wish you had an AI assistant that's ALWAYS there when you need it?

Introducing my new autonomous desktop agent that:
• Lives in your system tray, always running
• Summoned with Ctrl+Shift+A from ANY app
• Executes browser tasks autonomously
• Sends desktop notifications
• Runs in the background doing your bidding

Tech stack: Python + Playwright + CustomTkinter

Check it out: {link}

#Python #Automation #Productivity #Tech #Coding"""


def get_post() -> str:
    """Generate the LinkedIn post content."""
    return POST_TEMPLATE.format(link=PROJECT_LINK)


def generate_share_text() -> str:
    """Generate plain text version for clipboard copying."""
    return get_post()


def post_to_linkedin() -> dict:
    """Post to LinkedIn API.

    Requires environment variables:
    - LINKEDIN_TOKEN: OAuth2 access token
    - LINKEDIN_PERSONALITY_ID: Your person URN ID
    """
    import requests
    
    token = os.environ.get("LINKEDIN_TOKEN", "")
    personality_id = os.environ.get("LINKEDIN_PERSONALITY_ID", "")
    
    if not token:
        return {"error": "LINKEDIN_TOKEN not set"}
    
    if not personality_id:
        return {"error": "LINKEDIN_PERSONALITY_ID not set"}
    
    url = "https://api.linkedin.com/v2/ugcPosts"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    
    post_text = get_post()
    
    data = {
        "author": f"urn:li:person:{personality_id}",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": post_text},
                "shareMediaCategory": "ARTICLE",
                "media": [{
                    "status": "READY",
                    "originalUrl": PROJECT_LINK,
                    "title": {"text": "Ultimate Proactive Agent"},
                    "description": {"text": "An always-on desktop companion"}
                }]
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def save_draft(filename: str = "linkedin_post.md"):
    """Save post as a markdown file."""
    content = f"""# LinkedIn Post Draft

```
{get_post()}
```

## Copy & Paste

Copy the text above and paste it on LinkedIn!

## Or use API:

```bash
export LINKEDIN_TOKEN="your_token"
export LINKEDIN_PERSONALITY_ID="your_id"
python -m proactive_agent.scripts.linkedin_post --post
```
"""
    Path(filename).write_text(content)
    return filename


def copy_to_clipboard():
    """Copy post to system clipboard."""
    try:
        import pyperclip
        pyperclip.copy(get_post())
        return True
    except ImportError:
        # Fallback - print to stdout
        print("pyperclip not installed. Post:")
        print(get_post())
        return False


if __name__ == "__main__":
    import sys
    
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if "--post" in args:
        result = post_to_linkedin()
        print("📤 Result:", result)
    elif "--save" in args:
        filename = save_draft()
        print(f"💾 Saved: {filename}")
    elif "--copy" in args:
        if copy_to_clipboard():
            print("📋 Copied to clipboard!")
    else:
        print("=" * 60)
        print("📱 LINKEDIN POST")
        print("=" * 60)
        print(get_post())
        print("=" * 60)