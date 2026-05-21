# 🤖 Ultimate Proactive Agent

Your **always-on, autonomous desktop companion** that lives in your system tray and actively works to make your digital life easier.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Capabilities

### 🖥️ System Tray Residence
- **Always-on icon** in your system tray/dock
- **Right-click menu** for quick actions (Show, Toggle Autonomous, Quit)
- **Status indicator** - shows if agent is active or idle
- **Runs in background** even when minimized
- Persists across desktop sessions

### ⌨️ Global Hotkey Access
| Hotkey | Action |
|--------|--------|
| `Ctrl+Shift+A` | Toggle agent panel from anywhere |
| Works in any app | Even with other windows focused |
| `Escape` | Hide panel when open |

### 🌐 Browser Automation
- **Integrated Chrome/Chromium** via Playwright
- **Auto-navigation** to any URL
- **Form filling** - enters data automatically
- **Search execution** - performs web searches
- **Scraping** - extracts data from pages
- **Screenshot capture** - captures page visuals

### 🔔 Smart Notifications
- **Cross-platform** - works on Linux/macOS/Windows
- **Success alerts** - task completion notices
- **Warning alerts** - important warnings
- **Error alerts** - failure notifications
- **Custom titles** - personalize messages

### ⚡ Autonomous Mode
- **Task queue** - queues multiple tasks
- **Scheduled tasks** - run on intervals (minutely/hourly/daily)
- **Triggers** - react to conditions
- **Background monitoring** - watches for events
- **Self-healing** - recovers from errors

### 📝 Floating Panel
- **Draggable** - position anywhere
- **Always-on-top** - stays visible
- **Minimal footprint** - ~420x600px
- **Theme support** - dark/light modes
- **Live status** - shows current task
- **Quick actions** - one-click buttons

### 🌐 Quick Actions Buttons
| Button | Action |
|--------|--------|
| 🔍 Quick Search | Opens search interface |
| 📰 News | Opens news aggregator |
| 📧 Email | Opens email compose |
| 🛒 Shop | Opens shopping site |

### 💻 Programming API

```python
from proactive_agent import launch, quick_execute, get_notifier

# Launch GUI
launch()

# Execute task
result = quick_execute("Find Python tutorials", "https://google.com")

# Send notification
notifier = get_notifier()
notifier.notify("Done!", "Task completed successfully")
```

### 🔧 Advanced Features
- **Task history** - tracks all executed tasks
- **Config persistence** - saves user preferences
- **Logging** - detailed activity logs
- **Error recovery** - graceful failure handling
- **Extensible** - add custom triggers/actions

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python proactive_agent/main.py

# Or import in your app
from proactive_agent import launch
launch()
```

## 📖 Usage Examples

```python
from proactive_agent import quick_execute, get_notifier

# Execute browser tasks
result = quick_execute(
    "Search for Python tutorials",
    "https://google.com"
)

# Send notifications  
notifier = get_notifier()
notifier.notify("Task Complete", "Your search finished!")
```

## ⚙️ Configuration

Edit `~/.proactive_agent/config.json`:

```json
{
  "hotkey": "ctrl+shift+a",
  "start_minimized": false,
  "auto_start": true,
  "notifications": true,
  "autonomous_mode": true,
  "theme": "dark-blue"
}
```

### Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+Shift+A` | Toggle agent panel |
| `Ctrl+Shift+S` | Quick search |
| `Ctrl+Shift+X` | Emergency stop |
| `Escape` | Hide panel |

## 🏗️ Architecture

```
proactive_agent/
├── main.py              # Entry point & app controller
├── core/
│   ├── task_engine.py   # Browser task execution
│   ├── monitor.py      # Autonomous monitoring
│   └── browser_integration.py  # Browser bridge
└── utils/
    ├── notifications.py # Desktop alerts
    └── keybindings.py   # Hotkey management
```

## 🎯 Use Cases

- **Research Assistant** - Automatically search and gather info
- **Form Auto-fill** - Fill forms with your data
- **Price Tracking** - Monitor product prices
- **News Aggregation** - Daily news summaries
- **Email Automation** - Compose and send emails
- **Data Scraping** - Extract data from websites

## ⚠️ Requirements

- Python 3.8+
- System tray support (Linux/macOS/Windows)
- `libdbus` for Linux notifications (optional)

## 📝 License

MIT - Feel free to use and modify!

---

*🤖 Your tireless digital assistant*