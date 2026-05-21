# 🤖 Ultimate Proactive Agent

Your **always-on, autonomous desktop companion** that lives in your system tray and actively works to make your digital life easier.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖥️ **System Tray** | Lives in your system tray, always running |
| ⌨️ **Quick Access** | Press `Ctrl+Shift+A` to summon anytime |
| 🌐 **Browser Agent** | Integrated autonomous web automation |
| 🔔 **Smart Alerts** | Desktop notifications for important events |
| ⚡ **Autonomous Mode** | Proactively monitors and acts |
| 🎯 **Quick Actions** | One-click task buttons |
| 📝 **Floating Panel** | Draggable, always-on-top mini UI |

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