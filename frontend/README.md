# Autonomous Browser Agent Frontend

A modern, responsive web interface for the Autonomous Browser Agent, built with Next.js, TypeScript, and Tailwind CSS.

## ✨ Features

- 🎨 **Beautiful Dark Theme UI** - Modern glassmorphism design with animations
- ⚡ **Real-time Task Execution** - Execute browser automation tasks
- 🧠 **Task Planner** - Visualize task decomposition
- 📊 **Agent Status Dashboard** - Monitor agent health and activity
- 📜 **Task History** - Track and review past executions
- 📱 **Fully Responsive** - Works on desktop and mobile
- 🌐 **Deploy Anywhere** - Static export for GitHub Pages, Vercel, Netlify

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
cd frontend
npm install

# Run development server
npm run dev

# Open http://localhost:3000
```

### Build for Production

```bash
npm run build
```

The build output will be in the `out/` directory.

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx      # Root layout
│   │   ├── page.tsx         # Main page
│   │   └── globals.css      # Global styles
│   ├── components/
│   │   ├── Hero.tsx         # Hero section
│   │   ├── TaskExecutor.tsx # Task execution form
│   │   ├── TaskPlanner.tsx  # Task decomposition
│   │   ├── TaskHistory.tsx  # Task history list
│   │   ├── AgentStatus.tsx  # Agent status panel
│   │   └── About.tsx        # About section
│   ├── lib/
│   │   ├── api.ts           # API client
│   │   ├── store.ts         # Zustand store
│   │   └── utils.ts         # Utility functions
│   └── types/
│       └── index.ts         # TypeScript types
├── public/
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── next.config.js
```

## 🔌 API Integration

The frontend communicates with a FastAPI backend. Configure the API URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Endpoints

- `POST /execute` - Execute a task
- `POST /decompose` - Decompose a task
- `GET /status` - Get agent status
- `GET /health` - Health check

## 🎨 Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```javascript
colors: {
  primary: {
    50: '#f0f9ff',
    // ... custom colors
  },
}
```

### Components

All components are in `src/components/`. Modify them to fit your needs.

## 📦 Deploy

See [DEPLOYMENT.md](../DEPLOYMENT.md) for detailed deployment instructions.

### Quick Deploy

**GitHub Pages:**
```bash
npm run build
# Push to GitHub, enable Pages in settings
```

**Vercel:**
```bash
npx vercel
```

**Netlify:**
```bash
npx netlify deploy --prod
```

## 🛠️ Tech Stack

- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion
- **Icons**: Lucide React
- **State**: Zustand
- **Fonts**: Inter

## 📄 License

MIT License - See parent repository for details.