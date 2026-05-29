# 🚀 Deployment Guide

This guide covers deploying the Autonomous Browser Agent to free hosting platforms.

## 📋 Prerequisites

- GitHub account
- Docker Hub account (for Render)
- Basic understanding of terminal/command line

---

## Option 1: Deploy Frontend to GitHub Pages (Static Export)

GitHub Pages is free for open source projects. Since we're using Next.js static export, this works great!

### Step 1: Update Next.js Configuration

Update `frontend/next.config.js`:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  basePath: '/autonomous-browser-agent', // Change to your repo name
}

module.exports = nextConfig
```

### Step 2: Create GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: ['main']

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        working-directory: ./frontend
        run: npm ci

      - name: Build
        working-directory: ./frontend
        env:
          NEXT_PUBLIC_API_URL: ${{ secrets.API_URL }}
        run: npm run build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./frontend/out
```

### Step 3: Push to GitHub

```bash
git add .
git commit -m "Add Next.js frontend for deployment"
git push origin main
```

### Step 4: Configure GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** > **Pages**
3. Under "Source", select **GitHub Actions**
4. The workflow will automatically deploy on push

---

## Option 2: Deploy Full Stack to Render

Render offers free tier for web services with Docker.

### Backend (Render)

1. **Create Render Account**: Go to [render.com](https://render.com) and sign up

2. **Create Web Service**:
   - Click **New +** > **Web Service**
   - Connect your GitHub repository
   - Configure:
     - **Root Directory**: Leave empty (or specify `/backend`)
     - **Build Command**: `pip install -r requirements.txt` (or use Dockerfile)
     - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

3. **Set Environment Variables**:
   - `PYTHONUNBUFFERED=1`
   - `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium`

4. **Add Health Check**:
   - Path: `/health`
   - Interval: 30 seconds

5. **Deploy**: Click "Create Web Service"

### Frontend (Render Static)

1. **Create Render Account**: Same account as above

2. **Create Static Site**:
   - Click **New +** > **Static Site**
   - Connect your GitHub repository
   - Configure:
     - **Root Directory**: `frontend`
     - **Build Command**: `npm ci && npm run build`
     - **Publish Directory**: `out`

3. **Set Environment Variables**:
   - `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`

4. **Deploy**: Click "Create Static Site"

---

## Option 3: Deploy to Vercel (Recommended)

Vercel is the creator of Next.js and offers excellent free tier.

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Deploy

```bash
cd frontend
vercel
```

Follow the prompts:
- Set up account? Yes
- Which scope? Your personal account
- Link to existing project? No (create new)
- Project name? `autonomous-browser-agent`
- Directory? Current directory (.)
- Override settings? No

### Step 3: Configure Environment Variables

In Vercel Dashboard:
1. Go to your project
2. Navigate to **Settings** > **Environment Variables**
3. Add:
   - `NEXT_PUBLIC_API_URL` = your backend URL (e.g., `https://api.yourdomain.com`)

### Step 4: Custom Domain (Optional)

1. Go to **Settings** > **Domains**
2. Add your custom domain
3. Update DNS records as instructed

---

## Option 4: Deploy to Netlify

Netlify also offers excellent free hosting for static Next.js sites.

### Step 1: Install Netlify CLI

```bash
npm install -g netlify-cli
```

### Step 2: Deploy

```bash
cd frontend
netlify deploy --prod
```

### Step 3: Configure

Create `frontend/netlify.toml`:

```toml
[build]
  command = "npm run build"
  publish = "out"

[build.environment]
  NODE_VERSION = "18"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
```

---

## 🐳 Docker Deployment

### Build Images

```bash
# Build backend
docker build -f Dockerfile.backend -t autonomous-agent-backend .

# Build frontend
docker build -f Dockerfile.frontend -t autonomous-agent-frontend .
```

### Run with Docker Compose

```bash
docker-compose up -d
```

### Manual Run

```bash
# Run backend
docker run -p 8000:8000 autonomous-agent-backend

# Run frontend
docker run -p 3000:3000 autonomous-agent-frontend
```

---

## 🔧 Configuration for Production

### Backend

Create `backend/.env`:

```env
PYTHONUNBUFFERED=1
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
LOG_LEVEL=INFO
MAX_STEPS_DEFAULT=30
```

### Frontend

Update environment variables:

```env
NEXT_PUBLIC_API_URL=https://api.example.com
```

---

## 📊 Monitoring

### Health Check Endpoint

The backend provides a health check at `/health`:

```bash
curl https://your-backend.onrender.com/health
```

Response:
```json
{
  "status": "healthy",
  "agent_available": true,
  "timestamp": "2024-01-01T00:00:00"
}
```

### Status Endpoint

Get agent status at `/status`:

```bash
curl https://your-backend.onrender.com/status
```

---

## 🆘 Troubleshooting

### Common Issues

**1. CORS Errors**
- Ensure backend has CORS middleware configured
- Check that frontend URL is in allowed origins

**2. Build Failures on Render**
- Check that all dependencies are in `requirements.txt`
- Verify Python version compatibility

**3. Agent Not Available**
- Ensure all agent components are properly installed
- Check logs for import errors

**4. Static Export Issues**
- Ensure no server-only features are used
- Check that all API calls use environment variables

---

## 📝 Summary Table

| Platform | Type | Free Tier | Static Export | Docker |
|----------|------|-----------|----------------|--------|
| GitHub Pages | Static | ✅ Yes | ✅ Yes | ❌ |
| Render | Full Stack | ✅ Yes (750h/mo) | ✅ Yes | ✅ Yes |
| Vercel | Full Stack | ✅ Yes (100GB) | ✅ Yes | ❌ |
| Netlify | Static | ✅ Yes (100GB) | ✅ Yes | ❌ |

---

## 🔗 Quick Links

- [GitHub Pages](https://pages.github.com/)
- [Render](https://render.com/)
- [Vercel](https://vercel.com/)
- [Netlify](https://netlify.com/)
- [Docker](https://docker.com/)

---

## 💡 Tips

1. **Use separate backends**: For production, keep backend and frontend on separate services
2. **Environment variables**: Never commit API keys or secrets
3. **Health checks**: Implement health checks for auto-scaling
4. **Logs**: Set up centralized logging for debugging
5. **CDN**: Use CDN for static assets to improve performance

---

Happy Deploying! 🚀