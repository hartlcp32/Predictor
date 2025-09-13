# Free Automated Web App Options

## Option 1: GitHub Actions (EASIEST - 100% FREE)
**No external service needed!** GitHub provides 2,000 free minutes/month for Actions.

### Setup:
1. Push your repo to GitHub
2. The `.github/workflows/daily-predictions.yml` file will automatically run
3. Go to Actions tab and enable workflows
4. It will run automatically every weekday at 8 AM EST

### Manual trigger:
- Go to Actions tab → "Generate Daily Predictions" → Run workflow

---

## Option 2: Render.com (FREE TIER)

### Deploy the Flask app:
1. Create account at [render.com](https://render.com)
2. New → Web Service → Connect GitHub repo
3. Settings:
   - Build Command: `pip install -r webapp/requirements.txt`
   - Start Command: `gunicorn webapp.app:app`
4. Environment Variables:
   - `GITHUB_TOKEN`: Your GitHub Personal Access Token
   - `GITHUB_REPO`: `yourusername/stock-predictor`

### Create GitHub Token:
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo` (all)
4. Copy token and add to Render environment variables

---

## Option 3: Railway.app (FREE TIER)

1. Sign up at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Add environment variables (same as Render)
4. It auto-deploys when you push changes

---

## Option 4: Vercel (FREE TIER)

Create `webapp/vercel.json`:
```json
{
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel` in webapp directory
3. Add environment variables in Vercel dashboard

---

## Option 5: PythonAnywhere (FREE TIER)

1. Create account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files to PythonAnywhere
3. Set up scheduled task for daily runs
4. Limited to one scheduled task on free tier

---

## Option 6: Google Cloud Run (FREE TIER)
- 2 million requests/month free
- More complex setup but very reliable

---

## RECOMMENDED: Use GitHub Actions

It's the simplest, most reliable, and completely free option. No external services needed!

The workflow file is already created and will:
- Run every weekday at 8 AM EST
- Fetch latest stock data
- Generate predictions
- Commit and push to your repo
- GitHub Pages automatically updates

Just push your repo and enable Actions!