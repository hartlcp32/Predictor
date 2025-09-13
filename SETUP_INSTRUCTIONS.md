# Setup Instructions for Stock Predictor

## How This Works

1. **Python Scripts (Local)**: Generate predictions on your computer
2. **JSON Data**: Predictions saved as `docs/predictions_data.json`
3. **GitHub Pages**: Displays the predictions from the JSON file
4. **Daily Updates**: Run the script locally and push to GitHub

## Initial Setup

### 1. Install Python Dependencies
```bash
cd C:\Projects\Predictor
pip install -r requirements.txt
```

### 2. Create GitHub Repository
1. Open GitHub Desktop
2. File → Add Local Repository → Choose `C:\Projects\Predictor`
3. If prompted "This directory does not appear to be a Git repository", click "Create a Repository"
4. Repository name: `stock-predictor` (or your choice)
5. Description: "Transparent stock predictions using 10 strategies"
6. Keep "Initialize with README" unchecked (we already have one)
7. Click "Create Repository"

### 3. Push to GitHub
1. In GitHub Desktop, you should see all the files
2. Write commit message: "Initial commit - stock predictor with 10 strategies"
3. Click "Commit to main"
4. Click "Publish repository"
5. Uncheck "Keep this code private" if you want it public
6. Click "Publish Repository"

### 4. Enable GitHub Pages
1. Go to your repository on github.com
2. Settings → Pages
3. Source: Deploy from a branch
4. Branch: main
5. Folder: /docs
6. Click Save
7. Wait 5-10 minutes for the site to be available at:
   `https://[your-username].github.io/stock-predictor/`

## Daily Workflow

### Generate Predictions (Run Every Trading Day)
```bash
cd C:\Projects\Predictor
python generate_predictions.py
```

### Push Updates to GitHub
1. Open GitHub Desktop
2. You'll see changes in `docs/predictions_data.json`
3. Write commit message: "Predictions for [date]"
4. Click "Commit to main"
5. Click "Push origin"

### Verify Updates
- Visit your GitHub Pages site
- The predictions should update within 2-5 minutes

## Automation Options

### Option 1: Windows Task Scheduler
1. Create a batch file `run_predictions.bat`:
```batch
cd C:\Projects\Predictor
python generate_predictions.py
git add .
git commit -m "Daily predictions %date%"
git push
```
2. Schedule it to run daily at 9:00 AM

### Option 2: GitHub Actions (Advanced)
You could set up GitHub Actions to run the predictions, but this requires:
- Storing API keys as GitHub secrets
- More complex setup
- May hit API rate limits

### Option 3: Manual Daily Run
- Most reliable for starting out
- Run the script each morning before market open
- Review predictions before publishing

## Tracking Performance

After the market closes:
1. Edit `docs/predictions_data.json`
2. Update the "actual" field for completed predictions
3. Commit and push the updates

## Important Notes

- **GitHub Pages Limitation**: It only serves static files, cannot run Python
- **Update Frequency**: You must manually run and push updates
- **API Limits**: Yahoo Finance has rate limits, don't run too frequently
- **Market Hours**: Best to run 30-60 minutes before market open
- **Weekends**: Skip running on weekends/holidays

## Troubleshooting

### "No module named yfinance"
```bash
pip install yfinance
```

### GitHub Pages not updating
- Check Settings → Pages for build status
- May take up to 10 minutes for first deployment
- Clear browser cache

### Predictions not showing
- Check browser console for errors
- Ensure `predictions_data.json` exists in docs folder
- Verify JSON is valid (no syntax errors)