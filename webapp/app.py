"""
Free Web App for Automated Predictions
Deploy this to Render.com, Railway.app, or Vercel (all have free tiers)
"""

from flask import Flask, jsonify, render_template_string
import yfinance as yf
import json
import requests
from datetime import datetime
import os
import base64
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# GitHub configuration - Set these as environment variables
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')  # Personal Access Token
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'username/stock-predictor')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock Predictor Auto-Updater</title>
    <style>
        body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }
        .status { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }
        button { background: #e94560; color: white; border: none; padding: 10px 20px; 
                 border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #c13651; }
        .log { background: #0f3460; padding: 10px; border-radius: 5px; margin: 10px 0; }
        h1 { color: #e94560; }
    </style>
</head>
<body>
    <h1>📈 Stock Predictor Auto-Updater</h1>
    <div class="status">
        <h2>Status</h2>
        <p>Last Run: <span id="lastRun">{{ last_run }}</span></p>
        <p>Next Scheduled: <span id="nextRun">{{ next_run }}</span></p>
        <p>GitHub Repo: {{ github_repo }}</p>
    </div>
    
    <button onclick="runNow()">Run Predictions Now</button>
    
    <div class="status">
        <h2>Recent Logs</h2>
        <div id="logs">
            {% for log in logs %}
            <div class="log">{{ log }}</div>
            {% endfor %}
        </div>
    </div>
    
    <script>
        async function runNow() {
            const response = await fetch('/run-predictions');
            const data = await response.json();
            alert(data.message);
            location.reload();
        }
    </script>
</body>
</html>
"""

class GitHubUpdater:
    def __init__(self, token, repo, branch='main'):
        self.token = token
        self.repo = repo
        self.branch = branch
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.api_base = f'https://api.github.com/repos/{repo}'
    
    def get_file_sha(self, path):
        """Get the SHA of an existing file"""
        url = f"{self.api_base}/contents/{path}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()['sha']
        return None
    
    def update_file(self, path, content, message):
        """Update or create a file in the repository"""
        url = f"{self.api_base}/contents/{path}"
        
        # Encode content to base64
        content_encoded = base64.b64encode(content.encode()).decode()
        
        # Get existing file SHA if it exists
        sha = self.get_file_sha(path)
        
        data = {
            'message': message,
            'content': content_encoded,
            'branch': self.branch
        }
        
        if sha:
            data['sha'] = sha
        
        response = requests.put(url, json=data, headers=self.headers)
        return response.status_code in [200, 201]

def generate_simple_predictions():
    """Simplified prediction generator for web app"""
    predictions = {}
    strategies = ['momentum', 'mean_reversion', 'volume_breakout', 'technical',
                  'pattern', 'volatility', 'ma_cross', 'support_resist', 
                  'sentiment', 'ensemble']
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
              'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']
    
    try:
        # Fetch current prices (simplified)
        for i, strategy in enumerate(strategies):
            stock = stocks[i % len(stocks)]
            ticker = yf.Ticker(stock)
            hist = ticker.history(period="5d")
            
            if len(hist) > 1:
                change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                position = 'LONG' if change > 0 else 'SHORT'
                projected = f"{abs(change * 100):.1f}%"
                if position == 'SHORT':
                    projected = '-' + projected
                else:
                    projected = '+' + projected
            else:
                position = 'HOLD'
                projected = '0%'
            
            predictions[strategy] = {
                'stock': stock,
                'position': position,
                'projected': projected,
                'timeframe': '3D'
            }
    except Exception as e:
        print(f"Error generating predictions: {e}")
        # Return dummy predictions if error
        for strategy in strategies:
            predictions[strategy] = {
                'stock': 'ERROR',
                'position': 'HOLD',
                'projected': '0%',
                'timeframe': '-'
            }
    
    return predictions

def update_github_predictions():
    """Generate and push predictions to GitHub"""
    if not GITHUB_TOKEN:
        return {'success': False, 'message': 'GitHub token not configured'}
    
    try:
        # Generate predictions
        predictions = generate_simple_predictions()
        
        # Create JSON structure
        data = {
            'predictions': [{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'predictions': predictions,
                'generated_at': datetime.now().isoformat()
            }],
            'last_updated': datetime.now().isoformat()
        }
        
        # Update GitHub
        updater = GitHubUpdater(GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH)
        success = updater.update_file(
            'docs/predictions_data.json',
            json.dumps(data, indent=2),
            f"Daily predictions for {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        if success:
            app.config['LAST_RUN'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            app.config['LOGS'].append(f"{datetime.now()}: Successfully updated predictions")
            return {'success': True, 'message': 'Predictions updated successfully'}
        else:
            app.config['LOGS'].append(f"{datetime.now()}: Failed to update GitHub")
            return {'success': False, 'message': 'Failed to update GitHub'}
            
    except Exception as e:
        app.config['LOGS'].append(f"{datetime.now()}: Error - {str(e)}")
        return {'success': False, 'message': str(e)}

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=update_github_predictions,
    trigger="cron",
    hour=13,  # 1 PM UTC = 8 AM EST / 9 AM EDT
    minute=0,
    day_of_week='mon-fri',
    id='daily_predictions'
)
scheduler.start()

# Flask routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, 
        last_run=app.config.get('LAST_RUN', 'Never'),
        next_run='Weekdays at 13:00 UTC',
        github_repo=GITHUB_REPO,
        logs=app.config.get('LOGS', [])[-10:]  # Last 10 logs
    )

@app.route('/run-predictions')
def run_predictions():
    result = update_github_predictions()
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Initialize app config
app.config['LAST_RUN'] = 'Never'
app.config['LOGS'] = []

if __name__ == '__main__':
    # For local testing
    app.run(debug=True, port=5000)