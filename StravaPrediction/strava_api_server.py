from flask import Flask, request, redirect
import requests

# Replace with your Strava app details
CLIENT_ID = 'client_id'
CLIENT_SECRET = 'client_secret'
REDIRECT_URI = 'http://localhost:5000/exchange_token'

app = Flask(__name__)

@app.route('/')
def home():
    # Redirect to Strava's OAuth authorization page
    auth_url = (
        f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}&response_type=code"
        f"&redirect_uri={REDIRECT_URI}&approval_prompt=auto&scope=activity:read_all"
    )
    return redirect(auth_url)

@app.route('/exchange_token')
def exchange_token():
    # Get the authorization code from the query parameters
    code = request.args.get('code')
    
    # Exchange the code for an access token
    token_response = requests.post(
        'https://www.strava.com/oauth/token',
        data={
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
        }
    )
    token_data = token_response.json()
    access_token = token_data.get('access_token')
    refresh_token = token_data.get('refresh_token')
    expires_at = token_data.get('expires_at')
    
    return f"Access Token: {access_token}<br>Refresh Token: {refresh_token}<br>Expires At: {expires_at}"

if __name__ == '__main__':
    app.run(debug=True)
