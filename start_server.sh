#!/bin/bash
# Script to start the Flask server for number plate recognition project

# Navigate to project directory
cd /project/sandbox/user-workspace

# Activate virtual environment if exists (optional)
# source venv/bin/activate

# Kill any process using port 8000
fuser -k 8000/tcp

# Start the Flask app in background with nohup
nohup python3 plate_recognition/app.py --host=0.0.0.0 --port=8000 > flask_server.log 2>&1 &

echo "Flask server started on port 8000"
