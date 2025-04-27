#!/bin/bash
# Script to keep the Flask server running persistently

while true
do
    # Kill any process using port 5000
    fuser -k 5000/tcp

    # Start the Flask app in background
    nohup python3 plate_recognition/app.py --host=0.0.0.0 --port=5000 > flask_server.log 2>&1 &

    # Wait for 60 seconds before checking again
    sleep 60
done
