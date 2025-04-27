import os
import subprocess
import time

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_flask_server():
    port = 8000
    if is_port_in_use(port):
        print(f"Port {port} is already in use.")
    else:
        print(f"Starting Flask server on port {port}...")
        # Kill any process using the port
        subprocess.run(f"fuser -k {port}/tcp", shell=True)
        # Start the Flask app in background
        subprocess.Popen(f"nohup python3 plate_recognition/app.py --host=0.0.0.0 --port={port} > flask_server.log 2>&1 &", shell=True)
        time.sleep(3)
        if is_port_in_use(port):
            print(f"Flask server started successfully on port {port}.")
        else:
            print(f"Failed to start Flask server on port {port}.")

if __name__ == "__main__":
    start_flask_server()
