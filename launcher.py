import subprocess

# Launch Streamlit app
streamlit_proc = subprocess.Popen(["streamlit", "run", "app.py"])

# Launch ngrok tunnel (replace with fixed URL if needed)
subprocess.Popen(["ngrok", "http", "8501"])

