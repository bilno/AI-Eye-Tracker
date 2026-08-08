from flask import Flask, render_template, jsonify

import threading

from trackingdata import tracking_data
from main import run_eye_tracker

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/data")
def data():
    return jsonify(tracking_data)

if __name__ == "__main__":

    eye_tracker_thread = threading.Thread(
        target = run_eye_tracker,
        daemon = True
    )

    eye_tracker_thread.start()

    app.run(debug=True, use_reloader=False)