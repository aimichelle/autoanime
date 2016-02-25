from flask import Flask, render_template, request
app = Flask(__name__)

import cv2

@app.route("/")
def index():
    return render_template('index.html')



if __name__ == "__main__":
    app.debug = True
    app.run()