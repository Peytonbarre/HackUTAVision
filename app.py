from flask import Flask, render_template, Response, jsonify
import cv2

app = Flask(__name__)

video_stream = cv2.VideoCapture(0)

@app.route('/')
def index():
    #return render_template('index.html')
    return "<p>Go to /video_feed to see your webcam!</p>"

def gen(camera):
    while True:
        frame = get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
     return Response(gen(video_stream),mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    ret, frame = video_stream.read()

    # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV

    ret, jpeg = cv2.imencode('.jpg', frame)

    return jpeg.tobytes()


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")