from flask import Flask, render_template, session, redirect, url_for, request, jsonify, Response
import cv2
import numpy as np
import threading
import time
import os

app = Flask(__name__)
app.secret_key = 'integrated_test_key'

# --- [카메라 및 스트리밍 설정] ---
latest_frame = None
frame_lock = threading.Lock()
latest_score = 0  # 테스트용 실시간 점수

# 0번은 기본 웹캠입니다. DroidCam 사용 시 "http://IP:4747/video" 형태로 수정하세요.
cap = cv2.VideoCapture(0)


def capture_frames():
    global latest_frame, latest_score
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            latest_frame = frame.copy()
            # 테스트를 위해 0~100 사이의 랜덤 점수 생성 (실제 모델 대신)
            # 실제 모델 연결 시 이 부분에서 MediaPipe 분석을 수행하면 됩니다.
            latest_score = int(np.random.randint(0, 100))

        time.sleep(0.03)  # 약 30 FPS


def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                # 브라우저 부하를 줄이기 위해 640x480으로 리사이징
                f = cv2.resize(latest_frame, (640, 480))
                _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buf.tobytes()
            else:
                frame_bytes = None

        if frame_bytes:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)


# --- [테스트용 가상 데이터] ---
mock_logs = [
    {'id': 1, 'event_time': '2026-02-26 10:30:15', 'risk_score': 92, 'video_path': '#'},
    {'id': 2, 'event_time': '2026-02-26 14:22:05', 'risk_score': 85, 'video_path': '#'},
    {'id': 3, 'event_time': '2026-02-25 09:10:44', 'risk_score': 77, 'video_path': '#'}
]

mock_users = [
    {'userId': 'admin', 'username': '관리자', 'mail': 'admin@test.com'},
    {'userId': 'test1234', 'username': '홍길동', 'mail': 'hong@test.com'}
]


# --- [Routes] ---

@app.route('/')
def index_page():
    # 로그인 되어 있으면 바로 카메라 페이지로, 아니면 랜딩 페이지로
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('index.html') # 새로운 랜딩 페이지



@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/camera')
def camera():
    # 테스트를 위해 세션 강제 할당
    session['user_id'] = 'Test_User_01'
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_score')
def get_score():
    # capture_frames에서 생성된 실시간 점수 반환
    return jsonify({'risk_score': latest_score})


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('history.html', logs=mock_logs)


# --- [API] ---
@app.route('/check_id')
def check_id():
    user_id = request.args.get('id')
    exists = any(u['userId'] == user_id for u in mock_users)
    return jsonify({'exists': exists})


@app.route('/api/find-id')
def api_find_id():
    name = request.args.get('name')
    email = request.args.get('email')
    found = next((u for u in mock_users if u['username'] == name and u['mail'] == email), None)
    return jsonify({'success': True, 'user_id': found['userId']}) if found else jsonify({'success': False})


if __name__ == '__main__':
    # 프레임 캡처 스레드 시작
    threading.Thread(target=capture_frames, daemon=True).start()

    print("✨ 통합 테스트 서버 시작: http://127.0.0.1:5000/camera")
    app.run(debug=True, port=5000, use_reloader=False)