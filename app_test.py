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
latest_score = 0

cap = cv2.VideoCapture(0)

def capture_frames():
    global latest_frame, latest_score
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
            latest_score = int(np.random.randint(0, 100))
        time.sleep(0.03)

def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
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
    # 오류 수정: 'index'라는 함수가 없으므로 'camera'로 수정
    if 'user_id' in session:
        return redirect(url_for('camera'))
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/find-id')
def find_id_page():
    return render_template('find_id.html')

@app.route('/find-pw')
def find_pw_page():
    return render_template('find_pw.html')

# --- [API 및 가상 로직] ---

# 오류 수정: 중복 정의된 api_find_id를 하나로 통합
@app.route('/api/find-id')
def api_find_id():
    name = request.args.get('name')
    email = request.args.get('email')
    found = next((u for u in mock_users if u['username'] == name and u['mail'] == email), None)
    if found:
        return jsonify({'success': True, 'user_id': found['userId']})
    return jsonify({'success': False, 'message': '일치하는 정보가 없습니다.'})

@app.route('/api/verify-for-pw')
def api_verify_pw():
    user_id = request.args.get('id')
    email = request.args.get('email')
    user = next((u for u in mock_users if u['userId'] == user_id and u['mail'] == email), None)
    if user:
        print(f"📧 [테스트] {email}로 인증 번호 발송")
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '사용자 정보를 찾을 수 없습니다.'})

@app.route('/api/update-pw', methods=['POST'])
def api_update_pw():
    data = request.json
    print(f"🔒 [테스트] {data.get('id')} 비밀번호 변경 완료")
    return jsonify({'success': True})

@app.route('/camera')
def camera():
    # 테스트 편의를 위한 자동 세션 할당
    session['user_id'] = 'Test_User_01'
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_score')
def get_score():
    return jsonify({'risk_score': latest_score})

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('history.html', logs=mock_logs)

@app.route('/check_id')
def check_id():
    user_id = request.args.get('id')
    exists = any(u['userId'] == user_id for u in mock_users)
    return jsonify({'exists': exists})

# 로그아웃 추가 (테스트 필수 기능)
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index_page'))

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    print("✨ 통합 테스트 서버 시작: http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)