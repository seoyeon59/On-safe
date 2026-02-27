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
mock_users = [
    {'userId': 'admin', 'username': '관리자', 'mail': 'admin@test.com', 'password':'admin'},
    {'userId': 'test1234', 'username': '홍길동', 'mail': 'hong@test.com', 'password':'test1234'},
    {'userId': 'Test_User_01', 'username': '테스터', 'mail': 'test@test.com', 'password':'password'}
]


# --- [Routes] ---

@app.route('/')
def index_page():
    # 오류 수정: 'index'라는 함수가 없으므로 'camera'로 수정
    if 'user_id' in session:
        return redirect(url_for('camera'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST']) # GET과 POST를 모두 허용해야 함
def login():
    if request.method == 'POST':
        # 1. HTML 폼에서 보낸 데이터 가져오기
        user_id = request.form.get('id')
        password = request.form.get('password')

        # 2. 가상 DB(mock_users)에서 일치하는 유저 찾기
        user = next((u for u in mock_users if u['userId'] == user_id and u['password'] == password), None)

        if user:
            # 로그인 성공: 세션에 저장하고 카메라 페이지로 이동
            session['user_id'] = user_id
            return redirect(url_for('camera'))
        else:
            # 로그인 실패: 알림창 띄우기
            return "<script>alert('아이디 또는 비밀번호가 틀렸습니다.'); history.back();</script>"

    # GET 요청 시에는 로그인 페이지 보여주기
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
    # 1. 자동 할당을 주석 처리하세요. (로그인 페이지를 거치도록)
    if 'user_id' not in session:
        return redirect(url_for('login'))
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


# 1. 회원 탈퇴 페이지 렌더링
@app.route('/withdrawal')
def withdrawal_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('withdrawal.html')


# 2. 회원 탈퇴 처리 API
@app.route('/api/withdraw', methods=['POST'])
def api_withdraw():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '로그인이 필요합니다.'}), 403

    user_id = session.get('user_id')

    # 클라이언트가 보낸 데이터 가져오기
    # HTML의 name="reason_type"과 name="detailed_reason"에 대응합니다.
    reason_type = request.form.get('reason_type')
    detailed_reason = request.form.get('detailed_reason', '')

    try:
        # [STEP 1] 탈퇴 사유 저장 (SQL 기준 로직)
        # 실제 DB 연동 시:
        # cursor.execute("INSERT INTO withdrawal_reasons (reason_type, detailed_reason) VALUES (%s, %s)", (reason_type, detailed_reason))
        print(f"📉 [사유 기록] 유형: {reason_type} | 상세: {detailed_reason}")

        # [STEP 2] 사용자 삭제 (mock_users 리스트에서 제거)
        global mock_users
        mock_users = [u for u in mock_users if u['userId'] != user_id]
        print(f"🗑️ [회원 삭제] ID: {user_id} 가 시스템에서 제거되었습니다.")

        # [STEP 3] 세션 정리 및 로그아웃
        session.clear()

        # 탈퇴 완료 후 첫 페이지로 이동 (또는 별도의 완료 페이지)
        return """
            <script>
                alert('회원 탈퇴가 정상적으로 처리되었습니다. 그동안 이용해 주셔서 감사합니다.');
                window.location.href = '/';
            </script>
        """

    except Exception as e:
        print(f"❌ 탈퇴 처리 중 오류: {e}")
        return jsonify({'success': False, 'message': '처리 중 오류가 발생했습니다.'})


if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    print("✨ 통합 테스트 서버 시작: http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)