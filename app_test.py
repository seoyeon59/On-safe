from flask import Flask, render_template, session, redirect, url_for, request, jsonify

app = Flask(__name__)
app.secret_key = 'integrated_test_key'

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

# 초기 화면 (로그인 페이지)
@app.route('/')
def login():
    return render_template('login.html') # 기존에 만드신 로그인 페이지

# 아이디 찾기
@app.route('/find-id')
def find_id():
    return render_template('find_id.html')

# 회원가입
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_user = request.form.to_dict()
        mock_users.append(new_user)
        print(f"✨ 가상 DB 저장: {new_user}")
        return redirect(url_for('login'))
    return render_template('register.html')

# 실시간 모니터링 페이지 (메인)
@app.route('/camera')
def camera():
    # 테스트를 위해 세션에 임시 유저 할당
    session['user_id'] = 'Test_User_01'
    return render_template('camera.html')

# 4. 사고 이력 조회 페이지 (History)
@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('history.html', logs=mock_logs)

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/find-id')
def find_id_page():
    return render_template('find_id.html')

@app.route('/find-pw')
def find_pw_page():
    return render_template('find_pw.html')

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
    if found:
        return jsonify({'success': True, 'user_id': found['userId']})
    return jsonify({'success': False})


# 비밀번호 재설정을 위한 유저 확인 API
@app.route('/api/verify-for-pw')
def api_verify_pw():
    user_id = request.args.get('id')
    email = request.args.get('email')

    # 가상 DB(mock_users)에서 확인
    user = next((u for u in mock_users if u['userId'] == user_id and u['mail'] == email), None)

    if user:
        # 실제로는 여기서 이메일 발송 로직이 들어감
        print(f"📧 [이메일 발송] {email}님, 인증번호는 [123456] 입니다.")
        return jsonify({'success': True})
    return jsonify({'success': False})

if __name__ == '__main__':
    print("✨ 통합 UI 테스트 서버가 가동되었습니다.")
    print("1. http://127.0.0.1:5000/ -> 로그인/회원가입 동선 테스트")
    print("2. http://127.0.0.1:5000/camera -> 메인 UI 및 버튼 테스트")
    app.run(debug=True, port=5000)