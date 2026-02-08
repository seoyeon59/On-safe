from flask import Flask, Response, render_template, request, redirect, session, jsonify
import random
import os


app = Flask(__name__)


# ==========================
# Flask 라우팅
# ==========================

# 홈 (로그인 페이지)
@app.route('/')
def home():
    return render_template('login.html')


# ------ 로그인 기능 -------
@app.route('/login', methods=['POST'])
def login():
    # 실제 프로젝트에서는 여기서 DB 검증을 수행합니다.
    user_id = request.form.get('id')
    password = request.form.get('password')

    if user_id and password:
        session['user_id'] = user_id
        return redirect('/camera')

    return render_template('login.html', error_msg="정보를 입력해주세요.")


# ----- 회원가입 기능 ------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 회원가입 로직 처리 후 홈으로 이동
        return redirect('/')
    return render_template('register.html')


# ----- 카메라 페이지 -----
@app.route('/camera')
def index():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('camera.html')


# 실시간 위험 점수를 제공하는 API
@app.route('/get_score')
def get_score():
    # 실제 모델 연동 전 테스트용 랜덤 점수
    current_score = random.randint(10, 95)
    return jsonify({
        "risk_score": current_score
    })


if __name__ == '__main__':
    # 스레드(capture_frames 등)가 있다면 여기서 미리 실행하는 코드가 필요할 수 있습니다.
    app.run(host='0.0.0.0', port=5000, debug=True)