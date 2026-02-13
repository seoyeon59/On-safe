from flask import Flask, Response, render_template, request, redirect, session, jsonify
import random
import os

app = Flask(__name__)

# 세션을 사용하기 위해 반드시 필요합니다.
app.secret_key = os.urandom(24)

# ==========================
# Flask 라우팅
# ==========================

@app.route('/')
def home():
    return render_template('login.html')

# ------ 로그인 기능 -------
@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('id')
    password = request.form.get('password')

    # 테스트 용도: 아이디와 비번이 존재하면 로그인 성공으로 간주
    if user_id and password:
        session['user_id'] = user_id
        return redirect('/camera') # 오타 수정 완료

    return render_template('login.html', error_msg="정보를 입력해주세요.")

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/find-id')
def find_id_page():
    # 아이디 찾기 버튼을 눌렀을 때 보여줄 페이지
    return render_template('find_id.html')

@app.route('/find-pw')
def find_pw_page():
    # 비밀번호 찾기 버튼을 눌렀을 때 보여줄 페이지
    return render_template('find_pw.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 회원가입 로직 (DB 저장 등) 수행
        return redirect('/')
    return render_template('register.html')

@app.route('/camera')
def index():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('camera.html')

@app.route('/get_score')
def get_score():
    # 실시간 위험 점수 (테스트용 랜덤값)
    current_score = random.randint(10, 95)
    return jsonify({
        "risk_score": current_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)