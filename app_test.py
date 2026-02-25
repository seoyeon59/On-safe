from flask import Flask, render_template, session
import cv2

app = Flask(__name__)
app.secret_key = 'test_key'

# 1. 테스트를 위해 가상의 로그인 세션을 미리 생성 (로그인 생략용)
@app.before_request
def make_session():
    session['user_id'] = 'test_user'

# 2. 실시간 카메라 화면 (camera.html) 테스트
@app.route('/camera')
def camera_test():
    # 실제 영상 주소 대신 테스트용 이미지나 더미 데이터를 보낼 수 있게 설정
    # 여기서는 camera.html 파일이 잘 렌더링 되는지 확인합니다.
    return render_template('camera.html')

# 3. 사고 영상 목록 화면 (history.html) 테스트
@app.route('/history')
def history_test():
    # 버튼 눌렀을 때 이 화면으로 넘어오는지 확인하기 위한 가짜 데이터
    test_logs = [
        {'event_time': '2026-02-26 14:00:00', 'risk_score': 95, 'video_path': '/static/test.mp4'}
    ]
    return render_template('history.html', logs=test_logs)

# 4. 초기 접속 시 카메라 화면으로 바로 가기
@app.route('/')
def index():
    return '<script>location.href="/camera"</script>'

if __name__ == '__main__':
    print("🚀 [테스트 시작] 브라우저에서 http://127.0.0.1:5000 접속 후 '영상 보기' 버튼을 클릭하세요.")
    app.run(debug=True, port=5000)