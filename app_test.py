from flask import Flask, render_template, jsonify
import random

app = Flask(__name__)


# 메인 페이지 로드
@app.route('/')
def index():
    return render_template('camera.html')


# 실시간 위험 점수를 제공하는 API
@app.route('/get_score')
def get_score():
    # 0에서 100 사이의 랜덤 점수 생성 (실제 모델 연동 시 이 부분을 수정)
    # 예: score = model.predict(frame)
    current_score = random.randint(10, 95)

    return jsonify({
        "risk_score": current_score
    })


if __name__ == '__main__':
    # debug=True로 설정하면 코드를 수정할 때마다 서버가 자동으로 재시작됩니다.
    app.run(host='0.0.0.0', port=5000, debug=True)