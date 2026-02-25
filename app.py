from flask import Flask, Response, render_template, request, redirect, session, jsonify
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from datetime import datetime
import pandas as pd
import joblib
from scipy.signal import savgol_filter
from playsound import playsound
import os
from collections import deque # 데이터 버퍼링용

# 분리된 DB 설정 파일에서 가져오기
from db_config import get_db_connection, engine

from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.urandom(24)

# AI 모델 로드
scaler = joblib.load("pkl/scaler.pkl")
model = joblib.load("pkl/decision_tree_model.pkl")

# 사비골 필터 설정 (수치 직접 입력)
WINDOW_SIZE = 5  # 홀수여야 함
POLY_ORDER = 2   # WINDOW_SIZE보다 작아야 함

# 3. 데이터 버퍼 (사비골용)
kp_buffer = {f'kp{i}_{axis}': deque(maxlen=WINDOW_SIZE) for i in range(33) for axis in ['x', 'y', 'z']}

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

## 전역 변수 초기화
frame_idx = 0
latest_frame = None
frame_lock = threading.Lock()
current_user_id = None
cap = None
fps = 30

prev_angles = {}
prev_angular_velocity = {}
prev_center = None
prev_center_speed = 0.0

latest_score = 0.0
latest_label = "Normal"

joint_triplets = [
    ('neck', 0, 11, 12), ('shoulder_balance', 11, 0, 12),
    ('shoulder_left', 23, 11, 13), ('shoulder_right', 24, 12, 14),
    ('elbow_left', 11, 13, 15), ('elbow_right', 12, 14, 16),
    ('hip_left', 11, 23, 25), ('hip_right', 12, 24, 26),
    ('knee_left', 23, 25, 27), ('knee_right', 24, 26, 28),
    ('ankle_left', 25, 27, 31), ('ankle_right', 26, 28, 32),
    ('torso_left', 0, 11, 23), ('torso_right', 0, 12, 24),
    ('spine', 0, 23, 24),
]

# 영상 저장 경로 설정
RECORD_DIR = "static/recordings"
if not os.path.exists(RECORD_DIR):
    os.makedirs(RECORD_DIR)

is_recording = False # 중복 녹화 방지 플래그

def compute_center_dynamics(df, fps=30, left_pelvis='kp23', right_pelvis='kp24'):
    global prev_center, prev_center_speed
    centers = []
    for _, row in df.iterrows():
        try:
            center = np.array([
                (row[f'{left_pelvis}_x'] + row[f'{right_pelvis}_x']) / 2,
                (row[f'{left_pelvis}_y'] + row[f'{right_pelvis}_y']) / 2,
                (row[f'{left_pelvis}_z'] + row[f'{right_pelvis}_z']) / 2
            ])
        except KeyError:
            center = np.array([np.nan, np.nan, np.nan])

        displacement = speed = acceleration = velocity_change = 0.0
        if prev_center is not None:
            displacement = np.linalg.norm(center - prev_center)
            speed = displacement * fps
            acceleration = (speed - prev_center_speed) * fps
            velocity_change = abs(speed - prev_center_speed)

        centers.append({
            'center_displacement': displacement,
            'center_speed': speed,
            'center_acceleration': acceleration,
            'center_velocity_change': velocity_change,
            'center_mean_speed': speed,
            'center_mean_acceleration': acceleration
        })
        prev_center, prev_center_speed = center, speed
    return pd.DataFrame(centers)


# 사비츠키-골레이 필터 적용 함수
def smooth_with_savgol(row_dict):
    smoothed_row = row_dict.copy()
    for key in kp_buffer.keys():
        if key in row_dict:
            kp_buffer[key].append(row_dict[key])
            if len(kp_buffer[key]) == WINDOW_SIZE:
                # pkl 없이 scipy 함수로 직접 필터링
                data = np.array(kp_buffer[key])
                filtered_signal = savgol_filter(data, WINDOW_SIZE, POLY_ORDER)
                smoothed_row[key] = filtered_signal[-1]  # 현재 프레임 값
    return smoothed_row


def centralize_kp(df, pelvis_idx=(23, 24)):
    df_central = df.copy()
    pelvis_x = (df[f'kp{pelvis_idx[0]}_x'] + df[f'kp{pelvis_idx[1]}_x']) / 2
    pelvis_y = (df[f'kp{pelvis_idx[0]}_y'] + df[f'kp{pelvis_idx[1]}_y']) / 2
    pelvis_z = (df[f'kp{pelvis_idx[0]}_z'] + df[f'kp{pelvis_idx[1]}_z']) / 2
    for x, y, z in zip([c for c in df.columns if '_x' in c], [c for c in df.columns if '_y' in c],
                       [c for c in df.columns if '_z' in c]):
        df_central[x] -= pelvis_x;
        df_central[y] -= pelvis_y;
        df_central[z] -= pelvis_z
    return df_central


def scale_normalize_kp(df, ref_joints=(23, 24)):
    df_scaled = df.copy()
    l, r = ref_joints
    scale = np.sqrt((df[f'kp{l}_x'] - df[f'kp{r}_x']) ** 2 + (df[f'kp{l}_y'] - df[f'kp{r}_y']) ** 2 + (
                df[f'kp{l}_z'] - df[f'kp{r}_z']) ** 2)
    scale[scale == 0] = 1
    for col in df.columns:
        if any(s in col for s in ['_x', '_y', '_z']): df_scaled[col] = df[col] / scale
    return df_scaled


def compute_angle(a, b, c):
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calculate_angles(row, fps=30):
    global prev_angles, prev_angular_velocity
    result = {}
    for j_name, a_idx, b_idx, c_idx in joint_triplets:
        try:
            a, b, c = [np.array([row[f'kp{i}_x'], row[f'kp{i}_y'], row[f'kp{i}_z']]) for i in [a_idx, b_idx, c_idx]]
            angle = compute_angle(a, b, c)
            result[f'{j_name}_angle'] = angle
            p_ang = prev_angles.get(f'{j_name}_angle', angle)
            ang_vel = (angle - p_ang) * fps
            result[f'{j_name}_angular_velocity'] = ang_vel
            p_vel = prev_angular_velocity.get(f'{j_name}_angular_velocity', ang_vel)
            result[f'{j_name}_angular_acceleration'] = (ang_vel - p_vel) * fps
            prev_angles[f'{j_name}_angle'], prev_angular_velocity[f'{j_name}_angular_velocity'] = angle, ang_vel
        except KeyError:
            result[f'{j_name}_angle'] = result[f'{j_name}_angular_velocity'] = result[
                f'{j_name}_angular_acceleration'] = 0.0
    return result


def get_camera_url(user_id):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT camera_url FROM users WHERE user_id = %s", (user_id,))
                row = cursor.fetchone()
                return row['camera_url'] if row else None
    except Exception as e:
        print(f"⚠️ URL 조회 오류: {e}");
        return None


def get_video_capture(url):
    try:
        return cv2.VideoCapture(url if url != '0' else 0)
    except:
        return None


def connect_camera_loop():
    global cap, fps, current_user_id
    while True:
        if cap is not None and cap.isOpened():
            time.sleep(1);
            continue
        url = get_camera_url(current_user_id) if current_user_id else 0
        temp_cap = get_video_capture(url)
        if temp_cap and temp_cap.isOpened():
            cap = temp_cap
            fps_v = int(cap.get(cv2.CAP_PROP_FPS))
            fps = fps_v if fps_v > 0 else 30
        else:
            time.sleep(3)


def capture_frames():
    global latest_frame, cap, frame_idx, fps, latest_score, latest_label, current_user_id
    last_analysis_time = 0

    # 분석 주기 설정 (0.3초 = 초당 약 3.3회 분석)
    # 낙상은 순식간에 일어나므로 1초보다는 0.3초 내외가 적당합니다.
    ANALYSIS_INTERVAL = 0.3

    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        # 1. UI 및 스트리밍용 최신 프레임 업데이트
        with frame_lock:
            latest_frame = frame.copy()
            frame_idx += 1

        cur_t = time.time()

        # 2. 정해진 분석 주기마다 실행
        if cur_t - last_analysis_time >= ANALYSIS_INTERVAL:
            last_analysis_time = cur_t

            # Mediapipe 처리를 위한 리사이징 및 RGB 변환
            rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                # 데이터 추출
                raw_row = {'frame': frame_idx}
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    raw_row[f'kp{i}_x'], raw_row[f'kp{i}_y'], raw_row[f'kp{i}_z'] = lm.x, lm.y, lm.z

                # [Step 1] 사비골 필터로 노이즈 제거
                smoothed_row = smooth_with_savgol(raw_row)
                df = pd.DataFrame([smoothed_row])

                # [Step 2] 데이터 가공 (중심점 역학, 정규화, 각도 계산)
                c_info = compute_center_dynamics(df, fps=fps).iloc[-1].to_dict()
                df_processed = scale_normalize_kp(centralize_kp(df))
                calc = calculate_angles(df_processed.iloc[0].to_dict(), fps=fps)
                calc.update(c_info)

                # [Step 3] AI 모델 예측
                if model and scaler:
                    try:
                        # 모델 학습 시 사용했던 특성(Angle, Center 관련) 추출
                        feat = [col for col in calc.keys() if any(x in col.lower() for x in ["angle", "center"])]
                        X = pd.DataFrame([calc])[feat].reindex(columns=scaler.feature_names_in_, fill_value=0.0)

                        X_scaled = scaler.transform(X)
                        prob = model.predict_proba(X_scaled)

                        latest_score = float(prob[0][1] * 100)  # 낙상 확률(%)
                        latest_label = "Fall" if model.predict(X_scaled)[0] == 1 else "Normal"

                        # 계산된 결과값 반영
                        calc["risk_score"] = latest_score
                        # 참고: DB 스키마에 Label 컬럼이 없다면 calc에서는 제외하고 전역변수만 업데이트합니다.
                        latest_label = latest_label

                    except Exception as e:
                        print(f"⚠️ AI 예측 오류: {e}")

                # [Step 4] 통합된 DB 저장 함수 호출
                # 여기서 realtime_screen 저장 + 80점 이상 시 녹화 스레드 실행이 동시에 처리됩니다.
                if current_user_id:
                    save_realtime_data(current_user_id, calc)

        # CPU 점유율 조절 (카메라 FPS에 맞춤)
        time.sleep(1 / fps if fps > 0 else 0.03)


def gen_frames():
    while True:
        with frame_lock:
            f = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
        _, buf = cv2.imencode('.jpg', f)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.01)


# 특정 점수(80) 이상일 때 영상을 녹화하고 detection_logs 테이블에 저장
def record_and_save_log(user_id, score):
    global is_recording, latest_frame
    if is_recording: return

    is_recording = True
    now = datetime.now()
    # 파일명 예시: fall_user1_20231027_143005.mp4
    filename = f"fall_{user_id}_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(RECORD_DIR, filename)

    # 영상 설정 (640x480, 20fps 기준 약 10초 녹화 테스트)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))

    print(f"🚨 위험 감지({score}점)! 녹화 시작: {filepath}")

    # 약 10분간(12000 프레임) 녹화
    frames_to_record = 12000
    count = 0
    while count < frames_to_record:
        if latest_frame is not None:
            out.write(latest_frame)
            count += 1
        time.sleep(0.05)

    out.release()
    is_recording = False
    print(f"✅ 녹화 종료 및 DB 기록 중...")

    # [DB 저장] detection_logs 테이블에 입력
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = """INSERT INTO detection_logs (user_id, event_time, risk_score, video_path) 
                         VALUES (%s, %s, %s, %s)"""
                # 웹에서 접근 가능한 경로로 저장 (예: /static/recordings/...)
                web_path = f"/static/recordings/{filename}"
                cursor.execute(sql, (user_id, now, score, web_path))
            conn.commit()
    except Exception as e:
        print(f"❌ detection_logs 저장 오류: {e}")


# ----- DB 저장 (스키마 및 유저 연동 수정) -----
def save_realtime_data(user_id, analysis_result):
    if not user_id: return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                analysis_result['user_id'] = user_id
                # DB 스키마(realtime_screen)에 실제 존재하는 컬럼만 필터링
                # Label 등 스키마에 없는 키는 제외해야 에러가 안 납니다.
                allowed_cols = [
                    'user_id', 'neck_angle', 'neck_angular_velocity', 'neck_angular_acceleration',
                    'shoulder_balance_angle', 'shoulder_balance_angular_velocity',
                    'shoulder_balance_angular_acceleration',
                    'shoulder_left_angle', 'shoulder_left_angular_velocity', 'shoulder_left_angular_acceleration',
                    'shoulder_right_angle', 'shoulder_right_angular_velocity', 'shoulder_right_acceleration',
                    'elbow_left_angle', 'elbow_left_angular_velocity', 'elbow_left_angular_acceleration',
                    'elbow_right_angle', 'elbow_right_angular_velocity', 'elbow_right_angular_acceleration',
                    'hip_left_angle', 'hip_left_angular_velocity', 'hip_left_angular_acceleration',
                    'hip_right_angle', 'hip_right_angular_velocity', 'hip_right_angular_acceleration',
                    'knee_left_angle', 'knee_left_angular_velocity', 'knee_left_angular_acceleration',
                    'knee_right_angle', 'knee_right_angular_velocity', 'knee_right_angular_acceleration',
                    'torso_left_angle', 'torso_left_angular_velocity', 'torso_left_angular_acceleration',
                    'torso_right_angle', 'torso_right_angular_velocity', 'torso_right_angular_acceleration',
                    'spine_angle', 'spine_angular_velocity', 'spine_angular_acceleration',
                    'ankle_left_angle', 'ankle_left_angular_velocity', 'ankle_left_angular_acceleration',
                    'ankle_right_angle', 'ankle_right_angular_velocity', 'ankle_right_angular_acceleration',
                    'center_speed', 'center_acceleration', 'center_displacement',
                    'center_velocity_change', 'center_mean_speed', 'center_mean_acceleration', 'risk_score'
                ]

                final_data = {k: v for k, v in analysis_result.items() if k in allowed_cols}

                columns = ', '.join(final_data.keys())
                placeholders = ', '.join(['%s'] * len(final_data))
                sql = f"INSERT INTO realtime_screen ({columns}) VALUES ({placeholders})"
                cursor.execute(sql, list(final_data.values()))

                # 최신 2000개 유지 (DB 용량 관리)
                cursor.execute(
                    "DELETE FROM realtime_screen WHERE user_id = %s AND id NOT IN "
                    "(SELECT id FROM (SELECT id FROM realtime_screen WHERE user_id = %s ORDER BY timestamp DESC LIMIT 2000) AS t)",
                    (user_id, user_id)
                )
            conn.commit()

            # 80점 이상 시 녹화 실행
            score = final_data.get('risk_score', 0)
            if score >= 80 and not is_recording:
                threading.Thread(target=record_and_save_log, args=(user_id, score), daemon=True).start()
    except Exception as e:
        print(f"❌ DB 저장 오류: {e}")


# 회원가입 실제 DB 저장을 담당할 보조 함수
def background_register(data):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = """INSERT INTO users 
                         (user_id, password, guardian_name, guardian_phone, ward_name, email, camera_url) 
                         VALUES (%s, %s, %s, %s, %s, %s, %s)"""
                cursor.execute(sql, (
                    data['id'], data['password'], data['username'], data['phone_number'],
                    data['non_guardian_name'], data['mail'], data['camera_url']
                ))
            conn.commit()
            print(f"✅ DB 가입 완료: {data['id']}")
    except Exception as e:
        print(f"❌ 백그라운드 가입 오류: {e}")

#######################
# ------ Flask ------
#######################

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

# ------ 로그인 기능 ------
@app.route('/login', methods=['POST'])
def login():
    global current_user_id
    user_id = request.form.get('id')
    password = request.form.get('password')

    if not user_id or not password:
        return render_template('login.html', error_msg="아이디와 비밀번호를 모두 입력해주세요.")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 암호화된 비번을 비교하기 위해 해시값을 가져옴
                cursor.execute("SELECT user_id, password FROM users WHERE user_id=%s", (user_id,))
                user = cursor.fetchone()

        # check_password_hash로 암호화된 비번과 입력값 비교
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user_id
            current_user_id = user_id
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error_msg="아이디 또는 비밀번호가 일치하지 않습니다.")
    except Exception as e:
        print(f"로그인 처리 오류: {e}")
        return render_template('login.html', error_msg="서버 오류가 발생했습니다.")


# ------ 기타 안내 페이지 -------
@app.route('/support')
def support():
    return render_template('support.html')


# ------ 아이디 찾기 API ------
@app.route('/api/find-id')
def api_find_id():
    name = request.args.get('name')
    email = request.args.get('email')
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = "SELECT user_id FROM users WHERE guardian_name = %s AND email = %s"
                cursor.execute(sql, (name, email))
                user = cursor.fetchone()
        if user:
            return jsonify({'success': True, 'user_id': user['user_id']})
        return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ------ 비밀번호 재설정 관련 API ------
@app.route('/api/verify-for-pw')
def api_verify_pw():
    user_id = request.args.get('id')
    email = request.args.get('email')
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = "SELECT user_id FROM users WHERE user_id = %s AND email = %s"
                cursor.execute(sql, (user_id, email))
                user = cursor.fetchone()
        if user:
            # 실제 사업화 시 여기서 이메일 발송 로직 추가
            return jsonify({'success': True})
        return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/update-pw', methods=['POST'])
def api_update_pw():
    data = request.json
    user_id = data.get('id')
    new_pw = data.get('new_pw')

    # 새 비밀번호 암호화
    hashed_pw = generate_password_hash(new_pw)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = "UPDATE users SET password = %s WHERE user_id = %s"
                cursor.execute(sql, (hashed_pw, user_id))
            conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ------ 회원가입 기능 (비밀번호 암호화 저장) ------
def background_register(data):
    try:
        # 사업화 필수: 비밀번호 해싱
        hashed_password = generate_password_hash(data['password'])

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = """INSERT INTO users 
                         (user_id, password, guardian_name, guardian_phone, ward_name, email, camera_url) 
                         VALUES (%s, %s, %s, %s, %s, %s, %s)"""
                cursor.execute(sql, (
                    data['id'], hashed_password, data['username'], data['phone_number'],
                    data['non_guardian_name'], data['mail'], data['camera_url']
                ))
            conn.commit()
            print(f"✅ DB 가입 완료 (암호화 적용): {data['id']}")
    except Exception as e:
        print(f"❌ 가입 오류: {e}")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        f = request.form.to_dict()
        threading.Thread(target=background_register, args=(f,)).start()
        return redirect('/')
    return render_template('register.html')


# ------ ID 중복 확인 ------
@app.route('/check_id')
def check_id():
    u_id = request.args.get('id')
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (u_id,))
            return jsonify({"exists": cursor.fetchone() is not None})


# ------ 로그아웃 ------
@app.route('/logout')
def logout():
    session.clear()
    global current_user_id
    current_user_id = None
    return redirect('/')


# ------ 메인 카메라 페이지 ------
@app.route('/camera')
def index():
    if 'user_id' not in session:
        return redirect('/')

    # DB에서 현재 사용자의 카메라 URL 가져오기
    u_id = session.get('user_id')
    c_url = get_camera_url(u_id)

    return render_template('camera.html', camera_url=c_url)

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return Response(status=403)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ------ 실시간 위험 점수 반환 -------
@app.route('/get_score')
def get_score():
    if 'user_id' not in session:
        return jsonify({"risk_score": 0.0})

    # 테스트 모드가 아닌 실제 DB 최신 점수 가져오기 (사업화 필수)
    try:
        df = pd.read_sql_query(
            "SELECT risk_score FROM realtime_screen WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1",
            con=engine,
            params=[session.get('user_id')]
        )
        # 만약 DB에 아직 데이터가 없다면 전역 변수 최신값 반환
        current_score = round(df['risk_score'].iloc[0], 2) if not df.empty else latest_score

        return jsonify({
            "risk_score": current_score
        })
    except Exception as e:
        print(f"점수 조회 오류: {e}")
        return jsonify({"risk_score": 0.0})


# ------ 사고 이력 페이지 ------
@app.route('/history')
def history():
    # 1. 로그인 여부 확인
    if 'user_id' not in session:
        return redirect('/')

    user_id = session.get('user_id')
    logs = []

    try:
        # 2. DB 연결 및 사고 로그 조회
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 최신 사고 영상이 위로 오도록 DESC(내림차순) 정렬
                # detection_logs 테이블의 모든 정보를 가져옵니다.
                sql = """
                    SELECT id, event_time, risk_score, video_path 
                    FROM detection_logs 
                    WHERE user_id = %s 
                    ORDER BY event_time DESC
                """
                cursor.execute(sql, (user_id,))
                logs = cursor.fetchall()  # 모든 결과 리스트로 받기

    except Exception as e:
        print(f"❌ 사고 이력 조회 중 오류 발생: {e}")
        # 오류 발생 시 빈 리스트를 전달하여 페이지 에러 방지
        logs = []

    # 3. 조회된 로그 데이터를 history.html 템플릿으로 전달
    return render_template('history.html', logs=logs)

# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    threading.Thread(target=connect_camera_loop, daemon=True).start()
    threading.Thread(target=capture_frames, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)