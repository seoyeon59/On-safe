window.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("webcam");
  const statusText = document.getElementById("status");
  const riskScoreText = document.getElementById("riskScore");
  const arrow = document.getElementById("arrow");

  // ==========================
  // 1. DroidCam 및 카메라 연결
  // ==========================
async function initCamera() {
    try {
        // navigator.mediaDevices가 있는지 먼저 확인
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("브라우저가 카메라 기능을 지원하지 않거나, 보안 연결(HTTPS/localhost)이 아닙니다.");
        }

        statusText.textContent = "⏳ 카메라 권한 요청 중...";
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        let selectedDeviceId = videoDevices.find(d => d.label.toLowerCase().includes("droidcam"))?.deviceId
                               || videoDevices[0]?.deviceId;

        const constraints = {
            video: { deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined }
        };

        const finalStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = finalStream;
        statusText.textContent = "✅ 카메라 연결 성공";
    } catch (err) {
        statusText.textContent = "❌ 카메라 오류: " + err.message;
        console.error("상세 에러:", err);
    }
}

  // ==========================
  // 2. 위험 점수 업데이트 (1초마다)
  // ==========================
  async function fetchRiskScore() {
    try {
      const response = await fetch("/get_score");
      if (!response.ok) throw new Error();

      const data = await response.json();
      const score = Math.min(Math.max(Math.round(data.risk_score || 0), 0), 100);

      // 숫자 업데이트
      riskScoreText.textContent = `${score}%`;

      // 화살표 위치 (CSS left 0% ~ 100%)
      arrow.style.left = `calc(${score}% - 8px)`; // 8px은 화살표 두께의 절반

      // 색상 변경
      if (score > 70) riskScoreText.style.color = "#e74c3c";
      else if (score > 40) riskScoreText.style.color = "#f1c40f";
      else riskScoreText.style.color = "#27ae60";

    } catch (e) {
      // 서버가 꺼져있을 때 에러 방지
      console.log("서버 점수 데이터 수신 대기 중...");
    }
  }

  // ==========================
  // 3. 신고 팝업 기능
  // ==========================
  const btn119 = document.getElementById("btn119");
  const popup = document.getElementById("popup");
  const confirmBtn = document.getElementById("confirmBtn");
  const cancelBtn = document.getElementById("cancelBtn");

  btn119.onclick = () => popup.classList.add("show");
  cancelBtn.onclick = () => popup.classList.remove("show");
  confirmBtn.onclick = () => {
    window.location.href = "tel:119";
  };

  // 실행
  initCamera();
  setInterval(fetchRiskScore, 1000);
});