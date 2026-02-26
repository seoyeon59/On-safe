window.addEventListener("DOMContentLoaded", () => {
  const riskScoreText = document.getElementById("riskScore");
  const arrow = document.getElementById("arrow");
  const popup = document.getElementById("popup");
  const btn119 = document.getElementById("btn119");
  const confirmBtn = document.getElementById("confirmBtn");
  const cancelBtn = document.getElementById("cancelBtn");

  // ==========================
  // 1. 위험 점수 업데이트 (1초마다 호출)
  // ==========================
  async function fetchRiskScore() {
    try {
      const response = await fetch("/get_score");
      if (!response.ok) throw new Error("네트워크 응답 없음");

      const data = await response.json();
      const score = Math.min(Math.max(Math.round(data.risk_score || 0), 0), 100);

      // 숫자 및 화살표 업데이트
      riskScoreText.textContent = `${score}%`;
      arrow.style.left = `calc(${score}% - 8px)`;

      // 점수별 색상 강조 (시인성 향상)
      updateScoreColor(score);

    } catch (e) {
      console.log("⚠️ 서버로부터 데이터를 기다리는 중...");
    }
  }

  function updateScoreColor(score) {
    if (score > 70) {
      riskScoreText.style.color = "#e74c3c"; // 위험: 빨강
    } else if (score > 40) {
      riskScoreText.style.color = "#f1c40f"; // 주의: 노랑
    } else {
      riskScoreText.style.color = "#27ae60"; // 정상: 초록
    }
  }

  // ==========================
  // 2. 119 신고 팝업 기능
  // ==========================
  btn119.onclick = () => popup.classList.add("show");
  cancelBtn.onclick = () => popup.classList.remove("show");

  confirmBtn.onclick = () => {
    popup.classList.remove("show");
    window.location.href = "tel:119";
  };

  // 실행: 1초 간격으로 점수 갱신
  setInterval(fetchRiskScore, 1000);
});