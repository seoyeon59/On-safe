/**
 * 비디오 모달 창 열기 및 재생
 * @param {string} path - 비디오 파일 경로
 */
function playVideo(path) {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');
    const source = document.getElementById('videoSource');

    if (source && video && modal) {
        source.src = path;
        video.load(); // 새로운 소스 로드 필수
        modal.style.display = "block";
        video.play().catch(err => {
            print("자동 재생 방지 정책으로 인해 클릭 후 재생될 수 있습니다.");
        });
    }
}

/**
 * 모달 창 닫기 및 재생 중지
 */
function closeModal() {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');

    if (modal && video) {
        modal.style.display = "none";
        video.pause();
        video.currentTime = 0; // 재생 위치 초기화
    }
}

// 모달 바깥쪽(배경) 클릭 시 닫기 기능
window.addEventListener('click', function(event) {
    const modal = document.getElementById('videoModal');
    if (event.target === modal) {
        closeModal();
    }
});