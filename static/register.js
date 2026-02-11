document.addEventListener('DOMContentLoaded', function() {
  // --- 화면 섹션 요소 ---
  const privacySection = document.getElementById('privacySection');       // container1 (약관)
  const registrationSection = document.getElementById('registrationSection'); // container2 (폼 부모)
  const registerFormDiv = document.getElementById('registerForm');       // 실제 폼 영역

  // --- 동의 관련 요소 ---
  const agreeAll = document.getElementById('agreeAll');
  const subAgrees = document.querySelectorAll('.sub-agree');
  const nextStepBtn = document.getElementById('nextStepBtn');
  const agreeError = document.getElementById('agreeError'); // HTML에 이 ID를 가진 div가 없다면 생성 권장

  // --- 입력 및 에러 메시지 요소 ---
  const form = registerFormDiv.querySelector('form');
  const userIdInput = document.getElementById('userId');
  const userIdError = document.getElementById('userIdError');
  const password = document.getElementById('password');
  const passwordConfirm = document.getElementById('passwordConfirm');
  const passwordFormatError = document.getElementById('passwordFormatError');
  const passwordError = document.getElementById('passwordError');

  let isIdTaken = false;

  // ------------- 화면 전환: 전체 동의 체크 시 폼 표시 -------------
  agreeAll.addEventListener('change', function() {
    subAgrees.forEach(checkbox => {
      checkbox.checked = agreeAll.checked;
    });
    validateAgreements();
  });

  // ------------- 개별 체크박스 로직 -------------
  subAgrees.forEach(checkbox => {
    checkbox.addEventListener('change', function() {
      const allChecked = Array.from(subAgrees).every(cb => cb.checked);
      agreeAll.checked = allChecked;
      validateAgreements();
    });
  });

  // ------------- 동의 여부 확인 및 버튼 활성화 -------------
  function validateAgreements() {
    const allChecked = Array.from(subAgrees).every(cb => cb.checked);
    if (allChecked) {
      nextStepBtn.classList.add('active');
      nextStepBtn.style.cursor = 'pointer';
      agreeError.style.display = 'none';
    } else {
      nextStepBtn.classList.remove('active');
      nextStepBtn.style.cursor = 'not-allowed';
    }
  }

  // ------------- 다음 단계 버튼 클릭 시 화면 전환 -------------
  nextStepBtn.addEventListener('click', function() {
    const allChecked = Array.from(subAgrees).every(cb => cb.checked);
    if (allChecked) {
      privacySection.style.display = 'none';
      registrationSection.style.display = 'block';
      window.scrollTo(0, 0);
    } else {
      agreeError.style.display = 'block';
    }
  });

  // ------------- 아이디 중복 체크 (서버 통신) -------------
  userIdInput.addEventListener('blur', async function() {
    const userId = userIdInput.value.trim();
    if (!userId) {
      hideUserIdError();
      return;
    }

    try {
      const response = await fetch(`/check_id?id=${encodeURIComponent(userId)}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();

      if (data.exists) {
        showUserIdError("이미 존재하는 아이디입니다.");
      } else {
        hideUserIdError();
      }
    } catch (err) {
      console.error("아이디 체크 중 오류:", err);
      // 서버 연결 실패 시에도 에러 메시지 표시
      showUserIdError("아이디 중복 확인에 실패했습니다.");
    }
  });

  function showUserIdError(msg) {
    userIdError.textContent = msg;
    userIdError.style.display = 'block';
    userIdInput.classList.add('error-input');
    isIdTaken = true;
  }

  function hideUserIdError() {
    userIdError.style.display = 'none';
    userIdInput.classList.remove('error-input');
    isIdTaken = false;
  }

  // ------------- 비밀번호 검증 로직 -------------
  function checkPasswordFormat() {
    const value = password.value;
    // 8자 이상, 영문, 숫자, 특수문자 조합 정규식
    const isValid = /^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]).{8,}$/.test(value);

    if (value && !isValid) {
      passwordFormatError.style.display = 'block';
    } else {
      passwordFormatError.style.display = 'none';
    }
    return isValid;
  }

  function checkPasswordMatch() {
    const match = (password.value === passwordConfirm.value);
    // 비밀번호 확인 칸에 값이 있을 때만 불일치 메시지 표시
    if (passwordConfirm.value && !match) {
      passwordError.style.display = 'block';
    } else {
      passwordError.style.display = 'none';
    }
    return match;
  }

  password.addEventListener('input', () => {
    checkPasswordFormat();
    checkPasswordMatch();
  });

  passwordConfirm.addEventListener('input', checkPasswordMatch);

  // ------------- 최종 폼 제출 시 검증 -------------
  form.addEventListener('submit', function(event) {
    let hasError = false;

    // 약관 동의 최종 확인
    if (!agreeAll.checked) {
      if (agreeError) agreeError.style.display = 'block';
      hasError = true;
    }

    // 아이디 중복 여부 확인
    if (isIdTaken) {
      showUserIdError("아이디 중복 확인이 필요합니다.");
      hasError = true;
    }

    // 비밀번호 형식 및 일치 확인
    if (!checkPasswordFormat()) {
      password.focus();
      hasError = true;
    } else if (!checkPasswordMatch()) {
      passwordConfirm.focus();
      hasError = true;
    }

    // 오류가 하나라도 있으면 제출 중단
    if (hasError) {
      event.preventDefault();
    }
  });
});