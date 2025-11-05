# AI를 이용한 게임 불법 프로그램 탐지

Firebase에 모인 게임 플레이 영상을 입력으로 받아, 두 가지 컴퓨터 비전 파이프라인(Optical Flow 기반 에임 추적 + 색상/ROI 기반 헤드샷·명중 이벤트 감지)에서 특징을 추출하고, XGBoost로 플레이어가 AI인지 사람인지 분류합니다. SHAP으로 모델 예측 근거를 해석합니다.

## 관련 레포지토리
- KwangWick(직접 만든 게임): 사람/AI 플레이 영상과 메타데이터 수집
  - https://github.com/kaleido429/KwangWick
- ExtraVision: 기존 안티치트에 걸리지 않는, 게임 화면만 보고 작동하는 AI 핵 관련 실험(타 게임에 적용 및 악용 금지)
  - https://github.com/kaleido429/ExtraVision?tab=readme-ov-file

## 설계 요약 (Architecture Overview)
- 입력: Firebase Storage(`videos/*.webm`)
- 특징 추출:
  - Optical Flow(ROI 평균 흐름) → 에임 이동 시계열(aim_dx, aim_dy) → 속도/가속/저크/각도변화 등 1차 특징
  - 색상/ROI(사다리꼴 + 총구 박스, HSV 마스크) → 발사/명중/헤드샷/킬 이벤트 및 비율
- 윈도우 요약: WINDOW=150프레임(≈5초), STEP=30프레임(≈1초)로 슬라이딩 윈도우 통계(평균/중앙값/IQR/엔트로피/ZCR/상관계수 등)
- 학습/추론: XGBoost 분류 → `xgboost_cheater_detection_model.json`
- 해석: SHAP TreeExplainer로 전역/개별 중요도 시각화(Waterfall/Force/Summary/Bar/Interaction)

### 전체 흐름 (텍스트 다이어그램)
Firebase(Storage/Firestore)
→ 영상 다운로드
→ [분기1] Optical Flow로 에임 시계열 추출
→ [분기2] 색상/ROI 기반 발사·명중·헤드샷·킬 이벤트 추출
→ 시계열/이벤트를 슬라이딩 윈도우로 요약 통계화
→ XGBoost 학습/추론
→ SHAP 해석 및 시각화

### 구성 요소와 책임
- `Headshot_Tracker.ipynb`: Firebase 인증/초기화, 영상 다운로드, ROI/HSV 마스크로 이벤트 감지, 카운트 집계, (선택) 디버그 영상 저장
- `optical_flow_colab.ipynb`: Farnebäck Optical Flow로 에임 이동(aim_dx, aim_dy) 시계열, 1차 특징 생성  

https://github.com/Chunsaeng20, https://github.com/Chunsaeng20/Anti-Cheat-Model  
Led AI design and modeling. Most experiments/training were conducted on Colab Pro+ so they are not directly reflected in the this Git history.   
- `01_training_xgboost_model.ipynb`: 슬라이딩 윈도우 요약 통계(WINDOW=150, STEP=30), 상관계수(corr_*), 라벨 `cheat`, 모델 학습/저장
- `02_model_interpretation_with_shap.ipynb`: SHAP 기반 전역/개별 해석(여러 플롯)

### 실행 개요
1) `Headshot_Tracker.ipynb` → Storage/Firestore 초기화 → 샘플 영상 분석 → 카운트/디버그 영상(optional)
2) `optical_flow_colab.ipynb` → 에임 이동 시계열 및 1차 특징 추출
3) `01_training_xgboost_model.ipynb` → 윈도우 요약 통계 생성 → XGBoost 학습/저장
4) `02_model_interpretation_with_shap.ipynb` → SHAP 해석/시각화

## 윈도우 요약이란?
프레임 시계열을 일정 길이의 묶음(슬라이딩 윈도우)으로 잘라 각 묶음마다 평균·표준편차처럼 요약 통계를 계산해 학습에 쓰기 좋은 표 형태 특징으로 만드는 과정입니다.
- WINDOW_SIZE=150(≈5초), STEP_SIZE=30(≈1초) → 0~149, 30~179, 60~209 …처럼 80% 겹치며 진행
- 통계 예: mean/median/mode/range/std/IQR/max/min/skew/kurtosis/ZCR/entropy + 특징쌍 상관계수(corr_*)
- 윈도우 개수: floor((N - WINDOW_SIZE)/STEP_SIZE) + 1

## xAI를 추가한 이유
오탐(False Positive) 대응: 무고한 유저 제재 방지  
블랙박스 문제: AI가 정상 유저를 핵으로 오판 시, 운영팀은 "AI가 그렇게 판단했다" 외에 제재의 근거를 제시할 수 없습니다.  
이는 법적 분쟁 및 유저 커뮤니티 신뢰도 하락의 치명적인 원인이 됩니다.  

## 보안/주의
- `serviceAccountKey.json`은 반드시 `.gitignore`에 포함하고 커밋하지 마세요.
- 일부 WEBM은 FPS 메타가 비정상일 수 있어 상한/고정 로직이 포함되어 있습니다.
