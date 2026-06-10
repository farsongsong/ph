# 🧪 만능지시약 pH 예측기 (Streamlit)

용액 사진의 색을 자동으로 인식해 pH를 예측하는 웹앱입니다.
학습한 `ph_model.pkl`(RandomForest)을 그대로 사용하므로 노트북과 정확도가 동일합니다.

## 파일 구성
- `app.py` — 앱 본체
- `ph_model.pkl` — 학습된 모델 (이 폴더에 함께 두세요)
- `requirements.txt` — 필요한 라이브러리

## 1. 내 컴퓨터에서 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열립니다 (보통 http://localhost:8501).

## 2. 인터넷에 무료 배포 (Streamlit Community Cloud)

1. GitHub에 새 저장소를 만들고 `app.py`, `ph_model.pkl`, `requirements.txt` 세 파일을 올립니다.
2. https://share.streamlit.io 에 GitHub 계정으로 로그인합니다.
3. "New app" → 저장소와 `app.py`를 선택 → Deploy.
4. 몇 분 뒤 `https://...streamlit.app` 주소가 생기고, 누구나 접속할 수 있습니다.

## 기능 (3개 탭)
- **카메라 촬영**: 버튼을 눌러 사진을 찍으면 즉시 분석 (모바일·PC 모두 동작, 권장)
- **실시간 웹캠**: 영상에서 매 프레임 pH를 실시간 표시 (`streamlit-webrtc` 필요)
- **사진 업로드**: 저장된 사진(여러 장 가능) 분석 + 평균 계산

## 동작 방식
1. 사진에서 **채도가 높은(색이 있는) 영역 = 용액**을 자동으로 찾습니다. (배경 책상·벽은 무채색이라 제외)
2. 그 중심의 작은 원에서만 평균 색(RGB·HSV·LAB 9개 값)을 측정합니다.
3. 화이트밸런스 보정 후 RandomForest 모델로 pH를 예측합니다.

## 참고
- 예측은 학습에 쓴 만능지시약·조명 기준입니다. 다른 환경에서는 오차가 커질 수 있습니다.
- `scikit-learn`은 모델을 저장한 버전(1.6.1)으로 고정해두었습니다. 버전이 다르면 경고가 뜨거나 결과가 달라질 수 있습니다.
- 실시간 웹캠 설치가 번거로우면 "카메라 촬영" 탭을 쓰세요. 결과는 사실상 동일합니다.
