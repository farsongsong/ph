"""
만능지시약 실시간 pH 예측 — Streamlit 앱
=========================================
학습한 ph_model.pkl 을 그대로 사용 (변환 없음 → 정확도 100% 유지).
용액을 색(채도)으로 자동 감지해 그 중심만 측정한다.

[실행]  streamlit run app.py
[모델]  같은 폴더에 ph_model.pkl 을 두세요.
"""
import streamlit as st
import numpy as np
import cv2
import pickle
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="pH 예측기", page_icon="🧪", layout="centered")

PH_MIN, PH_MAX = 2, 10
DEFAULT_FEATURES = ["H", "S", "V", "L", "A", "B_lab", "R", "G", "B"]


# ════════════════════════════════════════════════
#  모델 로드 (캐시)
# ════════════════════════════════════════════════
@st.cache_resource
def load_model(path="ph_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ════════════════════════════════════════════════
#  핵심 분석 함수 (노트북과 동일)
# ════════════════════════════════════════════════
def find_liquid_roi(img_bgr):
    """색(채도)으로 비커 속 용액을 찾아 (중심x, 중심y, 반지름) 반환."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    sat_thresh = max(40, np.percentile(S, 90) * 0.5)
    mask = ((S > sat_thresh) & (V > 40) & (V < 250)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return w // 2, h // 2, int(min(h, w) * 0.15), False
    big = max(cnts, key=cv2.contourArea)
    M = cv2.moments(big)
    if M["m00"] == 0:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    r = max(15, int(np.sqrt(cv2.contourArea(big) / np.pi) * 0.5))
    # 용액을 충분히 찾았는지 (너무 작으면 실패로 간주)
    found = cv2.contourArea(big) > (h * w * 0.001)
    return cx, cy, r, found


def extract_features(img_bgr):
    h, w = img_bgr.shape[:2]
    cx, cy, r, found = find_liquid_roi(img_bgr)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    ch = lambda s, i: cv2.mean(s[:, :, i], mask=mask)[0]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return dict(R=ch(rgb, 0), G=ch(rgb, 1), B=ch(rgb, 2),
                H=ch(hsv, 0), S=ch(hsv, 1), V=ch(hsv, 2),
                L=ch(lab, 0), A=ch(lab, 1), B_lab=ch(lab, 2),
                cx=cx, cy=cy, r=r, found=found)


def white_balance(img_bgr):
    img_f = img_bgr.astype(np.float32)
    means = [img_f[:, :, c].mean() for c in range(3)]
    gray = sum(means) / 3
    if min(means) > 0:
        for c, m in enumerate(means):
            img_f[:, :, c] = np.clip(img_f[:, :, c] * gray / m, 0, 255)
    return img_f.astype(np.uint8)


def predict_ph(img_bgr, bundle):
    use_wb = bundle.get("use_wb", True)
    feats = bundle.get("features", DEFAULT_FEATURES)
    feat = extract_features(white_balance(img_bgr) if use_wb else img_bgr)
    vec = np.array([[feat[k] for k in feats]])
    ph = float(np.clip(bundle["model"].predict(vec)[0], PH_MIN, PH_MAX))
    return ph, feat


PH_COLORS = {2: (0, 0, 220), 4: (0, 100, 210), 5: (0, 160, 200), 6: (0, 200, 170),
             7: (0, 210, 100), 8: (0, 200, 0), 9: (60, 180, 0), 10: (130, 100, 0)}


def ph_color_bgr(ph):
    key = int(np.clip(round(ph), PH_MIN, PH_MAX))
    return PH_COLORS[sorted(PH_COLORS, key=lambda k: abs(k - key))[0]]


def draw_overlay(frame, feat, ph):
    out = frame.copy()
    cx, cy, r = feat["cx"], feat["cy"], feat["r"]
    color = ph_color_bgr(ph)
    cv2.circle(out, (cx, cy), r, color, 3)
    cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)
    txt = f"pH {ph:.1f}"
    fs = max(0.8, out.shape[1] / 900)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
    ty = cy + r + th + 14
    cv2.rectangle(out, (cx - tw // 2 - 10, cy + r + 4), (cx + tw // 2 + 10, ty + 6), (0, 0, 0), -1)
    cv2.putText(out, txt, (cx - tw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 3)
    return out


def ph_label(ph):
    if ph < 4.5:   return "강산성", "#d32f2f"
    if ph < 6.5:   return "약산성", "#f0a030"
    if ph < 7.5:   return "중성",   "#26c281"
    if ph < 9:     return "약염기성", "#3b6ea5"
    return "강염기성", "#2c4f8a"


# ════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════
st.title("🧪 만능지시약 pH 예측기")
st.caption("용액 사진의 색을 자동으로 인식해 pH를 예측합니다. (학습된 RandomForest 모델 사용)")

# 모델 로드 — 폴더에 없으면 업로드 받기
bundle = None
model_path = Path("ph_model.pkl")
if model_path.exists():
    try:
        bundle = load_model(str(model_path))
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
else:
    up = st.file_uploader("먼저 모델 파일(ph_model.pkl)을 올려주세요", type=["pkl"])
    if up is not None:
        bundle = pickle.loads(up.read())

if bundle is None:
    st.info("ph_model.pkl 을 앱과 같은 폴더에 두거나 위에서 업로드하면 시작됩니다.")
    st.stop()


def run_analysis(img_bgr, source_label=""):
    """이미지 분석 후 결과 표시. 여러 번 호출 가능."""
    ph, feat = predict_ph(img_bgr, bundle)
    if not feat["found"]:
        st.warning("⚠️ 색이 뚜렷한 용액을 찾지 못했습니다. 비커가 화면에 크게 보이도록 다시 찍어보세요.")
    label, hexc = ph_label(ph)
    overlay = draw_overlay(img_bgr, feat, ph)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=f"감지된 용액 영역 {source_label}", use_container_width=True)
    with col2:
        st.markdown(
            f"<div style='text-align:center;padding:18px;border-radius:14px;"
            f"background:{hexc}22;border:2px solid {hexc}'>"
            f"<div style='font-size:0.9rem;color:#666'>예측 pH</div>"
            f"<div style='font-size:3.2rem;font-weight:800;color:{hexc};line-height:1'>{ph:.1f}</div>"
            f"<div style='font-size:1.1rem;color:{hexc};font-weight:600'>{label}</div></div>",
            unsafe_allow_html=True)
        # 측정된 색 미리보기
        rgb = (int(feat['R']), int(feat['G']), int(feat['B']))
        st.markdown(
            f"<div style='margin-top:10px;text-align:center'>"
            f"<div style='font-size:0.8rem;color:#666'>측정된 용액 색</div>"
            f"<div style='display:inline-block;width:60px;height:24px;border-radius:6px;"
            f"border:1px solid #ccc;background:rgb{rgb}'></div>"
            f"<div style='font-size:0.75rem;color:#888'>RGB{rgb}</div></div>",
            unsafe_allow_html=True)
    return ph, feat


tab1, tab3 = st.tabs(["📷 카메라 촬영", "📁 사진 업로드"])

# ── 탭 1: 카메라 촬영 (후면 카메라 강제) ──
with tab1:
    st.write("카메라로 비커를 비추고 사진을 찍으면 바로 분석합니다.")

    cam_pref = st.radio("카메라", ["후면", "전면"], horizontal=True, key="cam_pref")
    facing = "environment" if cam_pref == "후면" else "user"

    # Streamlit 기본 카메라 위젯(촬영 버튼·크기 그대로)에 후면 카메라를 강제 적용.
    # 위젯이 video 스트림을 만들 때, 원하는 facingMode로 다시 잡아 교체한다.
    st.components.v1.html(f"""
    <script>
    (function() {{
      const want = "{facing}";
      const doc = window.parent.document;

      async function applyFacing() {{
        const video = doc.querySelector('video');
        if (!video) return false;
        try {{
          const newStream = await navigator.mediaDevices.getUserMedia({{
            video: {{ facingMode: {{ ideal: want }} }}, audio: false
          }});
          // 기존 스트림 정지 후 교체
          if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
          video.srcObject = newStream;
          await video.play().catch(()=>{{}});
          return true;
        }} catch (e) {{ return false; }}
      }}

      // 위젯이 늦게 뜰 수 있으니 잠깐 동안 반복 시도
      let tries = 0;
      const timer = setInterval(async () => {{
        tries++;
        const ok = await applyFacing();
        if (ok || tries > 20) clearInterval(timer);
      }}, 400);
    }})();
    </script>
    """, height=0)

    shot = st.camera_input("사진 촬영", label_visibility="collapsed")
    if shot is not None:
        arr = np.frombuffer(shot.getvalue(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            run_analysis(img, "(촬영)")

# ── 탭 2: 업로드 ──
with tab3:
    st.write("저장된 사진 파일을 올려 분석합니다. 여러 장을 한 번에 올리면 평균도 계산합니다.")
    files = st.file_uploader("이미지 업로드 (jpg/png)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if files:
        phs = []
        for f in files:
            arr = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                st.warning(f"{f.name}: 읽기 실패")
                continue
            st.markdown(f"**{f.name}**")
            ph, _ = run_analysis(img, f"({f.name})")
            phs.append(ph)
            st.divider()
        if len(phs) > 1:
            st.success(f"📊 {len(phs)}장 평균 예측 pH: **{np.mean(phs):.2f}** "
                       f"(표준편차 {np.std(phs):.2f})")

st.divider()
st.caption("색 → pH 예측은 학습 데이터(만능지시약·특정 조명) 기준입니다. "
           "다른 환경에서는 오차가 커질 수 있습니다.")
