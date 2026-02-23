"""
AI MASTER MENTOR v3  ·  "The Atelier" Edition
────────────────────────────────────────────────────────────
실행: streamlit run app.py --server.address=0.0.0.0 --server.port=8501

.env 파일 설정:
  GEMINI_API_KEY_1=AIza...
  GEMINI_API_KEY_2=AIza...
  GEMINI_API_KEY_3=AIza...
"""

import os, time
import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import speech_to_text
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  MASTER PROMPT
# ══════════════════════════════════════════════════════════════
SYSTEM_INSTRUCTION = """
# Role & Identity
너는 전 세계의 모든 학문적 지식과 실무적 통찰을 융합하여 최적의 해답을 도출하는 '초지능형 마스터 멘토'이다.
너의 임무는 단순한 답변 제공을 넘어, 사용자의 지적 지평을 넓히고 비즈니스와 개인의 성장을 가속화하는 '위대한 스승'의 역할을 수행하는 것이다.

# Core Mission
1. **세계 최고 전문가의 지식 합성**: 해당 분야의 세계적 권위자라면 어떻게 답할지 시뮬레이션하라. 표면적 정보가 아닌, 이면의 원리와 최신 트렌드를 결합한 '최고 수준의 통찰'을 제공하라.
2. **지식의 확장 (Teacher Mode)**: 반드시 알아야 할 상위 개념, 연결된 심화 지식, 실무 적용 사례를 능동적으로 제공하라.
3. **선제적 정보 발굴**: "이 질문과 관련해 다음으로 알아두면 좋은 3가지 지식"을 항상 포함하라.

# Operating Principles
1. **3차원 전략**: 세 가지 전략적 안을 구상한 뒤 최적 안을 논리적 근거와 함께 제안하라.
2. **냉철한 자기 비평**: 최종 답변 전 스스로 논리적 약점을 점검하고 완벽한 버전을 제출하라.
3. **데이터 & SEO 전문성**: 데이터 분석 시 통계적 통찰을, 콘텐츠 작성 시 SEO 최적화 구조를 먼저 제안하라.

# Communication Style
- 전문성을 갖추되 배우고자 하는 이에게 친절하고 명쾌한 스승의 말투.
- 복잡한 개념은 비유(Analogy)로 설명하고, 핵심은 표(Table)나 Markdown으로 시각화.
- 사용자 질문 언어를 기본으로, 글로벌 전문 용어는 병기.

# Interaction Guide
[직접 답변] → [심화 원리] → [연관 고급 지식 확장] → [스승의 Insight] → [다음 단계 제안]
"""

# ── API Keys & Model 설정 ─────────────────────────────────────
API_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY_1", "").strip(),
        os.getenv("GEMINI_API_KEY_2", "").strip(),
        os.getenv("GEMINI_API_KEY_3", "").strip(),
    ] if k
]

# 모델 우선순위 (무료 티어 안정 → 최신 순)
MODEL_FALLBACK = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Master Mentor",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
#  "THE ATELIER" — LUXURY EDITORIAL DESIGN
#  컨셉: 파리 최고급 컨설팅 펌 + 일본 미니멀리즘
#  크림 화이트 배경 · 딥 챠콜 텍스트 · 에메랄드 포인트
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ─── CSS 변수 ─────────────────────────────────────────── */
:root {
    --cream:   #F7F4EF;
    --paper:   #EFEBE3;
    --dark:    #1A1714;
    --mid:     #4A4540;
    --soft:    #8C8680;
    --accent:  #1B4D3E;   /* 딥 에메랄드 */
    --accent2: #C4955A;   /* 따뜻한 골드 */
    --line:    #D8D2C8;
    --shadow:  rgba(26,23,20,0.10);
}

/* ─── 전체 배경 & 기본 폰트 ────────────────────────────── */
html, body { margin: 0; padding: 0; }

[data-testid="stAppViewContainer"] {
    background-color: var(--cream) !important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    font-family: 'DM Sans', sans-serif !important;
    color: var(--dark) !important;
    min-height: 100vh;
}

[data-testid="stHeader"]       { background: transparent !important; }
[data-testid="stMain"]         { background: transparent !important; }
[data-testid="block-container"] { max-width: 780px; padding-top: 0 !important; }

/* ─── 헤더 블록 ────────────────────────────────────────── */
.atelier-header {
    text-align: center;
    padding: 40px 20px 28px;
    border-bottom: 1.5px solid var(--line);
    margin-bottom: 32px;
    position: relative;
}
.atelier-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
}
.atelier-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 44px;
    font-weight: 300;
    letter-spacing: -0.5px;
    color: var(--dark);
    line-height: 1.1;
    margin: 0 0 8px;
}
.atelier-title span {
    font-style: italic;
    color: var(--accent);
}
.atelier-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 400;
    color: var(--soft);
    letter-spacing: 1px;
    margin-top: 6px;
}
.atelier-pill {
    display: inline-block;
    background: var(--accent);
    color: #fff;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 2px;
    margin-top: 14px;
}

/* ─── 역할 레이블 ──────────────────────────────────────── */
.role-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 24px 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.role-label::after  { content: ''; flex: 1; height: 1px; background: var(--line); }
.role-label::before { content: ''; flex: 1; height: 1px; background: var(--line); }

.user-label {
    color: var(--accent2);
    flex-direction: row-reverse;
    justify-content: flex-start;
}
.user-label::after  { display: none; }
.user-label::before { flex: 0; width: 48px; }

.ai-label {
    color: var(--accent);
    justify-content: flex-start;
}
.ai-label::before { display: none; }
.ai-label::after  { flex: 0; width: 48px; }

/* ─── 말풍선 ────────────────────────────────────────────── */
.bubble-wrap-user { display: flex; justify-content: flex-end; }
.bubble-wrap-ai   { display: flex; justify-content: flex-start; }

.chat-bubble {
    padding: 18px 22px;
    max-width: 84%;
    line-height: 1.80;
    font-size: 15px;
    word-break: break-word;
    position: relative;
}

/* 사용자 버블 — 에메랄드 */
.user-bubble {
    background: var(--accent);
    color: #F0F7F4 !important;
    border-radius: 2px 16px 16px 16px;
    box-shadow: 0 4px 24px rgba(27,77,62,0.22);
}

/* AI 버블 — 따뜻한 화이트 카드 */
.ai-bubble {
    background: #FFFFFF;
    color: var(--dark) !important;
    border-radius: 16px 16px 16px 2px;
    border: 1px solid var(--line);
    box-shadow: 0 2px 20px var(--shadow), 0 1px 4px rgba(0,0,0,0.04);
}

/* AI 버블 내부 텍스트 스타일 */
.ai-bubble p     { color: var(--dark) !important; font-size: 15px; line-height: 1.8; }
.ai-bubble li    { color: var(--mid)  !important; font-size: 15px; line-height: 1.8; }
.ai-bubble strong{ color: var(--dark) !important; font-weight: 600; }
.ai-bubble em    { color: var(--accent); font-style: italic; }

.ai-bubble h1, .ai-bubble h2, .ai-bubble h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--accent) !important;
    font-weight: 600;
    letter-spacing: -0.3px;
    border-bottom: 1px solid var(--line);
    padding-bottom: 6px;
    margin-top: 20px;
}
.ai-bubble h1 { font-size: 24px !important; }
.ai-bubble h2 { font-size: 20px !important; }
.ai-bubble h3 { font-size: 17px !important; }

.ai-bubble code {
    background: var(--paper) !important;
    color: var(--accent) !important;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 13px;
    border: 1px solid var(--line);
}
.ai-bubble pre {
    background: var(--dark) !important;
    border-radius: 10px;
    padding: 16px;
    overflow-x: auto;
}
.ai-bubble pre code {
    background: transparent !important;
    color: #A8D8C0 !important;
    border: none;
    padding: 0;
    font-size: 13px;
}
.ai-bubble table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 14px;
}
.ai-bubble th {
    background: var(--paper) !important;
    color: var(--accent) !important;
    font-weight: 600;
    padding: 10px 14px;
    border: 1px solid var(--line);
    text-align: left;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.ai-bubble td {
    color: var(--dark) !important;
    padding: 9px 14px;
    border: 1px solid var(--line);
    vertical-align: top;
}
.ai-bubble tr:nth-child(even) td {
    background: rgba(239,235,227,0.4) !important;
}
.ai-bubble blockquote {
    border-left: 3px solid var(--accent);
    padding-left: 16px;
    margin: 12px 0;
    color: var(--mid) !important;
    font-style: italic;
}

/* ─── 마이크 버튼 ──────────────────────────────────────── */
div[data-testid="stButton"] > button {
    height: 56px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(27,77,62,0.25) !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    touch-action: manipulation;
    position: relative;
}
div[data-testid="stButton"] > button:hover {
    background: #163D31 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(27,77,62,0.35) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ─── Chat Input ────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-top: 1.5px solid var(--line) !important;
    background: var(--cream) !important;
    padding: 12px 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    border: 1.5px solid var(--line) !important;
    border-radius: 2px !important;
    color: var(--dark) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    caret-color: var(--accent) !important;
    box-shadow: 0 2px 12px var(--shadow) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 2px 16px rgba(27,77,62,0.12) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--soft) !important;
    font-style: italic;
}
[data-testid="stChatInput"] button svg { fill: var(--accent) !important; }

/* ─── 사이드바 ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--dark) !important;
    border-right: 1px solid #2D2A27 !important;
}
[data-testid="stSidebar"] * {
    color: rgba(240,235,227,0.85) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 20px !important;
    font-weight: 300 !important;
    color: #F0EBE3 !important;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #F0EBE3 !important;
    border-radius: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(240,235,227,0.85) !important;
    box-shadow: none !important;
    letter-spacing: 1.5px !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.12) !important;
    transform: none !important;
}

/* ─── 스피너 ────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-color: var(--accent) !important; }
[data-testid="stSpinner"] p    { color: var(--accent) !important; font-size: 13px !important; letter-spacing: 1px; }

/* ─── 구분선 ────────────────────────────────────────────── */
.elegant-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 20px 0;
    color: var(--soft);
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
}
.elegant-divider::before,
.elegant-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line);
}

/* ─── 알림 박스 ─────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(27,77,62,0.07) !important;
    border: 1px solid rgba(27,77,62,0.2) !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--dark) !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAlert"] p { color: var(--dark) !important; }

/* ─── 토스트 ────────────────────────────────────────────── */
[data-testid="stToast"] {
    background: var(--dark) !important;
    color: #F0EBE3 !important;
    border-radius: 2px !important;
}

/* ─── 스크롤 여백 ───────────────────────────────────────── */
.scroll-pad { height: 32px; }

/* ─── 에러 카드 ─────────────────────────────────────────── */
.error-card {
    background: #FFF8F0;
    border: 1px solid #F5C896;
    border-left: 3px solid #C4955A;
    border-radius: 2px;
    padding: 16px 20px;
    margin: 8px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: var(--dark);
}
.error-card strong { color: #8B5E2A; }

/* ─── 모바일 ─────────────────────────────────────────────── */
@media (max-width: 640px) {
    .atelier-title { font-size: 30px; }
    .chat-bubble   { font-size: 14px; max-width: 96%; padding: 14px 16px; }
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "chat"        not in st.session_state: st.session_state.chat        = None
if "key_index"   not in st.session_state: st.session_state.key_index   = 0
if "model_index" not in st.session_state: st.session_state.model_index = 0


# ── Gemini 세션 생성 ──────────────────────────────────────────
def get_chat(reset: bool = False):
    if reset:
        st.session_state.chat = None
    if st.session_state.chat is None:
        if not API_KEYS:
            return None
        key   = API_KEYS[st.session_state.key_index % len(API_KEYS)]
        model = MODEL_FALLBACK[st.session_state.model_index % len(MODEL_FALLBACK)]
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_INSTRUCTION)
        history = [{"role": x["role"], "parts": x["parts"]} for x in st.session_state.messages]
        st.session_state.chat = m.start_chat(history=history)
    return st.session_state.chat


# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◆ 설정")
    st.markdown("---")

    active = len(API_KEYS)
    cur_model = MODEL_FALLBACK[st.session_state.model_index % len(MODEL_FALLBACK)]

    status_rows = ""
    for i in range(3):
        icon = "●" if i < active else "○"
        color = "color:#4CAF7D" if i < active else "color:#6B6560"
        cur = " ← 사용중" if (active > 0 and i == st.session_state.key_index % active) else ""
        status_rows += f'<div style="font-size:13px;margin:4px 0;{color}">{icon} KEY {i+1}{cur}</div>'

    st.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);'
        f'border-radius:2px;padding:14px 16px">'
        f'<div style="font-size:10px;letter-spacing:2px;color:#888;margin-bottom:10px">API KEY 상태</div>'
        f'{status_rows}'
        f'<div style="margin-top:12px;font-size:10px;letter-spacing:1px;color:#888">MODEL</div>'
        f'<div style="font-size:13px;color:#A8D8C0;margin-top:4px">{cur_model}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("↺  대화 초기화", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.chat        = None
        st.session_state.key_index   = 0
        st.session_state.model_index = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:12px;color:rgba(255,255,255,0.3);line-height:1.8">'
        '.env 파일에서<br>API Key를 설정하세요.<br><br>'
        '할당량 초과 시<br>자동으로 다음 Key &<br>모델로 전환됩니다.</div>',
        unsafe_allow_html=True,
    )


# ── 헤더 ──────────────────────────────────────────────────────
active_count = len(API_KEYS)
pill_text = f"● {active_count} / 3  API KEY ACTIVE" if active_count else "○ API KEY 필요"
st.markdown(f"""
<div class="atelier-header">
    <div class="atelier-eyebrow">Supreme Intelligence System</div>
    <h1 class="atelier-title">Master <span>Mentor</span></h1>
    <div class="atelier-sub">Powered by Google Gemini  ·  세계 최고 수준의 통찰</div>
    <div class="atelier-pill">{pill_text}</div>
</div>
""", unsafe_allow_html=True)

# ── API Key 없을 때 ───────────────────────────────────────────
if not API_KEYS:
    st.markdown("""
<div style="background:#fff;border:1px solid #D8D2C8;border-left:3px solid #1B4D3E;
            border-radius:2px;padding:28px;margin:20px 0;font-family:'DM Sans',sans-serif">
<div style="font-size:10px;letter-spacing:3px;color:#1B4D3E;font-weight:600;margin-bottom:10px">
    SETUP REQUIRED
</div>
<div style="font-size:18px;font-family:'Cormorant Garamond',serif;color:#1A1714;margin-bottom:14px">
    API Key를 설정해 주세요
</div>
<p style="color:#4A4540;font-size:14px;line-height:1.8;margin-bottom:16px">
프로젝트 폴더에 <code style="background:#F7F4EF;padding:2px 7px;border-radius:2px;
border:1px solid #D8D2C8;color:#1B4D3E">.env</code> 파일을 만들고 아래 내용을 입력하세요.
</p>
<pre style="background:#1A1714;color:#A8D8C0;padding:18px;border-radius:2px;font-size:13px;line-height:2">GEMINI_API_KEY_1=여기에_첫번째_키_입력
GEMINI_API_KEY_2=여기에_두번째_키_입력
GEMINI_API_KEY_3=여기에_세번째_키_입력</pre>
<p style="color:#8C8680;font-size:13px;margin-top:14px">
→ <a href="https://aistudio.google.com/app/apikey" target="_blank" 
     style="color:#1B4D3E;font-weight:600">Google AI Studio</a>에서 무료 발급
</p>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── 대화 기록 렌더링 ──────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown('<div class="role-label user-label">✦ 나의 질문</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="bubble-wrap-user">'
            f'<div class="chat-bubble user-bubble">{msg["parts"][0]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="role-label ai-label">◈ 마스터 멘토</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="chat-bubble ai-bubble">', unsafe_allow_html=True)
            st.markdown(msg["parts"][0])
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="scroll-pad"></div>', unsafe_allow_html=True)


# ── 메시지 처리 — 자동 Key/Model 순환 + Retry ────────────────
def process_message(user_text: str):
    if not user_text.strip():
        return

    # 사용자 버블
    st.session_state.messages.append({"role": "user", "parts": [user_text]})
    st.markdown('<div class="role-label user-label">✦ 나의 질문</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="bubble-wrap-user">'
        f'<div class="chat-bubble user-bubble">{user_text}</div></div>',
        unsafe_allow_html=True,
    )

    answer      = None
    total_tries = len(API_KEYS) * len(MODEL_FALLBACK)

    for attempt in range(total_tries):
        try:
            chat = get_chat(reset=(attempt > 0))
            cur_model = MODEL_FALLBACK[st.session_state.model_index % len(MODEL_FALLBACK)]
            with st.spinner(f"◈  분석 중  ·  {cur_model}"):
                response = chat.send_message(user_text)
                answer   = response.text
            break  # 성공

        except Exception as e:
            err_str = str(e)

            # 429 할당량 초과 → 다음 Key 시도, 모든 Key 소진 시 다음 Model
            if "429" in err_str or "quota" in err_str.lower():
                next_key_idx = (st.session_state.key_index + 1) % max(len(API_KEYS), 1)

                if next_key_idx > st.session_state.key_index or attempt > 0:
                    # Key를 한 바퀴 돌았으면 모델 변경
                    if (attempt + 1) % max(len(API_KEYS), 1) == 0:
                        st.session_state.model_index += 1
                        new_model = MODEL_FALLBACK[st.session_state.model_index % len(MODEL_FALLBACK)]
                        st.toast(f"모델 전환 → {new_model}", icon="🔄")

                st.session_state.key_index = next_key_idx
                st.session_state.chat = None

                # 재시도 전 짧은 대기 (마지막 시도가 아닐 때)
                if attempt < total_tries - 1:
                    time.sleep(2)
                    continue

            # 마지막 시도도 실패
            if attempt == total_tries - 1:
                # 재시도 대기 시간 파싱
                wait = "잠시"
                import re
                m = re.search(r'retry.*?(\d+)', err_str, re.IGNORECASE)
                if m:
                    wait = f"{m.group(1)}초"

                answer = (
                    f"**⏳ 할당량 초과 — {wait} 후 다시 시도해 주세요**\n\n"
                    f"모든 API Key와 모델의 무료 할당량이 소진되었습니다.\n\n"
                    f"**해결 방법**\n"
                    f"- {wait} 기다린 후 다시 질문\n"
                    f"- .env 파일에 추가 API Key 등록\n"
                    f"- [Google AI Studio](https://aistudio.google.com/app/apikey)에서 새 Key 발급\n"
                    f"- 유료 플랜 업그레이드 시 제한 없음"
                )

    # AI 응답 버블
    st.session_state.messages.append({"role": "model", "parts": [answer]})
    st.markdown('<div class="role-label ai-label">◈ 마스터 멘토</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-bubble ai-bubble">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)


# ── 음성 입력 ─────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    voice_text = speech_to_text(
        language="ko",
        start_prompt="🎙  음성으로 질문하기",
        stop_prompt="⏹  녹음 중지",
        just_once=True,
        use_container_width=True,
        key="mic_input",
    )

if voice_text:
    process_message(voice_text)
    st.rerun()

# 우아한 구분선
st.markdown(
    '<div class="elegant-divider">or type below</div>',
    unsafe_allow_html=True,
)

# ── 텍스트 입력 ───────────────────────────────────────────────
user_input = st.chat_input("무엇이든 질문하세요  —  깊이 있는 통찰로 답변드립니다")
if user_input:
    process_message(user_input)
    st.rerun()