"""
AI MASTER MENTOR v6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
엔진: Groq API (무료, 하루 14,400회 × 3계정)
실행: streamlit run app.py --server.address=0.0.0.0 --server.port=8501

.env 파일 설정:
  GROQ_API_KEY_1=gsk_...
  GROQ_API_KEY_2=gsk_...
  GROQ_API_KEY_3=gsk_...

Streamlit Cloud Secrets:
  GROQ_API_KEY_1 = "gsk_..."
  GROQ_API_KEY_2 = "gsk_..."
  GROQ_API_KEY_3 = "gsk_..."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, time
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  ✏️  MASTER PROMPT — 페르소나를 바꾸려면 여기를 수정하세요
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
# Role & Identity
너는 전 세계의 모든 학문적 지식과 실무적 통찰을 융합하여 최적의 해답을 도출하는 '초지능형 마스터 멘토'이다.
너의 임무는 단순한 답변 제공을 넘어, 사용자의 지적 지평을 넓히고 비즈니스와 개인의 성장을 가속화하는 위대한 스승의 역할을 수행하는 것이다.

# Core Mission
1. 세계 최고 전문가의 지식 합성: 해당 분야 세계적 권위자의 시각으로 답하라. 표면적 정보가 아닌 이면의 원리와 최신 트렌드를 결합한 최고 수준의 통찰을 제공하라.
2. 지식의 확장 (Teacher Mode): 반드시 알아야 할 상위 개념, 연결된 심화 지식, 실무 적용 사례를 능동적으로 제공하라.
3. 선제적 정보 발굴: 이 질문과 관련해 다음으로 알아두면 좋은 3가지 지식을 항상 포함하라.

# Operating Principles
1. 3차원 전략: 세 가지 전략적 안을 구상한 뒤 최적 안을 논리적 근거와 함께 제안하라.
2. 냉철한 자기 비평: 논리적 약점을 스스로 점검하고 가장 완벽한 버전을 제출하라.
3. 데이터 & SEO 전문성: 데이터 분석 시 통계적 통찰을, 콘텐츠 작성 시 SEO 최적화 구조를 먼저 제안하라.

# Communication Style
- 전문성을 갖추되 배우고자 하는 이에게 친절하고 명쾌한 스승의 말투.
- 복잡한 개념은 비유(Analogy)로 설명하고 핵심은 표(Table)나 Markdown으로 시각화.
- 사용자 질문 언어를 기본으로, 글로벌 전문 용어는 병기.

# Interaction Guide
[직접 답변] → [심화 원리] → [연관 고급 지식 확장] → [스승의 Insight] → [다음 단계 제안]
"""

# ══════════════════════════════════════════════════════════════
#  GROQ 설정
# ══════════════════════════════════════════════════════════════

# 무료 모델 목록 (성능 순)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # 최강 성능
    "llama-3.1-8b-instant",      # 빠른 응답
    "mixtral-8x7b-32768",        # 긴 대화
    "gemma2-9b-it",              # 경량 백업
]

def load_groq_keys() -> list:
    """Streamlit Secrets → .env 순으로 키 로드"""
    keys = []
    try:
        for i in range(1, 4):
            k = st.secrets.get(f"GROQ_API_KEY_{i}", "").strip()
            if k:
                keys.append(k)
    except Exception:
        pass
    if not keys:
        for i in range(1, 4):
            k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
            if k:
                keys.append(k)
    return keys

GROQ_KEYS = load_groq_keys()

# ──────────────────────────────────────────────────────────────
#  핵심 수정: 키/모델 인덱스를 st.session_state에 저장
#  → rerun 후에도 전환된 키/모델이 유지됨
# ──────────────────────────────────────────────────────────────
def call_groq_with_rotation(messages: list) -> tuple[str, str]:
    """
    할당량 초과 시 key → model 순으로 자동 전환.
    rerun 후에도 전환 상태가 유지되도록 session_state 사용.
    Returns: (answer, error_message)
    """
    if not GROQ_KEYS:
        return "", "GROQ_API_KEY가 설정되지 않았습니다."

    total_keys   = len(GROQ_KEYS)
    total_models = len(GROQ_MODELS)
    total_tries  = total_keys * total_models

    for attempt in range(total_tries):
        ki = st.session_state.key_idx   % total_keys
        mi = st.session_state.model_idx % total_models

        current_key   = GROQ_KEYS[ki]
        current_model = GROQ_MODELS[mi]

        try:
            client = Groq(api_key=current_key)
            resp   = client.chat.completions.create(
                model      = current_model,
                messages   = messages,
                max_tokens = 4096,
                temperature= 0.7,
            )
            return resp.choices[0].message.content, ""

        except Exception as e:
            err = str(e)

            # 429 할당량 초과 → 다음 키 시도
            if "429" in err or "rate" in err.lower() or "quota" in err.lower():

                # 다음 키로 이동
                st.session_state.key_idx += 1

                # 모든 키 소진 → 다음 모델로 전환
                if st.session_state.key_idx % total_keys == 0:
                    st.session_state.model_idx += 1
                    next_model = GROQ_MODELS[st.session_state.model_idx % total_models]
                    st.toast(f"모델 전환 → {next_model}", icon="🔄")
                else:
                    next_key_num = (st.session_state.key_idx % total_keys) + 1
                    st.toast(f"KEY {next_key_num}로 전환 중...", icon="🔑")

                # 마지막 시도가 아니면 잠깐 대기 후 재시도
                if attempt < total_tries - 1:
                    time.sleep(1)
                    continue

            else:
                # 429 외 다른 오류 (인증 실패 등)
                return "", f"API 오류: {err}"

    # 모든 키/모델 소진
    return "", (
        "⏳ 모든 API Key의 분당 한도가 초과되었습니다.\n\n"
        "**1분 후 다시 시도해 주세요.**\n\n"
        "하루 한도가 소진된 경우 내일 자정(UTC)에 초기화됩니다."
    )

# ══════════════════════════════════════════════════════════════
#  STREAMLIT 앱
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Master Mentor",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

# ──────────────────────────────────────────────────────────────
#  CSS — THE ATELIER (Luxury Editorial Theme)
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:     #F7F4EF;
    --paper:  #EFEBE3;
    --dark:   #1A1714;
    --mid:    #4A4540;
    --soft:   #8C8680;
    --accent: #1B4D3E;
    --gold:   #C4955A;
    --line:   #D8D2C8;
    --white:  #FFFFFF;
    --shadow: rgba(26,23,20,0.08);
}

html, body { margin: 0; padding: 0; }

[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--dark) !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"]    { display: none !important; }
[data-testid="stMain"]          { background: transparent !important; }
[data-testid="block-container"] { max-width: 800px; padding-top: 0 !important; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 44px 20px 28px;
    border-bottom: 1.5px solid var(--line);
    margin-bottom: 30px;
}
.app-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
}
.app-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 48px;
    font-weight: 300;
    color: var(--dark);
    line-height: 1.1;
    margin: 0 0 8px;
}
.app-title em { font-style: italic; color: var(--accent); }
.app-sub {
    font-size: 13px;
    color: var(--soft);
    letter-spacing: 1.5px;
    margin-top: 4px;
}
.engine-badge {
    display: inline-block;
    margin-top: 16px;
    padding: 6px 18px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    background: #2D1B69;
    color: #E0D4FF;
}

/* ── Role Labels ── */
.role-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 22px 0 8px;
}
.role-label .ln { flex: 0; width: 36px; height: 1px; background: var(--line); }
.user-lbl { color: var(--gold);   justify-content: flex-end; }
.ai-lbl   { color: var(--accent); justify-content: flex-start; }

/* ── Bubbles ── */
.row-user { display: flex; justify-content: flex-end; }
.row-ai   { display: flex; justify-content: flex-start; }

.bubble {
    padding: 17px 22px;
    max-width: 86%;
    line-height: 1.80;
    font-size: 15px;
    word-break: break-word;
}
.bubble-user {
    background: var(--accent);
    color: #EEF7F2 !important;
    border-radius: 2px 16px 16px 16px;
    box-shadow: 0 4px 20px rgba(27,77,62,0.20);
}
.bubble-ai {
    background: var(--white);
    color: var(--dark) !important;
    border-radius: 16px 16px 16px 2px;
    border: 1px solid var(--line);
    box-shadow: 0 2px 18px var(--shadow);
}
.bubble-ai p      { color: var(--dark) !important; font-size:15px; line-height:1.85; }
.bubble-ai li     { color: var(--mid)  !important; font-size:15px; line-height:1.8; }
.bubble-ai strong { color: var(--dark) !important; font-weight:600; }
.bubble-ai em     { color: var(--accent); font-style:italic; }
.bubble-ai a      { color: var(--accent); text-underline-offset:3px; }
.bubble-ai h1, .bubble-ai h2, .bubble-ai h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--accent) !important;
    font-weight: 600;
    border-bottom: 1px solid var(--line);
    padding-bottom: 5px;
    margin-top: 20px;
}
.bubble-ai h1 { font-size:24px !important; }
.bubble-ai h2 { font-size:20px !important; }
.bubble-ai h3 { font-size:17px !important; }
.bubble-ai code {
    background: var(--paper) !important;
    color: var(--accent) !important;
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 13px;
}
.bubble-ai pre {
    background: #12100E !important;
    border-radius: 8px;
    padding: 16px 18px;
    overflow-x: auto;
    margin: 12px 0;
}
.bubble-ai pre code {
    background: transparent !important;
    color: #A8D8C0 !important;
    border: none; padding: 0;
}
.bubble-ai table { width:100%; border-collapse:collapse; margin:14px 0; font-size:14px; }
.bubble-ai th {
    background: var(--paper) !important;
    color: var(--accent) !important;
    font-size: 11px; font-weight:700;
    letter-spacing:1px; text-transform:uppercase;
    padding: 10px 14px;
    border: 1px solid var(--line);
}
.bubble-ai td {
    color: var(--dark) !important;
    padding: 9px 14px;
    border: 1px solid var(--line);
    vertical-align: top;
}
.bubble-ai tr:nth-child(even) td { background: rgba(239,235,227,0.55) !important; }
.bubble-ai blockquote {
    border-left: 3px solid var(--accent);
    padding: 4px 0 4px 16px;
    margin: 12px 0;
    color: var(--mid) !important;
    font-style: italic;
}

/* ── Mic Button ── */
div[data-testid="stButton"] > button {
    height: 54px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(27,77,62,0.22) !important;
    transition: all 0.22s ease !important;
    touch-action: manipulation;
}
div[data-testid="stButton"] > button:hover {
    background: #163D31 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(27,77,62,0.32) !important;
}
div[data-testid="stButton"] > button:active { transform: scale(0.98) !important; }

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    border-top: 1.5px solid var(--line) !important;
    background: var(--bg) !important;
    padding: 10px 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--white) !important;
    border: 1.5px solid var(--line) !important;
    border-radius: 2px !important;
    color: var(--dark) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    caret-color: var(--accent) !important;
    box-shadow: 0 2px 10px var(--shadow) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus  { border-color: var(--accent) !important; }
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--soft) !important;
    font-style: italic;
}
[data-testid="stChatInput"] button svg { fill: var(--accent) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #100E0C !important;
    border-right: 1px solid #242018 !important;
}
[data-testid="stSidebar"] * {
    color: rgba(240,235,225,0.82) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 20px !important;
    font-weight: 300 !important;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] hr { border-color: #2A2620 !important; }
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: rgba(240,235,225,0.82) !important;
    box-shadow: none !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    transform: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.11) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: var(--accent) !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
}

/* ── Divider ── */
.or-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 18px 0;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--soft);
}
.or-divider::before,
.or-divider::after { content:''; flex:1; height:1px; background:var(--line); }

/* ── Setup card ── */
.setup-card {
    background: var(--white);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 2px;
    padding: 24px 28px;
    margin: 16px 0;
    font-size: 14px;
    line-height: 1.9;
    color: var(--mid);
}
.setup-card h4 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    color: var(--accent);
    margin: 0 0 14px;
    font-weight: 600;
}
.setup-card code {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 12px;
    color: var(--accent);
}
.setup-card pre {
    background: #12100E;
    color: #A8D8C0;
    padding: 14px 16px;
    border-radius: 4px;
    font-size: 12px;
    line-height: 2;
    margin: 10px 0;
    overflow-x: auto;
}

.scroll-pad { height: 30px; }

@media (max-width: 640px) {
    .app-title { font-size: 32px; }
    .bubble    { font-size:14px; max-width:96%; padding:13px 15px; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  SESSION STATE 초기화
# ──────────────────────────────────────────────────────────────
defaults = {
    "messages":   [],  # [{"role":"user"|"assistant","content":"..."}]
    "key_idx":    0,   # 현재 사용 중인 Groq Key 인덱스 (rerun 후에도 유지)
    "model_idx":  0,   # 현재 사용 중인 모델 인덱스  (rerun 후에도 유지)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◆ 설정")
    st.markdown("---")

    total_keys = len(GROQ_KEYS)
    cur_ki     = st.session_state.key_idx   % max(total_keys, 1)
    cur_mi     = st.session_state.model_idx % len(GROQ_MODELS)
    cur_model  = GROQ_MODELS[cur_mi]

    # Key 상태 표시
    key_rows = ""
    for i in range(3):
        if i < total_keys:
            is_cur = (i == cur_ki) and total_keys > 0
            color  = "#5DBF8A" if is_cur else "#6B7070"
            marker = " ← 사용중" if is_cur else ""
            key_rows += f'<div style="font-size:12px;color:{color};margin:5px 0">● KEY {i+1}{marker}</div>'
        else:
            key_rows += f'<div style="font-size:12px;color:#3D3A37;margin:5px 0">○ KEY {i+1} (미등록)</div>'

    st.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:4px;padding:14px 16px;line-height:2">'
        f'<div style="font-size:9px;letter-spacing:3px;color:#555;margin-bottom:8px">API KEY 상태</div>'
        f'{key_rows}'
        f'<div style="margin-top:12px;font-size:9px;letter-spacing:2px;color:#555">현재 모델</div>'
        f'<div style="font-size:13px;color:#A8D8C0;font-weight:500;margin-top:4px">{cur_model}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("↺  대화 초기화", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.key_idx    = 0
        st.session_state.model_idx  = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:rgba(255,255,255,0.22);line-height:2.1">'
        '분당 한도 초과 시<br>Key → Model 자동 전환<br><br>'
        'Groq 무료: 분당 30회<br>3개 Key = 최대 90회/분</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
key_count = len(GROQ_KEYS)
badge_txt = f"⬡ Groq Cloud  ·  {key_count} / 3 Key Active" if key_count else "⚠ API Key 설정 필요"

st.markdown(f"""
<div class="app-header">
    <div class="app-eyebrow">Supreme Intelligence System</div>
    <h1 class="app-title">Master <em>Mentor</em></h1>
    <div class="app-sub">세계 최고 수준의 통찰 &nbsp;·&nbsp; Powered by Groq</div>
    <span class="engine-badge">{badge_txt}</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  API KEY 없을 때 안내
# ──────────────────────────────────────────────────────────────
if not GROQ_KEYS:
    st.markdown("""
<div class="setup-card">
<h4>🔧 Groq API Key를 설정해 주세요</h4>
<strong>Step 1.</strong> <a href="https://console.groq.com" target="_blank">console.groq.com</a> 접속 → 무료 가입<br>
<strong>Step 2.</strong> 왼쪽 메뉴 <code>API Keys</code> → <code>Create API Key</code> → 키 복사<br>
<strong>Step 3.</strong> 아래 중 하나로 등록:

<strong>로컬 실행 (.env 파일):</strong>
<pre>GROQ_API_KEY_1=gsk_여기에_키_입력
GROQ_API_KEY_2=gsk_여기에_키_입력
GROQ_API_KEY_3=gsk_여기에_키_입력</pre>

<strong>Streamlit Cloud (Secrets 탭):</strong>
<pre>GROQ_API_KEY_1 = "gsk_여기에_키_입력"
GROQ_API_KEY_2 = "gsk_여기에_키_입력"
GROQ_API_KEY_3 = "gsk_여기에_키_입력"</pre>

저장 후 앱을 재시작하면 바로 사용 가능합니다.
</div>
""", unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────
#  대화 기록 렌더링
# ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            '<div class="role-label user-lbl"><span class="ln"></span>✦ 나의 질문</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="row-user"><div class="bubble bubble-user">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="role-label ai-lbl">◈ 마스터 멘토<span class="ln"></span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="bubble bubble-ai">', unsafe_allow_html=True)
        st.markdown(msg["content"])
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="scroll-pad"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  메시지 처리
# ──────────────────────────────────────────────────────────────
def handle_message(user_text: str):
    if not user_text.strip():
        return

    # 사용자 버블 즉시 표시
    st.markdown(
        '<div class="role-label user-lbl"><span class="ln"></span>✦ 나의 질문</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="row-user"><div class="bubble bubble-user">{user_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Groq API 메시지 형식으로 변환
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})

    # 현재 모델명 표시
    cur_model = GROQ_MODELS[st.session_state.model_idx % len(GROQ_MODELS)]
    with st.spinner(f"◈  분석 중  ·  {cur_model}"):
        answer, error = call_groq_with_rotation(groq_messages)

    if error:
        answer = f"**⚠️ 오류**\n\n{error}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # AI 버블 표시
    st.markdown(
        '<div class="role-label ai-lbl">◈ 마스터 멘토<span class="ln"></span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="bubble bubble-ai">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  INPUT — 음성 + 텍스트
# ──────────────────────────────────────────────────────────────
_, col_c, _ = st.columns([1, 3, 1])
with col_c:
    voice = speech_to_text(
        language="ko",
        start_prompt="🎙  음성으로 질문하기",
        stop_prompt="⏹  녹음 중지",
        just_once=True,
        use_container_width=True,
        key="mic",
    )

if voice:
    handle_message(voice)
    st.rerun()

st.markdown('<div class="or-divider">or type below</div>', unsafe_allow_html=True)

user_input = st.chat_input("무엇이든 질문하세요  —  깊이 있는 통찰로 답변드립니다")
if user_input:
    handle_message(user_input)
    st.rerun()