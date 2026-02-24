"""
AI MASTER MENTOR v7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
엔진: OpenRouter (무료 최강 모델 — DeepSeek R1 등)
실행: streamlit run app.py --server.address=0.0.0.0 --server.port=8501

.env 파일 설정:
  OPENROUTER_API_KEY_1=sk-or-...
  OPENROUTER_API_KEY_2=sk-or-...
  OPENROUTER_API_KEY_3=sk-or-...

Streamlit Cloud Secrets:
  OPENROUTER_API_KEY_1 = "sk-or-..."
  OPENROUTER_API_KEY_2 = "sk-or-..."
  OPENROUTER_API_KEY_3 = "sk-or-..."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, time
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from dotenv import load_dotenv
from openai import OpenAI

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
#  OPENROUTER 설정
#  무료 모델 목록 — :free 태그 = 크레딧 차감 없음
# ══════════════════════════════════════════════════════════════
OPENROUTER_MODELS = [
    "deepseek/deepseek-r1:free",              # 최강 추론 (GPT-4o 급)
    "deepseek/deepseek-chat-v3-0324:free",    # 빠르고 스마트
    "meta-llama/llama-3.3-70b-instruct:free", # Meta 최강 무료
    "google/gemma-3-27b-it:free",             # Google 무료
    "mistralai/mistral-7b-instruct:free",     # 빠른 백업
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def load_api_keys() -> list:
    """Streamlit Secrets → .env 순서로 키 로드"""
    keys = []
    try:
        for i in range(1, 4):
            k = st.secrets.get(f"OPENROUTER_API_KEY_{i}", "").strip()
            if k:
                keys.append(k)
    except Exception:
        pass
    if not keys:
        for i in range(1, 4):
            k = os.getenv(f"OPENROUTER_API_KEY_{i}", "").strip()
            if k:
                keys.append(k)
    return keys

API_KEYS = load_api_keys()


def call_openrouter(messages: list) -> tuple[str, str]:
    """
    OpenRouter API 호출 + 자동 Key/Model 로테이션.
    session_state의 key_idx, model_idx를 직접 수정하여
    st.rerun() 후에도 전환 상태가 유지됨.
    Returns: (answer, error)
    """
    if not API_KEYS:
        return "", "API Key가 설정되지 않았습니다."

    total_keys   = len(API_KEYS)
    total_models = len(OPENROUTER_MODELS)
    total_tries  = total_keys * total_models

    for attempt in range(total_tries):
        ki = st.session_state.key_idx   % total_keys
        mi = st.session_state.model_idx % total_models

        current_key   = API_KEYS[ki]
        current_model = OPENROUTER_MODELS[mi]

        try:
            client = OpenAI(
                api_key  = current_key,
                base_url = OPENROUTER_BASE_URL,
            )
            resp = client.chat.completions.create(
                model      = current_model,
                messages   = messages,
                max_tokens = 4096,
                temperature= 0.7,
                extra_headers={
                    "HTTP-Referer": "https://ai-master-mentor.streamlit.app",
                    "X-Title": "AI Master Mentor",
                },
            )
            return resp.choices[0].message.content, ""

        except Exception as e:
            err = str(e)

            # 한도 초과 또는 모델 오류 → 다음 키/모델로 전환
            if any(code in err for code in ["429", "402", "503", "overloaded", "rate"]):
                # 다음 키로
                st.session_state.key_idx += 1

                # 모든 키 소진 시 다음 모델로
                if st.session_state.key_idx % total_keys == 0:
                    st.session_state.model_idx += 1
                    next_mi    = st.session_state.model_idx % total_models
                    next_model = OPENROUTER_MODELS[next_mi].split("/")[-1]
                    st.toast(f"모델 전환 → {next_model}", icon="🔄")
                else:
                    next_ki = st.session_state.key_idx % total_keys
                    st.toast(f"KEY {next_ki + 1}로 전환 중...", icon="🔑")

                if attempt < total_tries - 1:
                    time.sleep(0.5)
                    continue
            else:
                # 인증 오류 등 복구 불가 에러
                return "", f"API 오류 ({current_model.split('/')[-1]}): {err[:200]}"

    return "", (
        "⏳ **모든 Key와 모델의 한도가 초과되었습니다.**\n\n"
        "잠시 후 다시 시도해 주세요.\n"
        "또는 [OpenRouter](https://openrouter.ai/keys)에서 새 Key를 발급해 추가하세요."
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
#  CSS — THE ATELIER THEME
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

html, body { margin:0; padding:0; }

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
    background: #1A3A5C;
    color: #C8DEFF;
}
.engine-badge.warn { background: #5C1A1A; color: #FFC8C8; }

/* ── Model tag ── */
.model-tag {
    display: inline-block;
    background: rgba(27,77,62,0.10);
    border: 1px solid rgba(27,77,62,0.20);
    color: var(--accent);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 2px;
    margin-left: 6px;
    vertical-align: middle;
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

/* ── Chat Bubbles ── */
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

/* AI 버블 내부 텍스트 */
.bubble-ai p      { color: var(--dark) !important; font-size:15px; line-height:1.85; }
.bubble-ai li     { color: var(--mid)  !important; font-size:15px; line-height:1.8; }
.bubble-ai strong { color: var(--dark) !important; font-weight:600; }
.bubble-ai em     { color: var(--accent); }
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
    border: none;
    padding: 0;
}
.bubble-ai table { width:100%; border-collapse:collapse; margin:14px 0; font-size:14px; }
.bubble-ai th {
    background: var(--paper) !important;
    color: var(--accent) !important;
    font-size:11px; font-weight:700;
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
.bubble-ai tr:nth-child(even) td { background:rgba(239,235,227,0.55) !important; }
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
div[data-testid="stButton"] > button:active { transform:scale(0.98) !important; }

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
[data-testid="stChatInput"] textarea::placeholder { color:var(--soft) !important; font-style:italic; }
[data-testid="stChatInput"] button svg { fill: var(--accent) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0E0C0A !important;
    border-right: 1px solid #222018 !important;
}
[data-testid="stSidebar"] * {
    color: rgba(240,235,225,0.80) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 20px !important;
    font-weight: 300 !important;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] hr { border-color: #262218 !important; }
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: rgba(240,235,225,0.80) !important;
    box-shadow: none !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    transform: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.10) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: var(--accent) !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
}

/* ── OR Divider ── */
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

/* ── Setup Card ── */
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
    "messages":   [],
    "key_idx":    0,   # rerun 후에도 유지
    "model_idx":  0,   # rerun 후에도 유지
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

    total_keys = len(API_KEYS)
    cur_ki     = st.session_state.key_idx   % max(total_keys, 1)
    cur_mi     = st.session_state.model_idx % len(OPENROUTER_MODELS)
    cur_model  = OPENROUTER_MODELS[cur_mi].split("/")[-1].replace(":free", "")

    # Key 상태
    key_rows = ""
    for i in range(3):
        if i < total_keys:
            is_cur = (i == cur_ki)
            color  = "#5DBF8A" if is_cur else "#5A6060"
            mark   = " ← 사용중" if is_cur else ""
            key_rows += f'<div style="font-size:12px;color:{color};margin:5px 0">● KEY {i+1}{mark}</div>'
        else:
            key_rows += f'<div style="font-size:12px;color:#333;margin:5px 0">○ KEY {i+1} (미등록)</div>'

    st.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:4px;padding:14px 16px;line-height:2">'
        f'<div style="font-size:9px;letter-spacing:3px;color:#555;margin-bottom:8px">API KEY 상태</div>'
        f'{key_rows}'
        f'<div style="margin-top:12px;font-size:9px;letter-spacing:2px;color:#555">현재 모델</div>'
        f'<div style="font-size:12px;color:#A8D8C0;font-weight:500;margin-top:4px">{cur_model}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("↺  대화 초기화", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.key_idx   = 0
        st.session_state.model_idx = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:rgba(255,255,255,0.20);line-height:2.2">'
        'DeepSeek R1 = GPT-4o 급<br>'
        '무료 모델 = 크레딧 차감 없음<br><br>'
        '한도 초과 시<br>Key → Model 자동 전환</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
key_count = len(API_KEYS)
if key_count:
    cur_model_short = OPENROUTER_MODELS[
        st.session_state.model_idx % len(OPENROUTER_MODELS)
    ].split("/")[-1].replace(":free", "")
    badge = f'<span class="engine-badge">⬡ OpenRouter  ·  {key_count}/3 Key  ·  {cur_model_short}</span>'
else:
    badge = '<span class="engine-badge warn">⚠ API Key 설정 필요</span>'

st.markdown(f"""
<div class="app-header">
    <div class="app-eyebrow">Supreme Intelligence System</div>
    <h1 class="app-title">Master <em>Mentor</em></h1>
    <div class="app-sub">세계 최고 수준의 통찰 &nbsp;·&nbsp; Powered by OpenRouter</div>
    {badge}
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  API KEY 없을 때 설정 안내
# ──────────────────────────────────────────────────────────────
if not API_KEYS:
    st.markdown("""
<div class="setup-card">
<h4>🔧 OpenRouter API Key를 설정해 주세요</h4>

<strong>Step 1.</strong> <a href="https://openrouter.ai" target="_blank" style="color:#1B4D3E;font-weight:600">openrouter.ai</a> 접속 → 무료 가입<br>
<strong>Step 2.</strong> 우측 상단 프로필 → <code>API Keys</code> → <code>Create Key</code> → 키 복사<br>
<strong>Step 3.</strong> 아래 중 한 곳에 등록:

<strong>① 로컬 실행 (.env 파일)</strong>
<pre>OPENROUTER_API_KEY_1=sk-or-여기에_키_입력
OPENROUTER_API_KEY_2=sk-or-여기에_키_입력
OPENROUTER_API_KEY_3=sk-or-여기에_키_입력</pre>

<strong>② Streamlit Cloud (Secrets 탭)</strong>
<pre>OPENROUTER_API_KEY_1 = "sk-or-여기에_키_입력"
OPENROUTER_API_KEY_2 = "sk-or-여기에_키_입력"
OPENROUTER_API_KEY_3 = "sk-or-여기에_키_입력"</pre>

저장 후 앱을 재시작하면 바로 사용 가능합니다.<br>
<span style="color:#1B4D3E;font-weight:600">무료 모델(:free)은 크레딧 차감 없이 무제한 사용 가능합니다.</span>
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

    # 사용자 버블
    st.markdown(
        '<div class="role-label user-lbl"><span class="ln"></span>✦ 나의 질문</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="row-user"><div class="bubble bubble-user">{user_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "user", "content": user_text})

    # OpenRouter 메시지 구성
    or_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        or_messages.append({"role": m["role"], "content": m["content"]})

    # 현재 모델명
    cur_model = OPENROUTER_MODELS[
        st.session_state.model_idx % len(OPENROUTER_MODELS)
    ].split("/")[-1].replace(":free", "")

    with st.spinner(f"◈  분석 중  ·  {cur_model}"):
        answer, error = call_openrouter(or_messages)

    if error:
        answer = f"**⚠️ 오류**\n\n{error}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # AI 버블
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