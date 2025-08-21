# app.py
import os
import time
import random
import base64
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO

# --- Ensure Plotly is available (safe import) ---
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.20.0"])
    import plotly.graph_objects as go

# Google Sheets
import gspread
from google.oauth2 import service_account

# ========= Parameters =========
N_TRIALS = 40
TRIAL_TIMEOUT_SEC = 30
DATA_PATH = "colors_in_charts.csv" # Adjusted path to be local

GSHEET_ID = "1ePIoLpP0Y0d_SedzVcJT7ttlV_1voLTssTvWAqpMkqQ"
GSHEET_WORKSHEET_NAME = "Results"

REQUIRED_COLS = ["ID", "ImageFileName", "QuestionText", "QCorrectAnswer"]

# ========= Brand =========
LOGO_CANDIDATES = [
    "images/Logo.png", "images/logo.png",
    "images/Logo29.10.24_B.png", "Logo.png", "Logo"
]
USER_PHOTO_CANDIDATES = ["images/DanaSherlock.png", "DanaSherlock.png"]
WEBSITE_URL = "http://www.2dpoint.co.il"
SHERLOCK_GITHUB_URL = "https://raw.githubusercontent.com/danaarnonperry/graph-color-experiment/main/DanaSherlock.png"
SHERLOCK_IMG_WIDTH = 160

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

LOGO_PATH = _first_existing(LOGO_CANDIDATES)
USER_PHOTO_PATH = _first_existing(USER_PHOTO_CANDIDATES)

# ========= Page Setup =========
st.set_page_config(page_title="ניסוי בזיכרון חזותי של גרפים", page_icon="📊", layout="centered")
st.markdown(
    """
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; font-family: "Rubik","Segoe UI","Arial",sans-serif; }
blockquote, pre, code { direction: ltr; text-align: left; }

/* לקרב את שורת הכפתורים לגרף */
div[data-testid="stPlotlyChart"]{ margin-bottom: 6px !important; }

/* הסתרת fullscreen של Streamlit */
button[title="View fullscreen"]{ display:none !important; }

/* Hiding the timeout button */
button[kind="secondary"] {
    display: none;
}
</style>
""",
    unsafe_allow_html=True,
)

# ========= Session State =========
def _admin_ui_enabled() -> bool:
    try:
        return (st.query_params.get("admin") == "1")
    except Exception:
        return (st.experimental_get_query_params().get("admin", ["0"])[0] == "1")

def init_state():
    ss = st.session_state
    ss.setdefault("page", "welcome")      # welcome -> practice -> practice_end -> trial -> end
    ss.setdefault("df", None)
    ss.setdefault("practice_list", [])
    ss.setdefault("practice_idx", 0)
    ss.setdefault("trials", None)
    ss.setdefault("i", 0)
    ss.setdefault("t_start", None)
    ss.setdefault("results", [])
    ss.setdefault("image_cache", {})
    ss.setdefault("participant_id", "")
    ss.setdefault("run_start_iso", "")
    ss.setdefault("is_admin", False)
    ss.setdefault("awaiting_response", False)
    ss.setdefault("last_feedback_html", "")  # משוב – לתרגול בלבד
init_state()

# ========= Admin =========
def is_admin(show_ui: bool = False):
    show = show_ui or _admin_ui_enabled()
    if show:
        with st.sidebar:
            if LOGO_PATH:
                st.image(LOGO_PATH, use_container_width=True)
            st.markdown("**🔐 אזור מנהל**")
            if not st.session_state.is_admin:
                pin = st.text_input("הכנסי PIN:", type="password", key="admin_pin")
                if st.button("כניסה", key="admin_login_btn"):
                    admin_pin = None
                    try:
                        admin_pin = st.secrets["admin"].get("pin")
                    except Exception:
                        pass
                    if not admin_pin:
                        st.error("לא מוגדר PIN (admin.pin) ב-Secrets.")
                    elif str(pin).strip() == str(admin_pin).strip():
                        st.session_state.is_admin = True
                        st.success("מנהל מחובר ✅")
                    else:
                        st.error("PIN שגוי")
            else:
                st.success("מנהל מחובר ✅")
    return st.session_state.is_admin

# ========= Data =========
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except Exception:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(how="all").fillna("")
    df = df.astype({c: str for c in df.columns})
    aliases = {"QCorrectA": "QCorrectAnswer", "QuestionT": "QuestionText", "ImageFile": "ImageFileName"}
    for src, dst in aliases.items():
        if dst not in df.columns and src in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"בעיית עמודות בקובץ הנתונים: חסרות {', '.join(missing)}")
    return df

# ========= Google Sheets =========
def _read_service_account_from_secrets() -> dict:
    try:
        sa = dict(st.secrets["service_account"])
        if sa:
            return sa
    except Exception:
        pass
    keys = ["type","project_id","private_key_id","private_key","client_email","client_id",
            "auth_uri","token_uri","auth_provider_x509_cert_url","client_x509_cert_url","universe_domain"]
    sa = {}
    for k in keys:
        try:
            sa[k] = st.secrets[k]
        except Exception:
            pass
    if not sa:
        raise RuntimeError("Service Account לא נמצא ב-secrets.")
    return sa

@st.cache_resource
def _gs_client():
    sa_info = _read_service_account_from_secrets()
    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
    )
    return gspread.authorize(creds)

def _ensure_headers(ws, expected_headers):
    current = ws.get_all_values()
    headers = list(expected_headers)
    if not current:
        ws.append_row(headers); return
    first_row = current[0] if current else []
    if first_row != headers:
        ws.update("1:1", [headers])

def get_next_participant_seq(sheet_id: str) -> int:
    gc = _gs_client()
    sh = gc.open_by_key(sheet_id)
    try:
        meta = sh.worksheet("Meta")
    except gspread.WorksheetNotFound:
        meta = sh.add_worksheet(title="Meta", rows="2", cols="2")
        meta.update("A1", "counter"); meta.update("A2", "1")
        return 1
    try:
        cur = int(meta.acell("A2").value or "0")
    except Exception:
        cur = 0
    nxt = cur + 1
    meta.update("A2", str(nxt))
    return nxt

def _ensure_participant_id():
    if st.session_state.participant_id:
        return
    try:
        seq = get_next_participant_seq(GSHEET_ID)
        st.session_state.participant_id = f"S{seq:05d}"
    except Exception:
        st.session_state.participant_id = f"S{int(time.time())}"

def append_dataframe_to_gsheet(df: pd.DataFrame, sheet_id: str, worksheet_name: str = "Results"):
    gc = _gs_client(); sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=str(max(len(df) + 10, 1000)), cols=str(len(df.columns) + 5))
    _ensure_headers(ws, df.columns)
    if not df.empty:
        ws.append_rows(df.astype(str).values.tolist(), value_input_option="RAW")

# ========= Utils =========
def load_image(path: str):
    if not path:
        return None
    cache = st.session_state.image_cache
    if path in cache:
        return cache[path]
    try:
        if path.startswith(("http://", "https://")):
            r = requests.get(path, timeout=10); r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGBA")
        else:
            img = Image.open(path).convert("RGBA")
        cache[path] = img
        return img
    except Exception:
        return None

def build_alternating_trials(pool_df: pd.DataFrame, n_needed: int):
    if "V" in pool_df.columns:
        groups = {v: sub.sample(frac=1, random_state=None).to_dict(orient="records") for v, sub in pool_df.groupby("V")}
        vs = list(groups.keys()); random.shuffle(vs)
        result, last_v = [], None
        for _ in range(n_needed):
            candidates = [v for v in vs if groups[v]]
            if not candidates: break
            non_same = [v for v in candidates if v != last_v] or candidates
            v = random.choice(non_same)
            result.append(groups[v].pop(0)); last_v = v
        if len(result) < n_needed:
            extra = pool_df.sample(n=min(n_needed - len(result), len(pool_df)), replace=False).to_dict(orient="records")
            result += extra
        return result[:n_needed]
    else:
        if len(pool_df) <= n_needed:
            return pool_df.sample(frac=1, random_state=None).to_dict(orient="records")
        return pool_df.sample(n=n_needed, replace=False, random_state=None).to_dict(orient="records")

def _extract_option_values_and_colors(row: dict):
    letters = ["A","B","C","D","E"]
    vals = {}
    for L in letters:
        if f"Value{L}" in row and str(row[f"Value{L}"]).strip() != "":
            vals[L] = float(row[f"Value{L}"])
        elif L in row and str(row[L]).strip() != "":
            vals[L] = float(row[L])
    if len(vals) != 5:
        raise ValueError("נדרשים ערכים לעמודות A..E (או ValueA..ValueE).")
    colors = {}
    for L in letters:
        key = f"Color{L}"
        if key in row and str(row[key]).strip() != "":
            colors[L] = str(row[key]).strip()
    correct = str(row.get("QCorrectAnswer", "")).strip().upper()
    default_gray = "#6b7280"; correct_green = "#22c55e"
    if not colors:
        colors = {L: (correct_green if L == correct else default_gray) for L in letters}
    x = letters; y = [vals[L] for L in letters]; c = [colors[L] for L in letters]
    return x, y, c

def _correct_phrase(question_text: str) -> str:
    q = str(question_text or "")
    if ("נמוך" in q) or ("lowest" in q.lower()):  return "עם הערך הנמוך ביותר"
    if ("גבוה" in q) or ("highest" in q.lower()): return "עם הערך הגבוה ביותר"
    return "התשובה הנכונה"

def _render_graph_block(title_html, question_text, row_dict):
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(f"### {question_text}")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    try:
        x, y, colors = _extract_option_values_and_colors(row_dict)
    except Exception as e:
        img = load_image(row_dict.get("ImageFileName", ""))
        if img is not None:
            left, mid, right = st.columns([1,6,1])
            with mid:
                st.image(img, width=min(1500, img.width))
            st.info("טיפ: ניתן לעבור לגרף בקוד ע\"י הוספת ValueA..ValueE (ואופציונלית ColorA..ColorE).")
            return
        else:
            st.error(f"שגיאת גרף: {e}")
            return

    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors,
        text=[f"{v:.0f}" for v in y],
        textposition="outside", texttemplate="<b>%{text}</b>",
        cliponaxis=False
    ))
    fig.update_traces(
        textfont=dict(size=20, color="#111111"),
        outsidetextfont=dict(size=20, color="#111")
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=0),
        showlegend=False, bargap=0.35,
        uniformtext_minsize=12, uniformtext_mode="hide",
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="", showgrid=False, showticklabels=False, zeroline=False),
        hovermode=False,
    )

    left, mid, right = st.columns([1,6,1])
    with mid:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False, "responsive": True, "staticPlot": True})

# --- NEW: Function to render the centered answer buttons ---
def render_answer_bar(
    on_press,
    key_suffix,
    disabled=False,
    options=["A", "B", "C", "D", "E"],
    label="בחר/י תשובה",
):
    """Renders the answer radio buttons in a centered horizontal layout."""
    key = f"radio_{key_suffix}"

    # Use columns to center the radio buttons horizontally
    _left_co, mid_co, _right_co = st.columns([1, 2, 1])
    with mid_co:
        st.radio(
            label,
            options,
            key=key,
            index=None,
            horizontal=True,
            on_change=on_press,
            kwargs={"key": key},
            disabled=disabled,
            label_visibility="collapsed",
        )

# --- NEW: Robust timer and answer handler ---
def _radio_answer_and_timer(timeout, on_timeout, on_press):
    """Renders answer radio buttons and a robust, single-instance countdown timer."""
    timeout_ms = timeout * 1000
    # A key that is unique for each trial run
    key_suffix = f"{st.session_state.run_start_iso}_{st.session_state.i}"
    # A key to ensure the timer script is injected only once per trial
    timer_initiated_key = f"timer_initiated_{st.session_state.i}"

    # JavaScript to handle the countdown
    js_code = f"""
        <script>
            // Ensure a global timer object exists
            if (!window.trialTimer) {{
                window.trialTimer = {{ intervalId: null, startTime: null }};
            }}

            // Clear any previous timer to prevent multiple timers running
            if (window.trialTimer.intervalId) {{
                clearInterval(window.trialTimer.intervalId);
            }}

            window.trialTimer.startTime = new Date().getTime();

            function checkTimeout() {{
                const currentTime = new Date().getTime();
                const elapsedTime = currentTime - window.trialTimer.startTime;

                // Update remaining time display (optional, but good for UX)
                const remainingSeconds = Math.max(0, Math.ceil(({timeout_ms} - elapsedTime) / 1000));
                const timerElement = window.parent.document.getElementById('timer-display');
                if (timerElement) {{
                    timerElement.innerText = remainingSeconds;
                }}

                if (elapsedTime > {timeout_ms}) {{
                    // Find and click the hidden timeout button
                    const buttons = window.parent.document.querySelectorAll('button');
                    // Find the button with a specific key to make it unique
                    const timeoutButton = Array.from(buttons).find(btn => btn.innerText && btn.innerText.includes('Timeout_Internal'));
                    if (timeoutButton) {{
                        timeoutButton.click();
                    }}
                    clearInterval(window.trialTimer.intervalId); // Stop this timer
                }}
            }}

            // Start the new timer
            window.trialTimer.intervalId = setInterval(checkTimeout, 250); // Check 4 times a second
        </script>
    """

    # Inject the JavaScript only once per trial
    if not st.session_state.get(timer_initiated_key, False):
        st.session_state[timer_initiated_key] = True
        components.html(js_code, height=0)

    # Hidden button for the JavaScript to click on timeout
    if st.button(f"Timeout_Internal_{key_suffix}", key=f"timeout_btn_{key_suffix}"):
        on_timeout()

    # Render the answer buttons
    def _on_change(key):
        choice = st.session_state.get(key)
        if choice:
            on_press(choice)

    render_answer_bar(
        on_press=_on_change,
        key_suffix=key_suffix,
        disabled=st.session_state.get("answer_submitted", False),
    )
    
    # Display for the remaining time
    st.markdown(
        "<div style='text-align:center; margin-top:12px;'>⏳ זמן שנותר: <b id='timer-display'>{timeout}</b> שניות</div>",
        unsafe_allow_html=True,
    )


# ===== Helper: clickable logo via base64 =====
def _file_to_base64_html_img_link(path: str, href: str, width_px: int = 140) -> str:
    try:
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return (f"<a href='{href}' target='_blank'>"
                f"<img src='data:{mime};base64,{b64}' style='width:{width_px}px; border:0;'/>"
                f"</a>")
    except Exception:
        return ""

# ========= Screens =========
def screen_welcome():
    st.title("ניסוי בזיכרון חזותי של גרפים 📊")
    st.markdown(
        """
**שלום וברוכ/ה הבא/ה לניסוי** במהלך הניסוי יוצגו **40 גרפים** שלגביהם תתבקש/י לציין מהו הערך הנמוך ביותר או הגבוה ביותר.

חשוב לענות מהר ככל שניתן; לאחר **30 שניות**, אם לא נבחרה תשובה, יהיה מעבר אוטומטי לשאלה הבאה.

**איך עונים?** לוחצים על הכפתור עם האות המתאימה מתחת לגרף **A / B / C / D / E**.

לפני תחילת הניסוי, יוצגו **שתי שאלות תרגול** (לא נשמרות בתוצאות).

כדי להתחיל – לחצו על **המשך לתרגול**.
"""
    )

    if not os.path.exists(DATA_PATH):
        st.error(f"לא נמצא הקובץ: {DATA_PATH}."); st.stop()

    try:
        df = load_data()
    except Exception as e:
        st.error(str(e)); st.stop()

    total_rows = len(df)
    if total_rows < 2:
        st.error("בקובץ חייבות להיות לפחות 2 שורות תרגול בתחילתו."); st.stop()
    if total_rows < 2 + N_TRIALS:
        st.warning(f"התקבלו רק {max(0,total_rows-2)} שאלות לניסוי במקום 40. נריץ את הקיים.")

    if st.button("המשך לתרגול"):
        _ensure_participant_id()
        st.session_state.run_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")

        n_trials_final = min(N_TRIALS, max(0, total_rows - 2))
        practice_items = df.iloc[:2].to_dict(orient="records")
        pool_df = df.iloc[2: 2 + n_trials_final].copy()
        trials = build_alternating_trials(pool_df, n_trials_final)

        st.session_state.df = df
        st.session_state.practice_list = practice_items
        st.session_state.trials = trials
        st.session_state.practice_idx = 0
        st.session_state.i = 0
        st.session_state.t_start = None
        st.session_state.results = []
        st.session_state.page = "practice"
        st.rerun()

def _practice_one(idx: int):
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()
        st.session_state.awaiting_response = True
        st.session_state.last_feedback_html = ""

    t = st.session_state.practice_list[idx]
    title_html = f"<div style='font-size:20px; font-weight:700; text-align:right; margin-bottom:0.5rem;'>תרגול {idx+1} / {len(st.session_state.practice_list)}</div>"
    _render_graph_block(title_html, t["QuestionText"], t)

    if st.session_state.last_feedback_html:
        st.markdown(st.session_state.last_feedback_html, unsafe_allow_html=True)

    def on_timeout():
        st.session_state.t_start = None
        st.session_state.awaiting_response = False
        if st.session_state.practice_idx + 1 < len(st.session_state.practice_list):
            st.session_state.practice_idx += 1; st.rerun()
        else:
            st.session_state.page = "practice_end"; st.rerun()

    def on_press(key):
        correct_letter = str(t["QCorrectAnswer"]).strip().upper()
        chosen = key.strip().upper()
        phrase = _correct_phrase(t.get("QuestionText", ""))
        if chosen == correct_letter:
            st.session_state.awaiting_response = False
            st.session_state.last_feedback_html = (
                f"<div style='text-align:center; margin:10px 0; font-weight:700;'>✅ צדקת, עמודה <b>{correct_letter}</b> היא {phrase}.</div>"
            )
        else:
            st.session_state.awaiting_response = True
            st.session_state.last_feedback_html = (
                "<div style='text-align:center; margin:10px 0; font-weight:700;'>❌ לא מדויק – נסה/י שוב.</div>"
            )
        st.rerun()

    if st.session_state.awaiting_response:
        _radio_answer_and_timer(TRIAL_TIMEOUT_SEC, on_timeout, on_press)
    else:
        center = st.columns([1,6,1])[1]
        with center:
            if st.button("המשך", key=f"practice_next_{idx}"):
                st.session_state.t_start = None
                st.session_state.last_feedback_html = ""
                if st.session_state.practice_idx + 1 < len(st.session_state.practice_list):
                    st.session_state.practice_idx += 1; st.rerun()
                else:
                    st.session_state.page = "practice_end"; st.rerun()

def screen_practice():
    _practice_one(st.session_state.practice_idx)

def screen_practice_end():
    st.markdown(
        "<div style='text-align:center; font-size:28px; font-weight:800; margin:32px 0;'>התרגיל הסתיים</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center; font-size:20px; font-weight:600; margin-bottom:24px;'>לחץ על <u>התחל</u> כדי להמשיך</div>",
        unsafe_allow_html=True
    )
    mid = st.columns([1,6,1])[1]
    with mid:
        if st.button("התחל", type="primary"):
            st.session_state.page = "trial"
            st.session_state.t_start = None
            st.session_state.awaiting_response = False
            st.session_state.last_feedback_html = ""
            st.rerun()

def screen_trial():
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()
        st.session_state.awaiting_response = True

    i = st.session_state.i
    t = st.session_state.trials[i]
    title_html = f"<div style='font-size:20px; font-weight:700; text-align:right; margin-bottom:0.5rem;'>גרף מספר {i+1}</div>"
    _render_graph_block(title_html, t["QuestionText"], t)

    def finish_with(resp_key, rt_sec, correct):
        st.session_state.results.append(
            {
                "ParticipantID": st.session_state.participant_id,
                "RunStartISO": st.session_state.run_start_iso,
                "TrialIndex": st.session_state.i + 1,
                "ID": t["ID"],
                "ResponseKey": resp_key or "",
                "QCorrectAnswer": t["QCorrectAnswer"],
                "Accuracy": int(correct),
                "RT_sec": round(rt_sec, 3),
            }
        )
        st.session_state.t_start = None
        st.session_state.awaiting_response = False
        if st.session_state.i + 1 < len(st.session_state.trials):
            st.session_state.i += 1
        else:
            st.session_state.page = "end"
        st.rerun()

    def on_timeout():
        finish_with(resp_key=None, rt_sec=float(TRIAL_TIMEOUT_SEC), correct=0)

    def on_press(key):
        rt = time.time() - (st.session_state.t_start or time.time())
        correct_letter = str(t["QCorrectAnswer"]).strip().upper()
        chosen = key.strip().upper()
        is_correct = (chosen == correct_letter)
        finish_with(resp_key=chosen, rt_sec=rt, correct=is_correct)

    _radio_answer_and_timer(TRIAL_TIMEOUT_SEC, on_timeout, on_press)

def screen_end():
    st.title("סיום הניסוי")
    st.success("תודה על השתתפותך!")

    df = pd.DataFrame(st.session_state.results)
    admin = is_admin()

    if df.empty:
        st.info("לא נאספו תוצאות.")
    else:
        try:
            append_dataframe_to_gsheet(df, GSHEET_ID, worksheet_name=GSHEET_WORKSHEET_NAME)
            st.success("התשובות נשלחו בהצלחה ✅")
        except Exception as e:
            if admin:
                st.error(f"נכשלה כתיבה ל-Google Sheets: {type(e).__name__}: {e}")
            else:
                st.info("התשובות נשלחו. אם יידרש, נבצע שמירה חוזרת מאחורי הקלעים.")

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center; margin:24px 0;">
            <img src="{SHERLOCK_GITHUB_URL}" width="{SHERLOCK_IMG_WIDTH}" alt="Sherlock" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    if LOGO_PATH and WEBSITE_URL:
        html = _file_to_base64_html_img_link(LOGO_PATH, WEBSITE_URL, width_px=140)
        if html:
            st.markdown(f"<div style='text-align:center; margin-top:10px;'>{html}</div>", unsafe_allow_html=True)
        else:
            st.link_button("לאתר שלי", WEBSITE_URL, type="primary")
    elif WEBSITE_URL:
        st.link_button("לאתר שלי", WEBSITE_URL, type="primary")

    if admin and not df.empty:
        st.download_button(
            "הורדת תוצאות (CSV)",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"{st.session_state.participant_id}_{st.session_state.run_start_iso.replace(':','-')}.csv",
            mime="text/csv",
        )
        st.link_button("פתח/י את Google Sheet", f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit", type="primary")

# ========= Router =========
page = st.session_state.page
if page == "welcome":
    screen_welcome()
elif page == "practice":
    screen_practice()
elif page == "practice_end":
    screen_practice_end()
elif page == "trial":
    screen_trial()
else:
    screen_end()
