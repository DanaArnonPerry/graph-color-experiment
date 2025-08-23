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
N_TRIALS_DEFAULT = 40  # ← ניתן לשינוי בפרמטר כתובת ?n=
TRIAL_TIMEOUT_DEFAULT = 30  # ← ניתן לשינוי בפרמטר כתובת ?timeout=
DATA_PATH = "data/colors_in_charts.csv"

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

# ========= Helpers for query-params (חדש) =========
# תומך גם ב-st.query_params וגם ב-fallback ישן

def _qp_raw(name, default=None):
    try:
        val = st.query_params.get(name)
        return default if val is None else val
    except Exception:
        return st.experimental_get_query_params().get(name, [default])[0]

def _qp_int(name, default):
    try:
        val = _qp_raw(name, default)
        return int(val) if val is not None else default
    except Exception:
        return default

def _qp_str(name, default):
    val = _qp_raw(name, default)
    return default if val is None else str(val)

# ========= Pick first-existing =========

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

/* אפס מרווחים סביב גרף */
div[data-testid="stPlotlyChart"], .stPlotlyChart { margin-bottom: 0 !important; }

/* קומפקטיות – פחות רווחים כדי למנוע גלילה */
section.main > div.block-container { padding-top: 10px; padding-bottom: 12px; }

/* טיימר מקובע למעלה באמצע */
#fixed-timer {
  position: fixed; top: 0; left: 50%; transform: translateX(-50%);
  z-index: 9999; background: #111; color: #fff;
  padding: 6px 12px; margin: 0; border-radius: 0 0 12px 12px;
  font-weight: 800; letter-spacing: .5px;
}

/* פסי רווח תחתונים מיותרים */
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown("""
<style>
/* ===== Progress bar: black fill, grey track (new + old DOM) ===== */

/* NEWER: <progress> element */
div[data-testid="stProgress"] progress,
div[data-testid="stProgressBar"] progress {
  appearance: none; -webkit-appearance: none;
  width: 100%; height: 12px; border: none; background: transparent;
  accent-color: #000 !important;              /* צבע המילוי (תקני) */
}
/* WebKit track & fill */
div[data-testid="stProgress"] progress::-webkit-progress-bar,
div[data-testid="stProgressBar"] progress::-webkit-progress-bar {
  background-color: #e5e7eb !important;       /* צבע המסילה */
  border-radius: 9999px;
}
div[data-testid="stProgress"] progress::-webkit-progress-value,
div[data-testid="stProgressBar"] progress::-webkit-progress-value {
  background-color: #000 !important;          /* מילוי שחור */
  border-radius: 9999px;
}
/* Firefox fill */
div[data-testid="stProgress"] progress::-moz-progress-bar,
div[data-testid="stProgressBar"] progress::-moz-progress-bar {
  background-color: #000 !important;
  border-radius: 9999px;
}

/* OLDER: div-based progressbar (כמו בשרשור) */
.stProgress > div > div > div {               /* המסילה */
  background-color: #e5e7eb !important;
}
.stProgress > div > div > div > div {         /* המילוי */
  background-color: #000 !important;
}

/* fallback כללי ישן */
div[data-testid="stProgress"] div[role="progressbar"]        { background-color: #e5e7eb !important; border-radius: 9999px !important; }
div[data-testid="stProgress"] div[role="progressbar"] > div  { background-color: #000 !important;     border-radius: 9999px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{
  --graph-top: 28px;   /* כמה להוריד את הגרף (גדול יותר = נמוך יותר) */
  --buttons-up: -18px; /* כמה להרים את הכפתורים (שלילי=למעלה, חיובי=למטה) */
}

/* הזזת הגרף למטה/למעלה */
div[data-testid="stPlotlyChart"], .stPlotlyChart{
  margin-top: var(--graph-top) !important;
}

/* קירוב שורת הכפתורים לגרף */
.choice-wrap{ margin-top: var(--buttons-up) !important; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* מצמיד את סרגל ההתקדמות מתחת לטיימר הקבוע */
div[data-testid="stProgress"],
div[data-testid="stProgressBar"]{
  position: sticky;
  top: 48px;          /* מתחת ל-#fixed-timer (שגובהו ~36–40px) */
  z-index: 100;       /* נמוך מהטיימר (9999) */
  margin-top: -100px;    /* ריווח קטן מהרכיב שמעל */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{
  --graph-top: 28px;
  --buttons-up: -18px;
  --question-top: 8px;      /* כמה להוריד/להעלות את השאלה */
  --question-bottom: 12px;  /* רווח מתחת לשאלה */
}
.question-text{
  text-align: center !important;               /* מרכז אופקית */
  margin-top: var(--question-top) !important;  /* הזזה אנכית */
  margin-bottom: var(--question-bottom) !important;
  font-weight: 800;                             /* אופציונלי – דומה ל-### */
  font-size: clamp(22px, 3vw, 36px);            /* אופציונלי */
}
</style>
""", unsafe_allow_html=True)

# ========= Session State =========

def _admin_ui_enabled() -> bool:
    try:
        return (st.query_params.get("admin") == "1")
    except Exception:
        return (st.experimental_get_query_params().get("admin", ["0"])[0] == "1")


def init_state():
    ss = st.session_state
    ss.setdefault("page", "welcome")
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
    ss.setdefault("last_feedback_html", "")
    ss.setdefault("results_saved", False)
    # חדש – פרמטרים דינמיים מהרצת ה-URL
    ss.setdefault("timeout_sec", TRIAL_TIMEOUT_DEFAULT)
    ss.setdefault("n_trials_req", N_TRIALS_DEFAULT)
    ss.setdefault("trial_start_iso", "")

init_state()

# עדכון פרמטרים מה-URL (חדש)
st.session_state.timeout_sec = _qp_int("timeout", st.session_state.timeout_sec)
st.session_state.n_trials_req = _qp_int("n", st.session_state.n_trials_req)
_seed = _qp_str("seed", None)
if _seed is not None and _seed != "None":
    try:
        random.seed(int(_seed))
    except Exception:
        random.seed(_seed)

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
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(how="all").fillna("")
    df = df.astype({c: str for c in df.columns})
    df.columns = df.columns.str.strip()
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
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
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

# חדש – כתיבה עם ניסיונות חוזרים + גיבוי לוקאלי

def _write_results_with_retry(df: pd.DataFrame, retries: int = 3, base_delay: float = 1.5):
    last_err = None
    for attempt in range(retries):
        try:
            append_dataframe_to_gsheet(df, GSHEET_ID, worksheet_name=GSHEET_WORKSHEET_NAME)
            return True, None
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    # גיבוי לוקאלי אם כל הניסיונות כשלו
    os.makedirs("results", exist_ok=True)
    fname = f"results/{st.session_state.participant_id}_{st.session_state.run_start_iso.replace(':','-')}.csv"
    try:
        df.to_csv(fname, index=False, encoding="utf-8-sig")
        return False, f"נכשלו ניסיונות הכתיבה ל-Google Sheets. נשמר גיבוי מקומי: {fname}"
    except Exception as e2:
        return False, f"נכשלו הכתיבה ל-Google Sheets וגם לגיבוי מקומי ({type(e2).__name__}: {e2}). השגיאה המקורית: {type(last_err).__name__}: {last_err}"

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
    letters = ["E","D","C","B","A"]
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


# ========= Small UI helpers (חדש) =========

def _render_progress(current_index: int, total: int, label: str = ""):
    col = st.columns([1,6,1])[1]
    with col:
        st.progress((current_index) / max(1, total), text=label or f"{current_index}/{total}")


def _render_graph_block(title_html, question_text, row_dict):
    if title_html:  # הדפס רק אם יש טקסט
        st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(f"### {question_text}")
    
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
    fig.update_traces(textfont=dict(size=20, color="#111"))
    fig.update_layout(
        margin=dict(l=20, r=20, t=18, b=0),
        height=320,
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


# ---------- שורת כפתורים ממורכזת A–E ----------

def render_choice_buttons(key_prefix: str, on_press, letters=("A","B","C","D","E")):
    st.markdown(
        """
    <style>
    .choice-wrap { 
        display:flex; justify-content:center; 
        gap: clamp(10px,1.6vw,22px); 
        margin-top: var(--buttons-up);
    }
    .choice-wrap .stButton>button {
        width: clamp(44px, 6vw, 64px);
        height: clamp(44px, 6vw, 64px);
        border-radius: 9999px;
        background: #e5e7eb;
        border: 1.5px solid #9ca3af;
        font-weight: 800;
        font-size: clamp(16px, 2.2vw, 20px);
        color: #111;
        box-shadow: 0 1px 0 rgba(0,0,0,.08);
        padding: 0;
    }
    .choice-wrap .stButton>button:hover { filter: brightness(1.05); }
    </style>
    """, unsafe_allow_html=True)

    outer_cols = st.columns([1,6,1])
    with outer_cols[1]:
        st.markdown('<div class="choice-wrap">', unsafe_allow_html=True)
        cols = st.columns(len(letters), gap="small")
        for L, c in zip(letters, cols):
            with c:
                if st.button(L, key=f"{key_prefix}_btn_{L}"):
                    on_press(L)
        st.markdown('</div>', unsafe_allow_html=True)


def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def _radio_answer_and_timer(timeout_sec, on_timeout, on_press):
    """הצגת טיימר עליון + כפתורי A–E צמודים לגרף וממורכזים."""
    if not st.session_state.get("awaiting_response", False):
        return

    elapsed = time.time() - (st.session_state.t_start or time.time())
    remain = max(0, timeout_sec - int(elapsed))

    # טיימר קבוע למעלה
    st.markdown(f"<div id='fixed-timer'>⏳ זמן שנותר: <b>{remain}</b> שניות</div>", unsafe_allow_html=True)

    # אם הזמן נגמר – סוגרים את ה-trial
    if elapsed >= timeout_sec and st.session_state.awaiting_response:
        on_timeout()
        _safe_rerun()
        return

    # מפתח ייחודי לכפתורים
    current_index = (st.session_state.practice_idx
                     if st.session_state.page == "practice" else st.session_state.i)
    key_prefix = f"choice_{st.session_state.page}_{current_index}"

    # שורת כפתורים ממורכזת
    render_choice_buttons(key_prefix, on_press)

    # רענון עדין פעם בשנייה לעדכון הטיימר
    if st.session_state.get("awaiting_response", False):
        time.sleep(1)
        _safe_rerun()


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
**שלום וברוכ/ה הבא/ה לניסוי**

במהלך הניסוי יוצגו **40 גרפים** שלגביהם תתבקש/י לציין מהו הערך הנמוך ביותר או הגבוה ביותר.

חשוב לענות מהר ככל שניתן; לאחר **30 שניות**, אם לא נבחרה תשובה, יהיה מעבר אוטומטי לשאלה הבאה.

**איך עונים?** לוחצים על הכפתור עם האות המתאימה מתחת לגרף **A / B / C / D / E**.

לפני תחילת הניסוי, יוצגו **שתי שאלות תרגול.**


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

    # עדכון טקסט לפי פרמטרים דינמיים
    if st.session_state.timeout_sec != TRIAL_TIMEOUT_DEFAULT or st.session_state.n_trials_req != N_TRIALS_DEFAULT:
        st.info(f"הרצה זו תוגדר עם {st.session_state.n_trials_req} שאלות וזמן {st.session_state.timeout_sec} שניות לשאלה (ע\"י פרמטרי כתובת URL).")

    if total_rows < 2 + st.session_state.n_trials_req:
        st.warning(f"התקבלו רק {max(0,total_rows-2)} שאלות לניסוי במקום {st.session_state.n_trials_req}. נריץ את הקיים.")

    def on_start():
        _ensure_participant_id()
        st.session_state.run_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")
        n_trials_final = min(st.session_state.n_trials_req, max(0, total_rows - 2))
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
        st.session_state.results_saved = False
        st.session_state.page = "practice"

    st.button("המשך לתרגול", on_click=on_start)


def _practice_one(idx: int):
    total = len(st.session_state.practice_list)
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()
        st.session_state.awaiting_response = True
        st.session_state.last_feedback_html = ""
        st.session_state.trial_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")

    _render_progress(idx, total, label=f"תרגול {idx+1}/{total}")

    t = st.session_state.practice_list[idx]
    _render_graph_block("", t["QuestionText"], t)
  
    if st.session_state.last_feedback_html:
        st.markdown(st.session_state.last_feedback_html, unsafe_allow_html=True)

    def on_timeout():
        st.session_state.t_start = None
        st.session_state.awaiting_response = False
        if st.session_state.practice_idx + 1 < len(st.session_state.practice_list):
            st.session_state.practice_idx += 1
        else:
            st.session_state.page = "practice_end"

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
        _safe_rerun()

    if st.session_state.awaiting_response:
        _radio_answer_and_timer(st.session_state.timeout_sec, on_timeout, on_press)
    else:
        center = st.columns([1,6,1])[1]
        def on_next():
            st.session_state.t_start = None
            st.session_state.last_feedback_html = ""
            if st.session_state.practice_idx + 1 < len(st.session_state.practice_list):
                st.session_state.practice_idx += 1
            else:
                st.session_state.page = "practice_end"
        with center:
            st.button("המשך", key=f"practice_next_{idx}", on_click=on_next)


def screen_practice():
    _practice_one(st.session_state.practice_idx)


def screen_practice_end():
    st.session_state.awaiting_response = False
    st.session_state.t_start = None
    st.markdown(
        "<div style='text-align:center; font-size:28px; font-weight:800; margin:32px 0;'>התרגיל הסתיים</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center; font-size:20px; font-weight:600; margin-bottom:24px;'>לחץ על <u>התחל</u> כדי להמשיך</div>",
        unsafe_allow_html=True
    )
    mid = st.columns([1,6,1])[1]
    def on_start():
        st.session_state.page = "trial"
        st.session_state.t_start = None
        st.session_state.awaiting_response = False
        st.session_state.last_feedback_html = ""
    with mid:
        st.button("התחל", type="primary", on_click=on_start)


def screen_trial():
    total = len(st.session_state.trials)
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()
        st.session_state.awaiting_response = True
        st.session_state.trial_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")

    i = st.session_state.i
    _render_progress(i, total, label=f"גרף {i+1}/{total}")

    t = st.session_state.trials[i]
        
    _render_graph_block("", t["QuestionText"], t)


    def finish_with(resp_key, rt_sec, correct):
        st.session_state.results.append({
            "ParticipantID": st.session_state.participant_id,
            "RunStartISO": st.session_state.run_start_iso,
            "TrialStartISO": st.session_state.trial_start_iso,
            "TrialIndex": st.session_state.i + 1,
            "ID": t["ID"],
            "ResponseKey": resp_key or "",
            "QCorrectAnswer": t["QCorrectAnswer"],
            "Accuracy": int(correct),
            "RT_sec": round(rt_sec, 3),
        })
        st.session_state.t_start = None
        st.session_state.awaiting_response = False
        if st.session_state.i + 1 < len(st.session_state.trials):
            st.session_state.i += 1
        else:
            st.session_state.page = "end"

    def on_timeout():
        finish_with(resp_key=None, rt_sec=float(st.session_state.timeout_sec), correct=0)
        _safe_rerun()

    def on_press(key):
        rt = time.time() - (st.session_state.t_start or time.time())
        correct_letter = str(t["QCorrectAnswer"]).strip().upper()
        chosen = key.strip().upper()
        is_correct = (chosen == correct_letter)
        finish_with(resp_key=chosen, rt_sec=rt, correct=is_correct)
        _safe_rerun()

    _radio_answer_and_timer(st.session_state.timeout_sec, on_timeout, on_press)


def screen_end():
    st.session_state.awaiting_response = False
    st.session_state.t_start = None

    st.title("סיום הניסוי")
    st.success("תודה על השתתפותך!")
    df = pd.DataFrame(st.session_state.results)
    admin = is_admin()

    if df.empty:
        st.info("לא נאספו תוצאות.")
    elif not st.session_state.get("results_saved", False):
        ok, msg = _write_results_with_retry(df, retries=3)
        if ok:
            st.session_state.results_saved = True
            st.success("התשובות נשלחו בהצלחה ✅")
        else:
            if admin:
                st.error(msg)
                if st.button("נסה כתיבה חוזרת ל-Google Sheets"):
                    ok2, msg2 = _write_results_with_retry(df, retries=2)
                    if ok2:
                        st.session_state.results_saved = True
                        st.success("נשמר בהצלחה לאחר ניסיון חוזר ✅")
                    else:
                        st.error(msg2)
            else:
                st.warning("התשובות נשמרו בקובץ גיבוי מקומי. ניתן להוריד למטה ולשלוח/לשמור ידנית.")
    else:
        st.success("התשובות נשלחו בהצלחה ✅")

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
