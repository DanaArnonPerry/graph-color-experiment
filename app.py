# app.py
import os
import time
import random
import base64
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO

# === NEW ===
import plotly.graph_objects as go  # גרף בקוד, בלי אייקון הרחבה

# Google Sheets
import gspread
from google.oauth2 import service_account

# ========= Parameters =========
N_TRIALS = 40
TRIAL_TIMEOUT_SEC = 30
DATA_PATH = "data/colors_in_charts.csv"

# מזהה הגיליון ו-worksheet לתוצאות
GSHEET_ID = "1ePIoLpP0Y0d_SedzVcJT7ttlV_1voLTssTvWAqpMkqQ"
GSHEET_WORKSHEET_NAME = "Results"

# עמודות נדרשות מינימליות בקובץ ה-CSV
# נשאר כמו שהיה – לא שוברים תאימות. את עמודות הערכים/הצבעים נזהה דינמית.
REQUIRED_COLS = ["ID", "ImageFileName", "QuestionText", "QCorrectAnswer"]

# --- Admin UI toggle via URL (?admin=1) ---
def _admin_ui_enabled() -> bool:
    try:
        return (st.query_params.get("admin") == "1")
    except Exception:
        return (st.experimental_get_query_params().get("admin", ["0"])[0] == "1")


# ========= (Optional) Brand assets =========
LOGO_CANDIDATES = [
    "images/Logo.png", "images/logo.png",
    "images/Logo29.10.24_B.png", "Logo.png", "Logo"
]
USER_PHOTO_CANDIDATES = [
    "images/DanaSherlock.png",
    "DanaSherlock.png",
]
WEBSITE_URL = "http://www.2dpoint.co.il"

SHERLOCK_GITHUB_URL = (
    "https://raw.githubusercontent.com/danaarnonperry/graph-color-experiment/main/DanaSherlock.png"
)
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
/* RTL בסיסי + פונטים */
html, body, [class*="css"] { direction: rtl; text-align: right; font-family: "Rubik","Segoe UI","Arial",sans-serif; }
blockquote, pre, code { direction: ltr; text-align: left; }

/* רדיו אופקי – (משמש בדפים אחרים אם יש), השארנו */
div.stRadio > div[role="radiogroup"]{
  display:flex;
  justify-content:center;
  gap:12px;
  flex-wrap:wrap;
}
div.stRadio > div[role="radiogroup"] label{
  border:1px solid #d0d7de;
  border-radius:12px;
  padding:10px 16px;
  min-width:52px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:#fff;
  box-shadow:0 1px 2px rgba(0,0,0,.05);
  cursor:pointer;
}
div.stRadio > div[role="radiogroup"] label:hover{ background:#f6f8fa; }
div.stRadio input[type="radio"]{ position:absolute; opacity:0; pointer-events:none; }
div.stRadio > div[role="radiogroup"] label:has(input[type="radio"]:checked){
  background:#e6f0ff; border-color:#80b3ff; box-shadow:0 0 0 2px rgba(128,179,255,.25) inset;
}

/* === NEW === כפתורי st.button קומפקטיים */
div.stButton > button {
  height: 42px; min-width: 60px; width: 100%;
  padding: 0 8px; margin: 4px 0;
  font-size: 16px; border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ========= Session State =========
def init_state():
    ss = st.session_state
    ss.setdefault("page", "welcome")     # welcome -> practice -> trial -> end
    ss.setdefault("df", None)
    ss.setdefault("practice", None)
    ss.setdefault("trials", None)
    ss.setdefault("i", 0)
    ss.setdefault("t_start", None)
    ss.setdefault("results", [])
    ss.setdefault("image_cache", {})
    ss.setdefault("participant_id", "")
    ss.setdefault("run_start_iso", "")
    ss.setdefault("is_admin", False)
    ss.setdefault("awaiting_response", False)
    ss.setdefault("saved_to_sheets", False)
init_state()

# ========= Admin PIN =========
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

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"בעיית עמודות בקובץ הנתונים: חסרות {', '.join(missing)}")

    return df

# ========= Google Sheets helpers =========
def _read_service_account_from_secrets() -> dict:
    try:
        sa = dict(st.secrets["service_account"])
        if sa:
            return sa
    except Exception:
        pass

    keys = [
        "type", "project_id", "private_key_id", "private_key",
        "client_email", "client_id", "auth_uri", "token_uri",
        "auth_provider_x509_cert_url", "client_x509_cert_url",
        "universe_domain",
    ]
    sa = {}
    for k in keys:
        try:
            sa[k] = st.secrets[k]
        except Exception:
            pass
    if not sa:
        raise RuntimeError("Service Account לא נמצא ב-secrets. ודאי שהגדרת [service_account] או מפתחות SA בטופ-לבל.")
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
        ws.append_row(headers)
        return
    first_row = current[0]
    if first_row != headers:
        ws.update("1:1", [headers])

def get_next_participant_seq(sheet_id: str) -> int:
    gc = _gs_client()
    sh = gc.open_by_key(sheet_id)
    try:
        meta = sh.worksheet("Meta")
    except gspread.WorksheetNotFound:
        meta = sh.add_worksheet(title="Meta", rows="2", cols="2")
        meta.update("A1", "counter")
        meta.update("A2", "1")
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
    gc = _gs_client()
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=worksheet_name,
            rows=str(max(len(df) + 10, 1000)),
            cols=str(len(df.columns) + 5),
        )
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
            r = requests.get(path, timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGBA")
        else:
            img = Image.open(path).convert("RGBA")
        cache[path] = img
        return img
    except Exception:
        return None

def build_alternating_trials(pool_df: pd.DataFrame, n_needed: int):
    if "V" in pool_df.columns:
        groups = {
            v: sub.sample(frac=1, random_state=None).to_dict(orient="records")
            for v, sub in pool_df.groupby("V")
        }
        vs = list(groups.keys())
        random.shuffle(vs)
        result, last_v = [], None
        for _ in range(n_needed):
            candidates = [v for v in vs if groups[v]]
            if not candidates:
                break
            non_same = [v for v in candidates if v != last_v] or candidates
            v = random.choice(non_same)
            result.append(groups[v].pop(0))
            last_v = v
        if len(result) < n_needed:
            extra = pool_df.sample(
                n=min(n_needed - len(result), len(pool_df)),
                replace=False
            ).to_dict(orient="records")
            result += extra
        return result[:n_needed]
    else:
        if len(pool_df) <= n_needed:
            return pool_df.sample(frac=1, random_state=None).to_dict(orient="records")
        return pool_df.sample(n=n_needed, replace=False, random_state=None).to_dict(orient="records")

# -------- גרף: רוחב מקסימלי --------
GRAPH_MAX_WIDTH_PX = 1500

# === NEW === עזר: חילוץ ערכי/צבעי A..E מהשורה (תומך בשמות גמישים)
def _extract_option_values_and_colors(row: dict):
    """
    מחפש ערכים לפי העדיפות הבאה:
    1) ValueA..ValueE  או  A..E
    2) צבעים: ColorA..ColorE (אופציונלי). אם אין – ברירת מחדל אפור,
       והתשובה הנכונה (QCorrectAnswer) תסומן בירוק.
    """
    letters = ["A", "B", "C", "D", "E"]

    # ערכים
    vals = {}
    for L in letters:
        if f"Value{L}" in row and str(row[f"Value{L}"]).strip() != "":
            vals[L] = float(row[f"Value{L}"])
        elif L in row and str(row[L]).strip() != "":
            vals[L] = float(row[L])

    if len(vals) != 5:
        raise ValueError("נדרשים ערכים לעמודות A..E (או ValueA..ValueE).")

    # צבעים
    colors = {}
    for L in letters:
        key = f"Color{L}"
        if key in row and str(row[key]).strip() != "":
            colors[L] = str(row[key]).strip()

    # ברירת מחדל לצבעים אם לא הוגדרו בקובץ
    correct = str(row.get("QCorrectAnswer", "")).strip().upper()
    default_gray = "#6b7280"
    correct_green = "#22c55e"
    if not colors:
        colors = {L: (correct_green if L == correct else default_gray) for L in letters}

    # החזר לפי סדר A..E
    x = letters
    y = [vals[L] for L in letters]
    c = [colors[L] for L in letters]
    return x, y, c

# === NEW === גרף Plotly במקום תמונה
def _render_graph_block(title_html, question_text, row_dict):
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(f"### {question_text}")

    # חילוץ ערכי A..E וצבעים
    try:
        x, y, colors = _extract_option_values_and_colors(row_dict)
    except Exception as e:
        # אם אין עמודות ערכים – ננסה להציג את התמונה הישנה (תאימות לאחור)
        img = load_image(row_dict.get("ImageFileName", ""))
        if img is not None:
            target_w = min(GRAPH_MAX_WIDTH_PX, img.width)
            left, mid, right = st.columns([1, 6, 1])
            with mid:
                st.image(img, width=target_w)
            st.info("טיפ: ניתן לעבור ל'גרף בקוד' ע\"י הוספת עמודות ValueA..ValueE (ואופציונלית ColorA..ColorE) לקובץ.")
            return
        else:
            st.error(f"שגיאת גרף: {e}")
            return

    # ציור גרף (ללא כפתור מסך-מלא)
    fig = go.Figure(go.Bar(
        x=x,
        y=y,
        text=y,
        textposition="outside",
        marker_color=colors,
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="", yaxis_title="",
        showlegend=False,
        bargap=0.35,
        uniformtext_minsize=12, uniformtext_mode="hide",
    )
    left, mid, right = st.columns([1, 6, 1])
    with mid:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def _response_buttons_and_timer(timeout_sec, on_timeout, on_press):
    if not st.session_state.get("awaiting_response", False):
        return

    elapsed = time.time() - (st.session_state.t_start or time.time())
    remain = max(0, timeout_sec - int(elapsed))

    if elapsed >= timeout_sec and st.session_state.awaiting_response:
        st.session_state.awaiting_response = False
        on_timeout()
        st.stop()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    outer = st.columns([1, 6, 1])
    with outer[1]:
        row = st.columns(5)        # חמישה כפתורים בשורה
        labels = ["E", "D", "C", "B", "A"]
        unique = f"{st.session_state.i}_{int(st.session_state.t_start or 0)}"
        for i, lab in enumerate(labels):
            if row[i].button(lab, use_container_width=True, key=f"btn_{lab}_{unique}"):
                if st.session_state.awaiting_response:
                    st.session_state.awaiting_response = False
                    on_press(lab)
                    st.stop()

    st.markdown(
        f"<div style='text-align:center; margin-top:12px;'>⏳ זמן שנותר: <b>{remain}</b> שניות</div>",
        unsafe_allow_html=True,
    )

    time.sleep(1)
    st.rerun()


# ===== Helper: clickable logo via base64 =====
def _file_to_base64_html_img_link(path: str, href: str, width_px: int = 140) -> str:
    try:
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return (
            f"<a href='{href}' target='_blank'>"
            f"<img src='data:{mime};base64,{b64}' style='width:{width_px}px; border:0;'/>"
            f"</a>"
        )
    except Exception:
        return ""


# ========= Screens =========
def screen_welcome():
    st.title("ניסוי בזיכרון חזותי של גרפים 📊")
    st.markdown(
        """
**שלום וברוכ/ה הבא/ה לניסוי**  

במהלך הניסוי יוצגו **40 גרפים** שלגביהם תתבקש/י לציין מהו הערך הנמוך ביותר או הגבוה ביותר בגרף.

חשוב לענות מהר ככל שניתן; לאחר **30 שניות**, אם לא נבחרה תשובה, יהיה מעבר אוטומטי לשאלה הבאה.

**איך עונים?**  
לוחצים על האות המתאימה מתחת לגרף **A / B / C / D / E**.

לפני תחילת הניסוי, תוצג **שאלת תרגול אחת** (לא נשמרת בתוצאות).

כדי להתחיל – לחצו על **המשך לתרגול**.
"""
    )

    if not os.path.exists(DATA_PATH):
        st.error(f"לא נמצא הקובץ: {DATA_PATH}.")
        st.stop()

    try:
        df = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    if st.button("המשך לתרגול"):
        _ensure_participant_id()
        st.session_state.run_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")

        practice_item = df.iloc[0].to_dict()

        pool_df = df.iloc[1: 1 + N_TRIALS].copy()
        trials = build_alternating_trials(pool_df, N_TRIALS)

        st.session_state.df = df
        st.session_state.practice = practice_item
        st.session_state.trials = trials
        st.session_state.i = 0
        st.session_state.t_start = None
        st.session_state.results = []
        st.session_state.saved_to_sheets = False
        st.session_state.page = "practice"
        st.rerun()

def screen_practice():
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()
        st.session_state.awaiting_response = True

    t = st.session_state.practice
    title_html = "<div style='font-size:20px; font-weight:700; text-align:right; margin-bottom:0.5rem;'>תרגול</div>"
    _render_graph_block(title_html, t["QuestionText"], t)

    def on_timeout():
        st.session_state.t_start = None
        st.session_state.page = "trial"
        st.rerun()

    def on_press(_):
        st.session_state.t_start = None
        st.session_state.page = "trial"
        st.rerun()

    _response_buttons_and_timer(TRIAL_TIMEOUT_SEC, on_timeout, on_press)

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
        if st.session_state.i + 1 < len(st.session_state.trials):
            st.session_state.i += 1
            st.rerun()
        else:
            st.session_state.page = "end"
            st.rerun()

    def on_timeout():
        finish_with(resp_key=None, rt_sec=float(TRIAL_TIMEOUT_SEC), correct=0)

    def on_press(key):
        rt = time.time() - (st.session_state.t_start or time.time())
        correct = key.strip().upper() == str(t["QCorrectAnswer"]).strip().upper()
        finish_with(resp_key=key.strip().upper(), rt_sec=rt, correct=correct)

    _response_buttons_and_timer(TRIAL_TIMEOUT_SEC, on_timeout, on_press)

def screen_end():
    st.title("סיום הניסוי")
    st.success("תודה על השתתפותך!")

    df = pd.DataFrame(st.session_state.results)

    admin = is_admin()

    if not st.session_state.saved_to_sheets and not df.empty:
        try:
            append_dataframe_to_gsheet(df, GSHEET_ID, worksheet_name=GSHEET_WORKSHEET_NAME)
            st.session_state.saved_to_sheets = True
            st.success("התשובות נשלחו בהצלחה ✅")
            if admin:
                st.caption("נשמר ל-Google Sheets (למנהל/ת בלבד).")
        except Exception as e:
            if admin:
                st.error(f"נכשלה כתיבה ל-Google Sheets: {type(e).__name__}: {e}")
            else:
                st.info("התשובות נשלחו. אם יידרש, נבצע שמירה חוזרת מאחורי הקלעים.")
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

    if admin:
        st.download_button(
            "הורדת תוצאות (CSV)",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"{st.session_state.participant_id}_{st.session_state.run_start_iso.replace(':','-')}.csv",
            mime="text/csv",
        )
        st.link_button(
            "פתח/י את Google Sheet",
            f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit",
            type="primary",
        )

# ========= Router =========
page = st.session_state.page
if page == "welcome":
    screen_welcome()
elif page == "practice":
    screen_practice()
elif page == "trial":
    screen_trial()
else:
    screen_end()
