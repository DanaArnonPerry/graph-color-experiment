# app.py
import os
import time
import random
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO

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

# ========= (Optional) Brand assets =========
LOGO_CANDIDATES = [
    "images/Logo.png", "images/logo.png",
    "images/Logo29.10.24_B.png", "Logo.png", "Logo"
]
USER_PHOTO_CANDIDATES = [
    "images/DanaSherlok.png", "images/DanaSherlok.jpg",
    "DanaSherlok.png", "DanaSherlok.jpg", "DanaSherlok"
]
WEBSITE_URL = ""  # קישור אתר במסך הסיום (השאירי ריק אם לא צריך)

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
</style>
""",
    unsafe_allow_html=True,
)

# ========= Session State =========
def init_state():
    ss = st.session_state
    ss.setdefault("page", "welcome")     # welcome -> practice_intro -> practice -> trial -> end
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

init_state()

# ========= Admin PIN =========
def is_admin():
    with st.sidebar:
        if LOGO_PATH:
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown("**🔐 אזור מנהל**")
        if not st.session_state.is_admin:
            pin = st.text_input("הכנסי PIN:", type="password")
            if st.button("כניסה"):
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
    return df

# ========= Google Sheets helpers =========
def _read_service_account_from_secrets() -> dict:
    """
    תומך בשני פורמטים של secrets:
    1) סעיף [service_account] (מומלץ)
    2) מפתחות SA שטוחים בראש הקובץ + [admin]
    """
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
    """ודאי שתהיה שורת כותרת נכונה; אם חסרה/שגויה – נעדכן את השורה הראשונה."""
    current = ws.get_all_values()
    headers = list(expected_headers)
    if not current:
        ws.append_row(headers)
        return
    first_row = current[0]
    if first_row != headers:
        ws.update("1:1", [headers])

def get_next_participant_seq(sheet_id: str) -> int:
    """
    קורא/מגדיל מונה בגיליון 'Meta' (תא A2). אם אינו קיים – ייווצר.
    מחזיר את המספר הבא (1, 2, 3 ...).
    """
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
    """מנסה להימנע מ-V זהה פעמיים ברצף; אם אין איזון — ייתכן רצף."""
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
    return result

def _render_graph_block(title_html, question_text, image_file):
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(f"### {question_text}")
    img = load_image(image_file)
    if img is not None:
        st.image(img, use_container_width=True)

def _response_buttons_and_timer(timeout_sec, on_timeout, on_press):
    elapsed = time.time() - (st.session_state.t_start or time.time())
    remain = max(0, timeout_sec - int(elapsed))
    st.write(f"⏳ זמן שנותר: **{remain}** שניות")
    if elapsed >= timeout_sec:
        on_timeout()
        st.stop()

    # ריווח צדדים + 5 כפתורים (A הכי שמאלי)
    cols = st.columns([0.10, 1, 1, 1, 1, 1, 0.10])
    for idx, lab in enumerate(["A", "B", "C", "D", "E"], start=1):
        if cols[idx].button(lab, use_container_width=True):
            on_press(lab)
            st.stop()

    # רענון כל שנייה להצגת הטיימר
    time.sleep(1)
    st.rerun()

# ========= Screens =========
def screen_welcome():
    st.title("ניסוי בזיכרון חזותי של גרפים 📊")
    st.markdown(
        """
**שלום וברוכ/ה הבא/ה לניסוי**  
.במהלך הניסוי יוצגו **40 גרפים** שלגביהם תתבקש/י לציין מהו הערך הנמוך ביותר או הגבוה ביותר בגרף 
חשוב לענות מהר ככל שניתן, לאחר 30 שניות, אם לא נבחרה תשובה, יהיה מעבר אוטומטי לשאלה הבאה.   

**איך עונים?**

לוחצים על האות המתאימה מתחת לגרף A / B / C / D / E.

לפני תחילת הניסוי תהיה **שאלת תרגול אחת**.

להתחלת הניסוי יש ללחוץ על  **המשך לתרגול**.
"""
    )

    # מזהה נבדק אוטומטי ורץ (S00001, S00002, ...)
    if not st.session_state.participant_id:
        try:
            seq = get_next_participant_seq(GSHEET_ID)
            st.session_state.participant_id = f"S{seq:05d}"
        except Exception:
            # נפלנו על הרשאות/אופליין – נשתמש בזמן כמזהה חד־פעמי
            st.session_state.participant_id = f"S{int(time.time())}"

    st.info(f"**מזהה נבדק הוקצה אוטומטית:** {st.session_state.participant_id}")

    if not os.path.exists(DATA_PATH):
        st.error(f"לא נמצא הקובץ: {DATA_PATH}.")
        st.stop()
    df = load_data()

    if st.button("המשך לתרגול"):
        st.session_state.run_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")

        # תרגול = תמיד השורה הראשונה
        practice_item = df.iloc[0].to_dict()

        # ניסויים = 40 השורות הבאות (עם ניסיון לאזן קבוצות V)
        pool_df = df.iloc[1 : 1 + N_TRIALS].copy()
        trials = build_alternating_trials(pool_df, N_TRIALS)

        st.session_state.df = df
        st.session_state.practice = practice_item
        st.session_state.trials = trials
        st.session_state.i = 0
        st.session_state.t_start = None
        st.session_state.results = []
        st.session_state.page = "practice_intro"
        st.rerun()

def screen_practice_intro():
    st.title("תרגול – הוראות קצרות")
    st.markdown(
        """
התרגול הבא **לא נשמר** לתוצאות. מטרתו לוודא שהבנת בדיוק מה לעשות:

- קראי את השאלה בראש המסך (נמוך/גבוה ביותר).
- הסתכלי על הגרף.
- לחצי על הכפתור **A–E** שמתאים לעמודה הנכונה (A הכי שמאלית).
- יש **30 שניות** לכל מסך.

כשתהיי מוכנה, לחצי על **התחלת תרגול**.
"""
    )
    if st.button("התחלת תרגול"):
        st.session_state.page = "practice"
        st.rerun()

def screen_practice():
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()
    t = st.session_state.practice
    title_html = "<div style='font-size:20px; font-weight:700; text-align:center; margin-bottom:0.5rem;'>תרגול</div>"
    _render_graph_block(title_html, t["QuestionText"], t["ImageFileName"])

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
    i = st.session_state.i
    t = st.session_state.trials[i]

    title_html = f"<div style='font-size:20px; font-weight:700; text-align:center; margin-bottom:0.5rem;'>גרף מספר {i+1}</div>"
    _render_graph_block(title_html, t["QuestionText"], t["ImageFileName"])

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

    # שמירה ל-Google Sheets בלבד
    try:
        append_dataframe_to_gsheet(df, GSHEET_ID, worksheet_name=GSHEET_WORKSHEET_NAME)
        st.caption("התוצאות נשמרו ל-Google Sheets (פרטי).")
    except Exception as e:
        st.info(f"לא נשמר ל-Google Sheets (בדקו secrets/שיתוף): {e}")

    # אזור מנהל בלבד: הורדת CSV + קישור
    if is_admin():
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

    # חתימת מותג עדינה (אופציונלי)
    cols = st.columns([1, 1, 1])
    with cols[1]:
        if USER_PHOTO_PATH:
            st.image(USER_PHOTO_PATH, width=120)
        if WEBSITE_URL:
            st.markdown(
                f"<div style='text-align:center; margin-top:8px;'><a href='{WEBSITE_URL}' target='_blank' style='text-decoration:underline;'>לאתר שלי</a></div>",
                unsafe_allow_html=True,
            )

# ========= Router =========
page = st.session_state.page
if page == "welcome":
    screen_welcome()
elif page == "practice_intro":
    screen_practice_intro()
elif page == "practice":
    screen_practice()
elif page == "trial":
    screen_trial()
else:
    screen_end()
