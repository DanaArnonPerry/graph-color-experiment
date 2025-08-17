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
REQUIRED_COLS = ["ID", "ImageFileName", "QuestionText", "QCorrectAnswer"]

# ========= (Optional) Brand assets =========
LOGO_CANDIDATES = [
    "images/Logo.png", "images/logo.png",
    "images/Logo29.10.24_B.png", "Logo.png", "Logo"
]
# כולל גם fallback לכתיב הישן Sherlok
USER_PHOTO_CANDIDATES = [
    "images/DanaSherlock.png", "images/DanaSherlock.jpg",
    "images/DanaSherlok.png", "images/DanaSherlok.jpg",
    "DanaSherlock.png", "DanaSherlock.jpg",
    "DanaSherlok.png", "DanaSherlok.jpg",
]
WEBSITE_URL = "http://www.2dpoint.co.il"  # קישור אתר במסך הסיום

# תמונת שרלוק מתוך Github (fallback)
SHERLOCK_GITHUB_URL = (
    "https://raw.githubusercontent.com/danaarnonperry/graph-color-experiment/main/images/DanaSherlock.png"
)
SHERLOCK_IMG_WIDTH = 160  # רוחב תצוגה לתמונת שרלוק במסך הסיום

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
    # אנטי-כפילויות
    ss.setdefault("awaiting_response", False)  # מחכים לתשובה בסבב הנוכחי
    ss.setdefault("saved_to_sheets", False)    # נשמר כבר למסך הסיום
init_state()

# ========= Admin PIN =========
def is_admin(show_ui: bool = False):
    """מחזיר האם מנהל מחובר. מציג UI ב-sidebar רק אם show_ui=True."""
    if show_ui:
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

    # בדיקת עמודות נדרשות
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"בעיית עמודות בקובץ הנתונים: חסרות {', '.join(missing)}")

    return df

# ========= Google Sheets helpers =========
def _read_service_account_from_secrets() -> dict:
    """תומך גם ב-[service_account] וגם במפתחות SA בטופ-לבל."""
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
    """מונה רץ ב־'Meta'!A2 (נוצר אוטומטית אם לא קיים)."""
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
    """מזהה נבדק אוטומטי מאחורי הקלעים (לא מוצג)."""
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
    """אם קיימת עמודת V – ננסה לא לחזור עליה פעמיים רצוף; אחרת – דגימה אקראית."""
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

# -------- גרף: הגדרת רוחב מקסימלי ומרכוז --------
GRAPH_MAX_WIDTH_PX = 1500  # מקסימום רוחב תצוגה

def _render_graph_block(title_html, question_text, image_file):
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(f"### {question_text}")
    img = load_image(image_file)
    if img is None:
        return
    target_w = min(GRAPH_MAX_WIDTH_PX, img.width)
    left, mid, right = st.columns([1, 6, 1])
    with mid:
        st.image(img, width=target_w)

def _response_buttons_and_timer(timeout_sec, on_timeout, on_press):
    # מציגים כפתורים וטיימר רק אם מחכים לתשובה
    if not st.session_state.get("awaiting_response", False):
        return

    # חישוב זמן שנותר
    elapsed = time.time() - (st.session_state.t_start or time.time())
    remain = max(0, timeout_sec - int(elapsed))

    # אם הזמן נגמר – פעולה חד-פעמית
    if elapsed >= timeout_sec and st.session_state.awaiting_response:
        st.session_state.awaiting_response = False
        on_timeout()
        st.stop()

    # כפתורי התשובה (עם ריווח בצדדים) — A הכי שמאלי
    cols = st.columns([0.10, 1, 1, 1, 1, 1, 0.10])
    trial_index = st.session_state.i
    start_key = int(st.session_state.t_start or 0)
    for idx, lab in enumerate(["A", "B", "C", "D", "E"], start=1):
        if cols[idx].button(lab, key=f"resp_{trial_index}_{lab}_{start_key}", use_container_width=True):
            if st.session_state.awaiting_response:  # הגנה נוספת
                st.session_state.awaiting_response = False
                on_press(lab)
                st.stop()

    # הטיימר מתחת לאפשרויות
    st.markdown(
        f"<div style='text-align:center; margin-top:12px;'>⏳ זמן שנותר: "
        f"<b>{remain}</b> שניות</div>",
        unsafe_allow_html=True,
    )

    # רענון פעם בשנייה כל עוד מחכים
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

    # נטען כבר כאן כדי להתריע מוקדם על חסרים/שמות עמודות
    try:
        df = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    if st.button("המשך לתרגול"):
        # הקצאת מזהה נבדק אוטומטית ושקטה
        _ensure_participant_id()
        st.session_state.run_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")

        # תרגול = תמיד השורה הראשונה
        practice_item = df.iloc[0].to_dict()

        # ניסויים = 40 השורות הבאות
        pool_df = df.iloc[1: 1 + N_TRIALS].copy()
        trials = build_alternating_trials(pool_df, N_TRIALS)

        st.session_state.df = df
        st.session_state.practice = practice_item
        st.session_state.trials = trials
        st.session_state.i = 0
        st.session_state.t_start = None
        st.session_state.results = []
        st.session_state.saved_to_sheets = False  # איפוס לביטול כפילויות במסך הסיום
        st.session_state.page = "practice"
        st.rerun()

def screen_practice():
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()
        st.session_state.awaiting_response = True  # מחכים לתשובה

    t = st.session_state.practice
    title_html = "<div style='font-size:20px; font-weight:700; text-align:right; margin-bottom:0.5rem;'>תרגול</div>"
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
        st.session_state.awaiting_response = True  # מחכים לתשובה

    i = st.session_state.i
    t = st.session_state.trials[i]

    title_html = f"<div style='font-size:20px; font-weight:700; text-align:right; margin-bottom:0.5rem;'>גרף מספר {i+1}</div>"
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

    # מציגים UI של מנהל רק אם הגענו עם ?admin=1
    admin = is_admin(show_ui=_admin_ui_enabled())
    # קריאה אחת ל-is_admin
    admin = is_admin()

    # שמירה ל-Google Sheets – חד-פעמית; למשתתפים מציגים רק הודעה כללית
    if not st.session_state.saved_to_sheets and not df.empty:
        try:
            append_dataframe_to_gsheet(df, GSHEET_ID, worksheet_name=GSHEET_WORKSHEET_NAME)
            st.session_state.saved_to_sheets = True
            st.success("התשובות נשלחו בהצלחה ✅")
            if admin:
                st.caption("נשמר ל-Google Sheets (נראה רק למנהל/ת).")
        except Exception as e:
            if admin:
                st.error(f"נכשלה כתיבה ל-Google Sheets: {type(e).__name__}: {e}")
            else:
                st.info("התשובות נשלחו. אם יידרש, נבצע שמירה חוזרת מאחורי הקלעים.")
    else:
        st.success("התשובות נשלחו בהצלחה ✅")

    # ===== תמונת שרלוק: מקומי אם קיים, אחרת מגיטהאב =====
    try:
        cols = st.columns([1, 1, 1])
        with cols[1]:
            sherlock_src = USER_PHOTO_PATH or SHERLOCK_GITHUB_URL
            img = load_image(sherlock_src)
            if img is not None:
                st.image(img, width=SHERLOCK_IMG_WIDTH)
    except Exception:
        pass

    # ===== לוגו לחיץ אל האתר =====
    if LOGO_PATH and WEBSITE_URL:
        html = _file_to_base64_html_img_link(LOGO_PATH, WEBSITE_URL, width_px=140)
        if html:
            st.markdown(f"<div style='text-align:center; margin-top:10px;'>{html}</div>", unsafe_allow_html=True)
        else:
            st.link_button("לאתר שלי", WEBSITE_URL, type="primary")
    elif WEBSITE_URL:
        st.link_button("לאתר שלי", WEBSITE_URL, type="primary")

    # אזור מנהל בלבד: הורדת CSV + קישור ישיר לגיליון
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
