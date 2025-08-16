# app.py
import os
import time
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO

# ========= Parameters =========
N_TRIALS = 40
TRIAL_TIMEOUT_SEC = 30
RESPONSE_KEYS = ["A", "B", "C", "D", "E"]  # סדר לוגי

# ========= Required Columns (exactly as in Colors in charts.csv) =========
REQUIRED_COLS = [
    "ID", "V", "ConditionFull", "Color", "Condition", "LowowOrHhigh",
    "ChartNumber", "A", "B", "C", "D", "E",
    "ImageFileName", "QuestionText", "QCorrectAnswer"
]

# ========= Data path (fixed, no uploader) =========
DATA_PATH = "data/colors_in_charts.csv"

# ========= Brand assets =========
LOGO_CANDIDATES = [
    "images/Logo.png", "images/logo.png", "Logo.png", "Logo", "images/Logo29.10.24_B.png"
]
USER_PHOTO_CANDIDATES = [
    "images/DanaSherlok.png", "images/DanaSherlok.jpg",
    "DanaSherlok.png", "DanaSherlok.jpg", "DanaSherlok"
]
WEBSITE_URL = "https://example.com"  # <<< עדכני לכתובת האתר שלך

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

LOGO_PATH = first_existing(LOGO_CANDIDATES)
USER_PHOTO_PATH = first_existing(USER_PHOTO_CANDIDATES)

# ========= Page Setup =========
st.set_page_config(page_title="ניסוי גרפים", page_icon="📊", layout="centered")
st.markdown("""
<style>
html, body, [class*="css"]  { direction: rtl; text-align: right; font-family: "Rubik","Segoe UI","Arial",sans-serif; }
blockquote, pre, code { direction: ltr; text-align: left; }
</style>
""", unsafe_allow_html=True)

# ========= Admin (Sidebar + URL) =========
try:
    ADMIN_FROM_URL = str(st.query_params.get("admin", "0")).lower() in ("1", "true", "yes")
except Exception:
    ADMIN_FROM_URL = False

st.sidebar.header("⚙️ תפריט מנהל")
ADMIN_MODE = st.sidebar.checkbox("Admin Mode", value=ADMIN_FROM_URL, help="הצגת כלי בדיקה ותוצאות")

# לוגו – בסיידבר, עדין ולא מפריע
if LOGO_PATH:
    st.sidebar.image(LOGO_PATH, use_container_width=True)

# ברירות מחדל לריפודי הכפתורים
DEFAULT_LEFT_PAD = 0.10
DEFAULT_RIGHT_PAD = 0.10

# אם מנהל - אפשר לכוון בזמן אמת
if ADMIN_MODE:
    st.sidebar.subheader("יישור כפתורים לגרף")
    LEFT_PAD = st.sidebar.slider("ריפוד שמאל (אחוזי רוחב)", 0.00, 0.30, DEFAULT_LEFT_PAD, 0.01)
    RIGHT_PAD = st.sidebar.slider("ריפוד ימין (אחוזי רוחב)", 0.00, 0.30, DEFAULT_RIGHT_PAD, 0.01)
else:
    LEFT_PAD = DEFAULT_LEFT_PAD
    RIGHT_PAD = DEFAULT_RIGHT_PAD

# ========= Session State =========
def init_state():
    ss = st.session_state
    ss.setdefault("page", "welcome")     # welcome -> practice -> trial -> end
    ss.setdefault("df", None)
    ss.setdefault("practice", None)      # פריט התרגול (dict)
    ss.setdefault("trials", None)        # פריטי הניסוי (list[dict])
    ss.setdefault("i", 0)                # אינדקס ניסוי (לא כולל תרגול)
    ss.setdefault("t_start", None)
    ss.setdefault("results", [])
    ss.setdefault("image_cache", {})
    ss.setdefault("debug_log", [])

def log_debug(msg: str):
    ts = pd.Timestamp.now().isoformat()
    st.session_state.debug_log.append(f"[{ts}] {msg}")

init_state()

# ========= Data Loading =========
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH, encoding="utf-8")
    except Exception:
        return pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# ========= Utilities =========
def load_image(path: str):
    if not path:
        return None
    if path in st.session_state.image_cache:
        return st.session_state.image_cache[path]
    try:
        if path.startswith(("http://", "https://")):
            r = requests.get(path, timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGBA")
        else:
            img = Image.open(path).convert("RGBA")
        st.session_state.image_cache[path] = img
        return img
    except Exception:
        return None

def preflight_check(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.head(N_TRIALS).iterrows():
        issues = []
        ans = str(row.get("QCorrectAnswer","")).strip().upper()
        valid_ans = ans in {"A","B","C","D","E"}
        if not valid_ans:
            issues.append("QCorrectAnswer not in A–E")

        img = str(row.get("ImageFileName","")).strip()
        image_ok = True
        if img:
            if img.startswith(("http://","https://")):
                try:
                    r = requests.head(img, timeout=5)
                    image_ok = (r.status_code < 400)
                except Exception:
                    image_ok = False
            else:
                image_ok = os.path.exists(img)
        else:
            image_ok = False
            issues.append("ImageFileName empty")

        if not image_ok and "ImageFileName empty" not in issues:
            issues.append("Image not found / URL error")

        rows.append({
            "RowIndex": idx+1,
            "TrialID": row.get("ID",""),
            "ImageFileName": img,
            "QCorrectAnswer": ans,
            "ValidAnswer": valid_ans,
            "ImageExists": image_ok,
            "Issues": "; ".join(issues) if issues else ""
        })
    return pd.DataFrame(rows)

# ========= Screens =========
def screen_welcome():
    st.title("ניסוי גרפים")
    st.write("""
    **הנחיות:**  
    יוצגו לך 40 גרפים. בכל מסך עליך לזהות את העמודה עם הערך הנמוך או הגבוה ביותר (לפי השאלה).  
    יש להשיב מהר ככל האפשר. אם לא תהיה תגובה ב־30 שניות, עוברים אוטומטית לגרף הבא.
    """)

    # טוענים את הקובץ — לא מציגים כלום למשתתף רגיל
    if not os.path.exists(DATA_PATH):
        st.error(f"לא נמצא הקובץ: {DATA_PATH}.")
        st.stop()
    try:
        df = load_data()
    except Exception as e:
        st.error(f"שגיאה בקריאת הקובץ: {e}")
        st.stop()

    # כלי מנהל בלבד
    if ADMIN_MODE:
        st.success("הקובץ נטען בהצלחה!")
        with st.expander("תצוגה מקדימה (אופציונלי)"):
            if st.checkbox("הצג 5 שורות ראשונות", value=False):
                st.dataframe(df.head(), use_container_width=True, hide_index=True)

        if st.button("בדיקת תקינות (Preflight)"):
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                st.error("חסרות עמודות בקובץ: " + ", ".join(missing))
            elif len(df) < (N_TRIALS + 1):
                st.error(f"הקובץ צריך לכלול לפחות {N_TRIALS+1} שורות (1 תרגול + {N_TRIALS} ניסויים).")
            else:
                rep = preflight_check(df)
                bad = rep[(~rep["ValidAnswer"]) | (~rep["ImageExists"])]
                st.success("בדיקת תקינות הושלמה.")
                with st.expander("דו״ח Preflight"):
                    st.dataframe(rep, use_container_width=True, hide_index=True)
                if len(bad) > 0:
                    st.warning("נמצאו בעיות בחלק מהשורות הראשונות. מומלץ לתקן לפני הרצה.")

    # כפתור המשך (לכולם) — יכין תרגול וניסוי
    if st.button("המשך"):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error("חסרות עמודות בקובץ: " + ", ".join(missing)); return
        if len(df) < (N_TRIALS + 1):
            st.error(f"הקובץ צריך לכלול לפחות {N_TRIALS+1} שורות (1 תרגול + {N_TRIALS} ניסויים)."); return

        df = df.fillna("").astype({c: str for c in df.columns})
        practice_item = df.iloc[0].to_dict()
        trials = df.iloc[1:N_TRIALS+1].to_dict(orient="records")

        st.session_state.df = df
        st.session_state.practice = practice_item
        st.session_state.trials = trials
        st.session_state.i = 0
        st.session_state.t_start = None
        st.session_state.results = []
        st.session_state.page = "practice"
        st.rerun()

def _render_graph_block(title_html, question_text, image_file):
    # כותרת קטנה ומרוכזת
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(f"### {question_text}")

    img = load_image(image_file)
    if img is None and image_file:
        st.warning(f"לא ניתן לטעון תמונה: {image_file}")
    if img is not None:
        st.image(img, use_container_width=True)

def _response_buttons_and_timer(timeout_sec, on_timeout, on_press):
    elapsed = time.time() - (st.session_state.t_start or time.time())
    remain = max(0, timeout_sec - int(elapsed))
    st.write(f"⏳ זמן שנותר: **{remain}** שניות")
    if elapsed >= timeout_sec:
        on_timeout()
        st.stop()

    cols = st.columns([LEFT_PAD, 1, 1, 1, 1, 1, RIGHT_PAD])
    labels = ["A", "B", "C", "D", "E"]
    for idx, lab in enumerate(labels, start=1):
        if cols[idx].button(lab, use_container_width=True):
            on_press(lab)
            st.stop()

    time.sleep(1)
    st.rerun()

# -------- Practice screen (one warm-up trial) --------
def screen_practice():
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()

    t = st.session_state.practice
    title_html = "<div style='font-size:20px; font-weight:700; text-align:center; margin-bottom:0.5rem;'>תרגול</div>"
    _render_graph_block(title_html, t["QuestionText"], t["ImageFileName"])

    def on_timeout():
        # בתרגול לא שומרים תוצאה
        st.session_state.t_start = None
        st.session_state.page = "trial"
        st.rerun()

    def on_press(key):
        # בתרגול לא שומרים תוצאה
        st.session_state.t_start = None
        st.session_state.page = "trial"
        st.rerun()

    _response_buttons_and_timer(TRIAL_TIMEOUT_SEC, on_timeout, on_press)

# -------- Actual trial screen --------
def screen_trial():
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()

    i = st.session_state.i
    t = st.session_state.trials[i]

    title_html = f"<div style='font-size:20px; font-weight:700; text-align:center; margin-bottom:0.5rem;'>גרף מספר {i+1}</div>"
    _render_graph_block(title_html, t["QuestionText"], t["ImageFileName"])

    def on_timeout():
        finish_trial(resp_key=None, rt_sec=TRIAL_TIMEOUT_SEC, correct=0)

    def on_press(key):
        t0 = st.session_state.t_start or time.time()
        rt = time.time() - t0
        acc = int(key == str(t["QCorrectAnswer"]).strip().upper())
        finish_trial(resp_key=key, rt_sec=rt, correct=acc)

    _response_buttons_and_timer(TRIAL_TIMEOUT_SEC, on_timeout, on_press)

def finish_trial(resp_key: str | None, rt_sec: float | None, correct: int):
    t = st.session_state.trials[st.session_state.i]
    st.session_state.results.append({
        "TrialIndex": st.session_state.i + 1,
        "ID": t["ID"], "V": t["V"], "ConditionFull": t["ConditionFull"],
        "Color": t["Color"], "Condition": t["Condition"],
        "LowowOrHhigh": t["LowowOrHhigh"], "ChartNumber": t["ChartNumber"],
        "A": t["A"], "B": t["B"], "C": t["C"], "D": t["D"], "E": t["E"],
        "ImageFileName": t["ImageFileName"], "QuestionText": t["QuestionText"],
        "ResponseKey": resp_key or "", "QCorrectAnswer": t["QCorrectAnswer"],
        "Accuracy": correct, "RT_ms": int(rt_sec * 1000) if rt_sec is not None else "",
        "Timestamp": pd.Timestamp.now().isoformat()
    })
    st.session_state.t_start = None
    if st.session_state.i + 1 < len(st.session_state.trials):
        st.session_state.i += 1; st.rerun()
    else:
        st.session_state.page = "end"; st.rerun()

def screen_end():
    st.title("סיום הניסוי")
    st.success("תודה על השתתפותך!")

    # פרטי מותג עדינים למטה (לא מפריע למשתתף)
    cols = st.columns([1,1,1])
    with cols[1]:
        if USER_PHOTO_PATH:
            st.image(USER_PHOTO_PATH, width=120)  # בלי use_column_width (דפריקייטד)
        if WEBSITE_URL:
            st.markdown(
                f"<div style='text-align:center; margin-top:8px;'>"
                f"<a href='{WEBSITE_URL}' target='_blank' style='text-decoration:underline;'>לאתר שלי</a>"
                f"</div>",
                unsafe_allow_html=True
            )

    # תוצאות למנהל בלבד
    if ADMIN_MODE:
        df = pd.DataFrame(st.session_state.results)
        st.info("מצב מנהל — הצגת תוצאות:")
        st.subheader("תוצאות גולמיות")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty:
            st.subheader("סיכום לפי Color")
            agg = (
                df.assign(RT_ms=pd.to_numeric(df["RT_ms"], errors="coerce"))
                  .groupby("Color", dropna=False)
                  .agg(Mean_RT_ms=("RT_ms", "mean"),
                       SD_RT_ms=("RT_ms", "std"),
                       Accuracy_pct=("Accuracy", lambda s: 100*s.mean() if len(s)>0 else 0))
                  .round(2)
                  .reset_index()
            )
            st.dataframe(agg, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("הורדת תוצאות (CSV)", data=csv_bytes,
                           file_name=f"results_{int(time.time())}.csv", mime="text/csv")

        from io import BytesIO
        xbuf = BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="RawResults", index=False)
            try:
                agg.to_excel(xw, sheet_name="SummaryByColor", index=False)
            except Exception:
                pass
        st.download_button("הורדת תוצאות (Excel)", data=xbuf.getvalue(),
                           file_name=f"results_{int(time.time())}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
