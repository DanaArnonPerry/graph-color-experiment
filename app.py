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
RESPONSE_KEYS = ["A", "B", "C", "D", "E"]  # סדר לוגי; העימוד קובע מי שמאלי/ימני

# ========= Required Columns (exactly as in Colors in charts.csv) =========
REQUIRED_COLS = [
    "ID", "V", "ConditionFull", "Color", "Condition", "LowowOrHhigh",
    "ChartNumber", "A", "B", "C", "D", "E",
    "ImageFileName", "QuestionText", "QCorrectAnswer"
]

# ========= Data path (fixed, no uploader) =========
DATA_PATH = "data/colors_in_charts.csv"

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

# ========= Session State =========
def init_state():
    ss = st.session_state
    ss.setdefault("page", "welcome")
    ss.setdefault("df", None)
    ss.setdefault("trials", None)
    ss.setdefault("i", 0)
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
    st.title("ניסוי גרפים – גרסה פשוטה ומדויקת")
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
            elif len(df) < N_TRIALS:
                st.error(f"הקובץ מכיל פחות מ-{N_TRIALS} שורות.")
            else:
                rep = preflight_check(df)
                bad = rep[(~rep["ValidAnswer"]) | (~rep["ImageExists"])]
                st.success("בדיקת תקינות הושלמה.")
                with st.expander("דו״ח Preflight"):
                    st.dataframe(rep, use_container_width=True, hide_index=True)
                if len(bad) > 0:
                    st.warning("נמצאו בעיות בחלק מהשורות הראשונות. מומלץ לתקן לפני הרצה.")

    # כפתור המשך (לכולם)
    if st.button("המשך"):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error("חסרות עמודות בקובץ: " + ", ".join(missing)); return
        if len(df) < N_TRIALS:
            st.error(f"הקובץ מכיל פחות מ-{N_TRIALS} שורות."); return

        df = df.fillna("").astype({c: str for c in df.columns})
        trials = df.iloc[:N_TRIALS].to_dict(orient="records")

        st.session_state.df = df
        st.session_state.trials = trials
        st.session_state.i = 0
        st.session_state.t_start = None
        st.session_state.results = []
        st.session_state.page = "trial"
        st.rerun()

def screen_trial():
    if st.session_state.t_start is None:
        st.session_state.t_start = time.time()

    i = st.session_state.i
    t = st.session_state.trials[i]

    st.subheader(f"גרף מספר {i+1}")
    st.markdown(f"### {t['QuestionText']}")

    img = load_image(t["ImageFileName"])
    if img is None and t["ImageFileName"]:
        st.warning(f"לא ניתן לטעון תמונה: {t['ImageFileName']}")
    if img is not None:
        st.image(img, use_container_width=True)

    elapsed = time.time() - (st.session_state.t_start or time.time())
    remain = max(0, TRIAL_TIMEOUT_SEC - int(elapsed))
    st.write(f"⏳ זמן שנותר: **{remain}** שניות")
    if elapsed >= TRIAL_TIMEOUT_SEC:
        finish_trial(resp_key=None, rt_sec=TRIAL_TIMEOUT_SEC, correct=0)
        st.stop()

    # כפתורי תשובה בלבד — A הכי שמאלי
    cols = st.columns(5)
    keys_layout = ["E", "D", "C", "B", "A"]  # אם יוצא הפוך אצלך, החליפי ל-["E","D","C","B","A"]
    for idx, label in enumerate(keys_layout):
        if cols[idx].button(label, use_container_width=True):
            handle_response(label); st.stop()

    time.sleep(1)
    st.rerun()

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

def handle_response(key_pressed: str):
    key = key_pressed.strip().upper()
    if key not in RESPONSE_KEYS:
        return
    t0 = st.session_state.t_start or time.time()
    rt = time.time() - t0
    t = st.session_state.trials[st.session_state.i]
    acc = int(key == str(t["QCorrectAnswer"]).strip().upper())
    log_debug(f"Response: key={key} correct={acc} RT={rt:.3f}s")
    finish_trial(resp_key=key, rt_sec=rt, correct=acc)

def screen_end():
    st.title("סיום הניסוי")
    st.success("תודה על השתתפותך!")

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

    if st.button("התחלה מחדש"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_state(); st.rerun()

# ========= Router =========
if st.session_state.page == "welcome":
    screen_welcome()
elif st.session_state.page == "trial":
    screen_trial()
else:
    screen_end()
