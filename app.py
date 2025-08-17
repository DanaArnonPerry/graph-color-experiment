# app.py
"""
Visual Memory Experiment for Charts
Improved version with better code structure, error handling, and responsive design
"""

import os
import time
import random
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from io import BytesIO

import requests
import pandas as pd
import streamlit as st
from PIL import Image

# Google Sheets
import gspread
from google.oauth2 import service_account


# ========= Configuration =========
@dataclass
class ExperimentConfig:
    """Central configuration for the experiment"""
    MAX_TRIALS: int = 40
    TRIAL_TIMEOUT_SEC: int = 30
    DATA_PATH: str = "data/colors_in_charts.csv"
    
    # Google Sheets configuration
    GSHEET_ID: str = "1ePIoLpP0Y0d_SedzVcJT7ttlV_1voLTssTvWAqpMkqQ"
    GSHEET_WORKSHEET_NAME: str = "Results"
    
    # Brand assets
    LOGO_CANDIDATES: List[str] = None
    USER_PHOTO_CANDIDATES: List[str] = None
    WEBSITE_URL: str = ""  # קישור אתר במסך הסיום (השאירי ריק אם לא צריך)
    
    def __post_init__(self):
        if self.LOGO_CANDIDATES is None:
            self.LOGO_CANDIDATES = [
                "images/Logo.png", "images/logo.png",
                "images/Logo29.10.24_B.png", "Logo.png", "Logo"
            ]
        if self.USER_PHOTO_CANDIDATES is None:
            self.USER_PHOTO_CANDIDATES = [
                "images/DanaSherlok.png", "images/DanaSherlok.jpg",
                "DanaSherlok.png", "DanaSherlok.jpg", "DanaSherlok"
            ]

config = ExperimentConfig()


# ========= Utility Functions =========
def find_first_existing_file(paths: List[str]) -> Optional[str]:
    """Find the first existing file from a list of paths"""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


# We can remove the ResponseTimer class since we're using direct timing
# class ResponseTimer:
#     """Handles timing for trial responses"""
#     
#     def __init__(self, timeout_seconds: int):
#         self.timeout = timeout_seconds
#         self.start_time = time.time()
#     
#     @property
#     def elapsed_seconds(self) -> float:
#         """Get elapsed time in seconds"""
#         return time.time() - self.start_time
#     
#     @property
#     def remaining_seconds(self) -> int:
#         """Get remaining time in seconds (integer)"""
#         return max(0, self.timeout - int(self.elapsed_seconds))
#     
#     @property
#     def is_expired(self) -> bool:
#         """Check if timer has expired"""
#         return self.elapsed_seconds >= self.timeout


class SessionManager:
    """Centralized session state management"""
    
    @staticmethod
    def initialize() -> None:
        """Initialize session state with default values"""
        defaults = {
            "page": "welcome",     # welcome -> practice -> trial -> end
            "experiment_data": None,
            "practice_trial": None,
            "trials": None,
            "current_trial_index": 0,
            "trial_start_time": None,
            "results": [],
            "image_cache": {},
            "participant_id": "",
            "run_start_iso": "",
            "is_admin": False
        }
        
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)


# ========= Page Setup =========
def setup_page_config() -> None:
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="ניסוי בזיכרון חזותי של גרפים",
        page_icon="📊",
        layout="centered"
    )
    
    # RTL styling for Hebrew
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { 
            direction: rtl; 
            text-align: right; 
            font-family: "Rubik","Segoe UI","Arial",sans-serif; 
        }
        blockquote, pre, code { 
            direction: ltr; 
            text-align: left; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ========= Admin Functions =========
class AdminManager:
    """Handle admin authentication and features"""
    
    @staticmethod
    def show_admin_sidebar() -> None:
        """Show admin authentication in sidebar"""
        logo_path = find_first_existing_file(config.LOGO_CANDIDATES)
        
        with st.sidebar:
            if logo_path:
                st.image(logo_path, use_container_width=True)
            
            st.markdown("**🔐 אזור מנהל**")
            
            if not st.session_state.is_admin:
                pin = st.text_input("הכנסי PIN:", type="password", key="admin_pin_input")
                if st.button("כניסה", key="admin_login_button"):
                    if AdminManager._validate_pin(pin):
                        st.session_state.is_admin = True
                        st.success("מנהל מחובר ✅")
                    else:
                        st.error("PIN שגוי")
            else:
                st.success("מנהל מחובר ✅")
    
    @staticmethod
    def is_admin() -> bool:
        """Check if current user is admin (without showing UI)"""
        return st.session_state.get("is_admin", False)
    
    @staticmethod
    def _validate_pin(entered_pin: str) -> bool:
        """Validate admin PIN against secrets"""
        try:
            admin_pin = st.secrets["admin"].get("pin")
            if not admin_pin:
                st.error("לא מוגדר PIN (admin.pin) ב-Secrets.")
                return False
            return str(entered_pin).strip() == str(admin_pin).strip()
        except Exception:
            st.error("שגיאה בבדיקת PIN.")
            return False


# ========= Data Loading =========
@st.cache_data
def load_experiment_data() -> pd.DataFrame:
    """Load and validate experiment data from CSV"""
    try:
        # Try UTF-8 first, fallback to UTF-8-BOM
        try:
            data = pd.read_csv(config.DATA_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            data = pd.read_csv(config.DATA_PATH, encoding="utf-8-sig")
        
        # Clean and validate data
        data = data.dropna(how="all").fillna("")
        data = data.astype({col: str for col in data.columns})
        
        # Validate required columns
        required_columns = ["ID", "QuestionText", "ImageFileName", "QCorrectAnswer", "V"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return data
        
    except FileNotFoundError:
        st.error(f"קובץ הנתונים לא נמצא: {config.DATA_PATH}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("קובץ הנתונים ריק")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"שגיאה בטעינת הנתונים: {e}")
        return pd.DataFrame()


# ========= Image Handling =========
@st.cache_data
def load_and_cache_image(image_path: str) -> Optional[Image.Image]:
    """Load and cache images with proper error handling"""
    if not image_path:
        return None
    
    try:
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGBA")
        else:
            if not os.path.exists(image_path):
                st.warning(f"קובץ תמונה לא נמצא: {image_path}")
                return None
            return Image.open(image_path).convert("RGBA")
    except (requests.RequestException, IOError, ValueError) as e:
        st.warning(f"שגיאה בטעינת תמונה {image_path}: {e}")
        return None


# ========= Google Sheets Integration =========
class GoogleSheetsManager:
    """Handle Google Sheets operations"""
    
    @staticmethod
    def _get_service_account_info() -> Dict[str, str]:
        """Get service account info from Streamlit secrets"""
        try:
            # Try structured service_account section first
            sa_info = dict(st.secrets["service_account"])
            if sa_info and len(sa_info) > 3:  # Basic validation
                return sa_info
        except (KeyError, AttributeError):
            pass

        # Fallback to flat structure
        keys = [
            "type", "project_id", "private_key_id", "private_key",
            "client_email", "client_id", "auth_uri", "token_uri",
            "auth_provider_x509_cert_url", "client_x509_cert_url",
            "universe_domain",
        ]
        
        sa_info = {}
        missing_keys = []
        
        for key in keys:
            try:
                value = st.secrets[key]
                if value:  # Only add non-empty values
                    sa_info[key] = value
                else:
                    missing_keys.append(key)
            except (KeyError, AttributeError):
                missing_keys.append(key)
        
        # Check if we have minimum required keys
        required_keys = ["type", "project_id", "private_key", "client_email"]
        if not all(key in sa_info for key in required_keys):
            missing_required = [key for key in required_keys if key not in sa_info]
            raise RuntimeError(f"חסרים מפתחות נדרשים ב-Service Account: {missing_required}")
        
        return sa_info

    @staticmethod
    @st.cache_resource
    def get_gspread_client():
        """Get authenticated gspread client"""
        try:
            sa_info = GoogleSheetsManager._get_service_account_info()
            credentials = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ],
            )
            client = gspread.authorize(credentials)
            # Test the connection
            client.list_spreadsheet_files()  # This will fail if auth is bad
            return client
        except Exception as e:
            raise RuntimeError(f"שגיאה ביצירת חיבור ל-Google Sheets: {str(e)}")

    @staticmethod
    def ensure_worksheet_headers(worksheet, expected_headers: List[str]) -> None:
        """Ensure worksheet has correct headers"""
        current_values = worksheet.get_all_values()
        headers = list(expected_headers)
        
        if not current_values:
            worksheet.append_row(headers)
            return
        
        first_row = current_values[0]
        if first_row != headers:
            worksheet.update("1:1", [headers])

    @staticmethod
    def get_next_participant_sequence(sheet_id: str) -> int:
        """Get next participant sequence number from Meta worksheet"""
        try:
            gc = GoogleSheetsManager.get_gspread_client()
            spreadsheet = gc.open_by_key(sheet_id)
            
            try:
                meta_worksheet = spreadsheet.worksheet("Meta")
            except gspread.WorksheetNotFound:
                meta_worksheet = spreadsheet.add_worksheet(title="Meta", rows="2", cols="2")
                meta_worksheet.update("A1", "counter")
                meta_worksheet.update("A2", "1")
                return 1

            try:
                current_count = int(meta_worksheet.acell("A2").value or "0")
            except (ValueError, TypeError):
                current_count = 0
            
            next_count = current_count + 1
            meta_worksheet.update("A2", str(next_count))
            return next_count
            
        except Exception as e:
            st.warning(f"שגיאה בקבלת מספר נבדק: {e}")
            return int(time.time())  # Fallback to timestamp

    @staticmethod
    def save_results_to_sheet(results_df: pd.DataFrame, sheet_id: str, worksheet_name: str) -> None:
        """Save results DataFrame to Google Sheet"""
        if results_df.empty:
            raise ValueError("אין תוצאות לשמירה")
            
        try:
            gc = GoogleSheetsManager.get_gspread_client()
            
            # Try to open the spreadsheet first
            try:
                spreadsheet = gc.open_by_key(sheet_id)
            except gspread.SpreadsheetNotFound:
                raise RuntimeError(f"גיליון לא נמצא עם ID: {sheet_id}")
            except Exception as e:
                raise RuntimeError(f"לא ניתן לגשת לגיליון: {str(e)}")
            
            # Get or create worksheet
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except gspread.WorksheetNotFound:
                try:
                    worksheet = spreadsheet.add_worksheet(
                        title=worksheet_name,
                        rows=str(max(len(results_df) + 100, 1000)),
                        cols=str(len(results_df.columns) + 5),
                    )
                except Exception as e:
                    raise RuntimeError(f"לא ניתן ליצור worksheet: {str(e)}")

            # Ensure headers
            GoogleSheetsManager.ensure_worksheet_headers(worksheet, results_df.columns)

            # Append data
            try:
                data_to_append = results_df.astype(str).values.tolist()
                worksheet.append_rows(data_to_append, value_input_option="RAW")
            except Exception as e:
                raise RuntimeError(f"שגיאה בהוספת נתונים: {str(e)}")
                
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"שגיאה בשמירה ל-Google Sheets: {str(e)}")


# ========= Trial Management =========
def build_alternating_trials(pool_df: pd.DataFrame, n_trials: int) -> List[Dict[str, Any]]:
    """
    Build alternating trial sequence to avoid consecutive identical V values
    
    Args:
        pool_df: DataFrame containing trial data with 'V' column for grouping
        n_trials: Number of trials to generate
        
    Returns:
        List of trial dictionaries in alternating order
    """
    # Group by V and shuffle each group
    groups = {
        v: sub_df.sample(frac=1, random_state=None).to_dict(orient="records")
        for v, sub_df in pool_df.groupby("V")
    }
    
    available_values = list(groups.keys())
    random.shuffle(available_values)
    
    trials = []
    last_v = None
    
    for _ in range(n_trials):
        # Find groups that still have trials
        available_groups = [v for v in available_values if groups[v]]
        if not available_groups:
            break
        
        # Prefer different V than last trial, but use any available if needed
        different_v_groups = [v for v in available_groups if v != last_v]
        candidates = different_v_groups if different_v_groups else available_groups
        
        # Select random group and pop trial
        selected_v = random.choice(candidates)
        trial = groups[selected_v].pop(0)
        trials.append(trial)
        last_v = selected_v
    
    return trials


# ========= UI Components =========
def render_graph_display(title_html: str, question_text: str, image_filename: str) -> None:
    """Render graph with responsive sizing"""
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(f"### {question_text}")

    image = load_and_cache_image(image_filename)
    if image is None:
        st.error("לא ניתן להציג את הגרף")
        return

    # Use responsive container width for better display on all devices
    st.image(image, use_container_width=True)


def render_response_interface(start_time: float, timeout_sec: int, on_timeout_callback, on_response_callback) -> None:
    """Render response buttons and timer with proper spacing"""
    
    # Calculate time remaining
    elapsed = time.time() - start_time
    remaining = max(0, timeout_sec - int(elapsed))
    
    # Check for timeout
    if elapsed >= timeout_sec:
        on_timeout_callback()
        st.stop()

    # Response buttons with proper spacing and unique keys
    button_cols = st.columns([0.1, 1, 1, 1, 1, 1, 0.1])
    response_options = ["E", "D", "C", "B", "A"]
    
    # Generate unique key based on session state and timestamp
    page_key = st.session_state.get("page", "unknown")
    trial_key = st.session_state.get("current_trial_index", 0)
    base_key = f"{page_key}_{trial_key}_{int(start_time)}"
    
    for idx, option in enumerate(response_options, start=1):
        if button_cols[idx].button(option, use_container_width=True, key=f"resp_{base_key}_{option}"):
            on_response_callback(option)
            st.stop()

    # Timer display
    st.markdown(
        f"<div style='text-align:center; margin-top:12px;'>⏳ זמן שנותר: "
        f"<b>{remaining}</b> שניות</div>",
        unsafe_allow_html=True,
    )

    # Auto-refresh every second
    time.sleep(1)
    st.rerun()


# ========= Screen Functions =========
def show_welcome_screen() -> None:
    """Display welcome screen with instructions"""
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

    # Generate participant ID automatically (hidden from user)
    if not st.session_state.participant_id:
        try:
            sequence = GoogleSheetsManager.get_next_participant_sequence(config.GSHEET_ID)
            st.session_state.participant_id = f"S{sequence:05d}"
        except Exception:
            # Fallback to timestamp-based ID if Google Sheets unavailable
            st.session_state.participant_id = f"S{int(time.time())}"

    # Load and validate experiment data
    experiment_data = load_experiment_data()
    if experiment_data.empty:
        st.stop()

    if st.button("המשך לתרגול", key="start_practice_button"):
        # Initialize experiment session
        st.session_state.run_start_iso = pd.Timestamp.now().isoformat(timespec="seconds")
        st.session_state.practice_trial = experiment_data.iloc[0].to_dict()
        
        # Prepare trials (skip first row used for practice)
        pool_data = experiment_data.iloc[1: 1 + config.MAX_TRIALS].copy()
        st.session_state.trials = build_alternating_trials(pool_data, config.MAX_TRIALS)
        
        # Reset trial state
        st.session_state.experiment_data = experiment_data
        st.session_state.current_trial_index = 0
        st.session_state.trial_start_time = None
        st.session_state.results = []
        st.session_state.page = "practice"
        st.rerun()


def show_practice_screen() -> None:
    """Display practice trial screen"""
    # Initialize timer if needed
    if st.session_state.trial_start_time is None:
        st.session_state.trial_start_time = time.time()
    
    practice_trial = st.session_state.practice_trial
    title_html = "<div style='font-size:20px; font-weight:700; text-align:right; margin-bottom:0.5rem;'>תרגול</div>"
    
    render_graph_display(title_html, practice_trial["QuestionText"], practice_trial["ImageFileName"])

    def on_practice_timeout():
        st.session_state.trial_start_time = None
        st.session_state.page = "trial"
        st.rerun()

    def on_practice_response(_):
        st.session_state.trial_start_time = None
        st.session_state.page = "trial"
        st.rerun()

    render_response_interface(
        st.session_state.trial_start_time, 
        config.TRIAL_TIMEOUT_SEC, 
        on_practice_timeout, 
        on_practice_response
    )


def show_trial_screen() -> None:
    """Display main trial screen"""
    # Initialize timer if needed
    if st.session_state.trial_start_time is None:
        st.session_state.trial_start_time = time.time()
    
    trial_index = st.session_state.current_trial_index
    current_trial = st.session_state.trials[trial_index]
    
    title_html = f"<div style='font-size:20px; font-weight:700; text-align:right; margin-bottom:0.5rem;'>גרף מספר {trial_index + 1}</div>"
    
    render_graph_display(title_html, current_trial["QuestionText"], current_trial["ImageFileName"])

    def record_trial_result(response_key: Optional[str], reaction_time: float, is_correct: bool):
        """Record trial result and advance to next trial or end"""
        result = {
            "ParticipantID": st.session_state.participant_id,
            "RunStartISO": st.session_state.run_start_iso,
            "TrialIndex": trial_index + 1,
            "ID": current_trial["ID"],
            "ResponseKey": response_key or "",
            "QCorrectAnswer": current_trial["QCorrectAnswer"],
            "Accuracy": int(is_correct),
            "RT_sec": round(reaction_time, 3),
        }
        
        st.session_state.results.append(result)
        st.session_state.trial_start_time = None
        
        # Advance to next trial or end
        if trial_index + 1 < len(st.session_state.trials):
            st.session_state.current_trial_index += 1
            st.rerun()
        else:
            st.session_state.page = "end"
            st.rerun()

    def on_trial_timeout():
        record_trial_result(
            response_key=None,
            reaction_time=float(config.TRIAL_TIMEOUT_SEC),
            is_correct=False
        )

    def on_trial_response(response_key: str):
        reaction_time = time.time() - st.session_state.trial_start_time
        correct_answer = str(current_trial["QCorrectAnswer"]).strip().upper()
        is_correct = response_key.strip().upper() == correct_answer
        
        record_trial_result(response_key.strip().upper(), reaction_time, is_correct)

    render_response_interface(
        st.session_state.trial_start_time, 
        config.TRIAL_TIMEOUT_SEC, 
        on_trial_timeout, 
        on_trial_response
    )


def show_end_screen() -> None:
    """Display end screen with results and thanks"""
    st.title("סיום הניסוי")
    st.success("תודה על השתתפותך!")

    results_df = pd.DataFrame(st.session_state.results)

    # Save to Google Sheets (attempt only, don't show errors to regular users)
    try:
        GoogleSheetsManager.save_results_to_sheet(
            results_df, 
            config.GSHEET_ID, 
            config.GSHEET_WORKSHEET_NAME
        )
        st.caption("התוצאות נשמרו בהצלחה ✅")
    except Exception as e:
        if AdminManager.is_admin():
            st.error(f"נכשלה שמירה ל-Google Sheets: {e}")
        else:
            # Don't show error to regular users - save will be handled in background
            pass

    # Admin-only features
    if AdminManager.is_admin():
        st.download_button(
            "הורדת תוצאות (CSV)",
            data=results_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"{st.session_state.participant_id}_{st.session_state.run_start_iso.replace(':', '-')}.csv",
            mime="text/csv",
        )
        st.link_button(
            "פתח/י את Google Sheet",
            f"https://docs.google.com/spreadsheets/d/{config.GSHEET_ID}/edit",
            type="primary",
        )

    # Optional branding section
    branding_cols = st.columns([1, 1, 1])
    with branding_cols[1]:
        user_photo_path = find_first_existing_file(config.USER_PHOTO_CANDIDATES)
        if user_photo_path:
            st.image(user_photo_path, width=120)
        if config.WEBSITE_URL:
            st.markdown(
                f"<div style='text-align:center; margin-top:8px;'>"
                f"<a href='{config.WEBSITE_URL}' target='_blank' style='text-decoration:underline;'>"
                f"לאתר שלי</a></div>",
                unsafe_allow_html=True,
            )


# ========= Main Application Router =========
def main() -> None:
    """Main application entry point"""
    setup_page_config()
    SessionManager.initialize()
    
    # Show admin sidebar on all pages
    AdminManager.show_admin_sidebar()
    
    # Route to appropriate screen
    current_page = st.session_state.page
    
    if current_page == "welcome":
        show_welcome_screen()
    elif current_page == "practice":
        show_practice_screen()
    elif current_page == "trial":
        show_trial_screen()
    elif current_page == "end":
        show_end_screen()
    else:
        # Fallback to welcome if invalid page
        st.session_state.page = "welcome"
        st.rerun()


if __name__ == "__main__":
    main()
