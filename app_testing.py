import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from auth import AuthManager
import os
import copy
import time
import math
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo  # available in Python 3.9+
import json
import re
import numpy as np
import matplotlib.pyplot as plt



ss = st.session_state
# -- PAGE CONFIG --
st.set_page_config(page_title="Experiment Viewer", layout="wide")


# -- CONFIG --
try:
    # ✅ Try to load from Streamlit secrets (Cloud or local if file exists)
    SUPABASE_URL = st.secrets["supabase_url"]
    SUPABASE_KEY = st.secrets["anon_key"]
    MY_EMAIL = st.secrets.get("my_email", "")
    MY_PWD = st.secrets.get("my_pwd", "")
except st.errors.StreamlitSecretNotFoundError:
    # 🔁 Fallback to local .env file
    load_dotenv()
    SUPABASE_URL = os.getenv("supabase_url")
    SUPABASE_KEY = os.getenv("anon_key")
    MY_EMAIL = os.getenv("my_email", "")
    MY_PWD = os.getenv("my_pwd", "")
if "supabase_client" not in ss:
    ss.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = ss.supabase_client
auth = AuthManager(supabase)

# -- GLOBAL FILTER FIELDS --
FILTER_FIELDS = {
    "wafer_id": "Wafer ID",
    "chip_id": "Chip ID",
    "device_name": "Device Name"
}


def get_unique_values(field_name):
    try:
        response = supabase.table("unique_ids").select("id_value").eq("id_type", field_name).execute()
        return sorted([item["id_value"] for item in response.data])
    except Exception as e:
        st.warning(f"Failed to load {field_name}s: {e}")
        return []

# -- SESSION STATE --
def ensure_session_state():
    defaults = {
        "user": None,
        "show_change_pwd": False,
        "update_needed": True,
        "rows_per_page": 50,
        "default_rows_per_page": 50,
        "current_page": 1,
        "total_pages": 1,
        "data_table": pd.DataFrame(),
        "user_timezone": ZoneInfo("America/New_York"),
        "bucket_name": "experiments",
        "start_datetime": None,
        "end_datetime": None,
        "page_input": 1,
        "plotting_data": dict(), # {data_path1: {"metadata": {}, "raw_data": {}, "plots": {}}, ...}
        "legend_format": "{wafer_id} | {chip_id} | {device_name} | ({x:.3f}, {y:.3f})",
        "action_name": "apply_fiter", # apply_filter, change_page, 
        "count": 0,
    }
    for key in defaults:
        if key not in ss:
            ss[key] = defaults[key]

    for key in FILTER_FIELDS:
        if f"selected_{key}s" not in ss:
            ss[f"selected_{key}s"] = []
            ss[f"select_all_{key}"] = True

# -- HELPERS --
def get_query(select_option="*", count_option=None):
    query = (
        supabase.table("experiments")
        .select(select_option, count=count_option)
        .order("datetime", desc=True)
    )
    for key in FILTER_FIELDS:
        values = ss.get(f"selected_{key}s", [])
        query = query.in_(key, values)
    start_datetime = ss.start_datetime
    end_datetime = ss.end_datetime
    assert start_datetime is None or start_datetime.tzinfo is not None
    assert end_datetime is None or end_datetime.tzinfo is not None

    if start_datetime is not None:
        query = query.gte("datetime", start_datetime.isoformat())
    if end_datetime is not None:
        query = query.lte("datetime", end_datetime.isoformat())
    return query

def get_experiments(offset, limit):
    try:
        query = get_query()
        query = query.range(offset, offset + limit - 1)
        response = query.execute()
        return pd.DataFrame(response.data)

    except Exception as e:
        st.error(f"Failed to fetch experiments: {e}")
        return pd.DataFrame()

def get_total_pages(limit):
    try:
        query = get_query("id", "exact")
        response = query.execute()
        total_count = response.count or 0
        total_pages = math.ceil(total_count / limit)
        return total_pages
    except Exception as e:
        st.error(f"Failed to get total page count: {e}")
        return 1

def load_plotting_data(plotted_paths):
    # store 
    plotted_paths = set(plotted_paths)
    for path in set(ss.plotting_data):
        if path not in plotted_paths:
            del ss.plotting_data[path]
    for path in plotted_paths:
        if path not in ss.plotting_data:
            response = supabase.storage.from_(ss.bucket_name).download(path)
            ss.plotting_data[path] = json.loads(response.decode("utf-8"))
            ss.plotting_data[path]["plots"] = {}
    #st.write(ss.plotting_data)    

def make_plots(override=False):
    for data_path, data in ss.plotting_data.items():
        if "fiber_scan" not in data["plots"] or override:
            fiber = data["raw_data"]["fiber_scan"]
            legend = ss.legend_format.format(**data["metadata"])
            x = np.array(fiber["x"])
            y = np.array(fiber["y"])
            power = np.array(fiber["output_power"])  # shape: (len(y), len(x))
            fig, ax = plt.subplots()
            im = ax.imshow(power, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", aspect="auto")
            ax.set_title(f"Fiber Scan: {legend}", pad=20)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            fig.colorbar(im, ax=ax, label="Output Power")
            data["plots"]["fiber_scan"] = fig
        if "power_spectrum" not in data["plots"] or override:
            fig, ax = plt.subplots()
            spectrum = data["raw_data"]["power_spectrum"]
            legend = ss.legend_format.format(**data["metadata"])
            wavelength = np.array(spectrum["wavelength"])
            output_power = np.array(spectrum["output_power"])
            ax.plot(wavelength, output_power, label=legend)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Output Power (Watt)")
            ax.set_title(f"Spectrum: {legend}")
            data["plots"]["power_spectrum"] = fig
    fig, ax = plt.subplots()
    for data_path, data in ss.plotting_data.items():
        spectrum = data["raw_data"]["power_spectrum"]
        legend = ss.legend_format.format(**data["metadata"])
        wavelength = np.array(spectrum["wavelength"])
        output_power = np.array(spectrum["output_power"])
        ax.plot(wavelength, output_power, label=legend)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Output Power (Watt)")
    ax.set_title("Power Spectrum")
    ax.legend()
    #ax.set_aspect(1.2)
    ss["power_spectrum_plot"] = fig
        

# -- AUTH UI --
def login_ui():
    st.header("Sign In")
    email = st.text_input("Email", value=MY_EMAIL)
    password = st.text_input("Password", type="password", value=MY_PWD)
    if st.button("Sign In"):
        user = auth.sign_in(email, password)
        if isinstance(user, str):
            st.error(user)
        else:
            ss.user = user
            ss.show_change_pwd = False
            st.success("Signed in!")
            st.rerun()

def signup_ui():
    st.header("Sign Up")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pwd")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
    strength_msg = auth.check_password_strength(password)
    passwords_match = password == confirm_password
    if strength_msg:
        st.warning(strength_msg)
    if not passwords_match:
        st.warning("Passwords don't match")
    if st.button("Create Account", disabled=not (passwords_match and strength_msg is None)):
        user = auth.sign_up(email, password)
        if isinstance(user, str):
            st.error(user)
        else:
            st.success("If this is a new email, check your email for confirmation.")

# -- DASHBOARD UI --
def dashboard_ui():
    st.header("Welcome!")
    st.write(f"Logged in as: {ss.user.email}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign Out"):
            msg = auth.sign_out()
            ss.user = None
            st.success(msg)
            st.rerun()
    with col2:
        if st.button("Change Password"):
            ss.show_change_pwd = not ss.show_change_pwd

    if ss.show_change_pwd:
        st.subheader("Change Password")
        new_pwd = st.text_input("New Password", type="password")
        confirm_pwd = st.text_input("Confirm New Password", type="password")
        strength_msg = auth.check_password_strength(new_pwd)
        if strength_msg:
            st.warning(strength_msg)
        if st.button("Confirm", key="confirm_change_pwd"):
            if new_pwd != confirm_pwd:
                st.error("Passwords do not match.")
            else:
                msg = auth.change_password(new_pwd)
                st.success(msg if isinstance(msg, str) else "Password changed.")

    st.divider()
    # experiment_viewer_ui()
    filters_ui()
    data_table_ui()
    st.divider()
    plotter_get_legend_ui()

def change_page(current_page):
    st.write(f"change_page:{current_page}, count: {ss.count}")
    ss.update_needed = True
    ss.current_page = current_page
    
    ss.current_page = min(max(1, current_page), ss.total_pages)

def combine_datetime(date_val, time_val):
    if date_val is None:
        return None
    if time_val is None:
        time_val = dt_time()  # default to midnight
    dt_local = datetime.combine(date_val, time_val)
    dt = dt_local.replace(tzinfo=ss.user_timezone)
    return dt

def filters_ui():
    with st.expander("Filters", expanded=True):
        for key, label in FILTER_FIELDS.items():
            all_keys = get_unique_values(key)
            selected_keys = f"selected_{key}s"
            select_all_keys = f"select_all_{key}"
            col1, col2 = st.columns([85,15])
            with col1:
                container = st.container()
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add vertical spacing
                ss[select_all_keys] = st.checkbox(f"Select all", key=f"c_select_all_{key}", value=True)
            
            if ss[select_all_keys]:
                ss[selected_keys] = container.multiselect(label, all_keys, all_keys)
            else:
                ss[selected_keys] = container.multiselect(label, all_keys)
        #st.markdown("#### Time Range Filter")
        col1, col2, col3 = st.columns(3, border=True)
        with col1:
            start_date = st.date_input("Start Date", value=None)
            start_time = st.time_input("Start Time", value=None)   
        with col2:
            end_date = st.date_input("End Date", value=None)
            end_time = st.time_input("End Time", value=None)
        with col3:
            ss.rows_per_page = st.number_input("Rows per page", min_value=10, max_value=200, value=ss.default_rows_per_page, step=10)
        ss.start_datetime = combine_datetime(start_date, start_time)
        ss.end_datetime = combine_datetime(end_date, end_time)
        
        st.button("Apply Filters", on_click=lambda : setattr(ss, "action_name", "apply_filter"))

def data_table_ui():
    # Get table from Supabase and update parameters
    if ss.action_name in ["apply_filter", "change_page"]: #ss.update_needed
        ss["total_pages"] = get_total_pages(limit=ss.rows_per_page)
        if ss.action_name == "apply_filter":
            ss.current_page = 1
        elif ss.action_name == "change_page":
            ss.current_page = ss.page_input
        offset = (ss.current_page - 1) * ss.rows_per_page
        
        #st.write(f"current_page:{ss.current_page}")
        df = get_experiments(offset=offset, limit=ss.rows_per_page)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(ss.user_timezone).apply(lambda x: x.isoformat())
            df["plotted"] = False  # or whatever name you want
            cols = list(df.columns)
            cols.remove("plotted")
            cols.insert(cols.index("datetime") + 1, "plotted")
            df = df[cols]
        ss["data_table"] = df
    else:
        df = ss["data_table"]
    
    # Make UI
    st.info("Filters won't take effect unless you click on 'Apply filters'")
    if df is not None and not df.empty:
        cols = list(df.columns)
        cols.remove("plotted")
        editable_df = st.data_editor(
            df.sort_values("datetime", ascending=False),
            column_config={
                "plotted": st.column_config.CheckboxColumn("Plotted"),
            },
            disabled=cols,
        )
        plotted_paths = editable_df[editable_df["plotted"]]["data_path"].tolist()
        load_plotting_data(plotted_paths)
        bucket_name = "experiments"
        if len(plotted_paths) > 0:
            
            data_path = plotted_paths[0]
            response = supabase.storage.from_(bucket_name).download(data_path)
            content = json.loads(response.decode("utf-8"))
            #st.write("Parsed JSON content:", content)
        col1, col2, col3 = st.columns(3)
        with col1:
            total_pages = ss.total_pages
            container = st.container()
            if ss.action_name == "apply_filter":
                container.number_input(f"Go to page (total {total_pages})", min_value=1, max_value=total_pages, step=1, format="%d", key="page_input", on_change=lambda : setattr(ss, "action_name", "change_page"), value=1)
                ss.action_name = "change_page"
            else:
                container.number_input(f"Go to page (total {total_pages})", min_value=1, max_value=total_pages, step=1, format="%d", key="page_input", on_change=lambda : setattr(ss, "action_name", "change_page"), value=ss.current_page)
            # change_page(int(page))
    ss.update_needed = False

def plotter_get_legend_ui():
    if ss["data_table"] is None or ss["data_table"].empty:
        return
    if len(ss.plotting_data) == 0:
        st.info("Check the ‘Plotted’ column above to plot data.")
        return
    try:
        keys = list(ss.plotting_data.keys())
        data0 = ss.plotting_data[keys[0]]
        metadata_keys = set(data0['metadata'].keys())
    except Exception as e:
        st.error(f"Error accessing metadata: {e}")
        return  # prevent further errors if data0 is not defined
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_input(
            f"Enter your legend format (available fields: {', '.join(sorted(metadata_keys))})",
            ss.legend_format
        )
    try:
        legend = user_input.format(**data0["metadata"])
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add vertical spacing
            st.success(f"Legend preview: {legend}")
            ss.legend_format = user_input
    except KeyError as e:
        st.error(f"Unknown field name: {e}")
    except (AttributeError, ValueError) as e:
        st.error(f"Invalid formatting: {e}")
    
    
    make_plots()
    #st.write(ss.legend_format)
    st.button("Apply format", on_click=lambda : make_plots(override=True))

    if "power_spectrum_plot" in ss:
        col0, col1, col2 = st.columns([3,4,3])
        with col1:
            st.pyplot(ss.power_spectrum_plot)
    for data_path, data in ss.plotting_data.items():
        #st.write(f"data legend: {data["legend"]}")
        col0, col1, col2, col4 = st.columns([15,35,35,15])
        with col1:
            st.pyplot(data["plots"]["fiber_scan"])
        with col2:
            st.pyplot(data["plots"]["power_spectrum"])


# -- MAIN --
st.title("Experiment Data Viewer")

ensure_session_state()



if ss.user:
    dashboard_ui()
else:
    page = st.radio("Choose page", ["Sign In", "Sign Up"], horizontal=True, label_visibility="collapsed")
    if page == "Sign In":
        login_ui()
    else:
        signup_ui()



ss.count += 1
ss.action_name = ""

    
