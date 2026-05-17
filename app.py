

import os
import tempfile

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
from ultralytics import YOLO

from database import (
    delete_user, get_all_logs, get_all_users,
    get_user_history, init_db, log_detection,
    login_user, register_user,
)
from email_utils import send_congestion_alert, send_summary_email

init_db()

st.set_page_config(
    page_title="Traffic Congestion System",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded",
)

_DEFAULTS = {
    "logged_in":     False,
    "user_id":       None,
    "username":      None,
    "role":          None,
    "page":          "login",
    "notifications": [],
    "email_enabled": False,
    "email_cfg":     {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Global CSS (loaded from style.css) ───────────────────────
def _load_css():
    import os
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # fallback minimal dark theme if css file missing
        st.markdown("""<style>
        .stApp { background:#080b10 !important; color:#e8edf5 !important; }
        [data-testid="stSidebar"] { background:#0a0f1a !important; }
        </style>""", unsafe_allow_html=True)

_load_css()

# ── YOLO model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov5s.pt")

# ── City presets ──────────────────────────────────────────────
CITIES = {
    "Kolkata":   (22.5726,  88.3639),
    "Mumbai":    (19.0760,  72.8777),
    "Delhi":     (28.6139,  77.2090),
    "Bangalore": (12.9716,  77.5946),
    "Chennai":   (13.0827,  80.2707),
    "Hyderabad": (17.3850,  78.4867),
    "Pune":      (18.5204,  73.8567),
    "Ahmedabad": (23.0225,  72.5714),
    "Jaipur":    (26.9124,  75.7873),
    "Custom":    None,
}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def show_login():
    st.markdown("""
    <div style='text-align:center;padding:3rem 0 2rem'>
        <div style='display:inline-block;background:linear-gradient(135deg,#f59e0b22,#ef444411);
                    border:1px solid #f59e0b33;border-radius:16px;
                    padding:0.5rem 1.4rem;margin-bottom:1.2rem;
                    font-family:"Rajdhani",sans-serif;font-size:0.8rem;
                    letter-spacing:0.2em;color:#f59e0b;text-transform:uppercase'>
            ⬡ AI-Powered Detection System
        </div>
        <h1 style='font-family:"Rajdhani",sans-serif;font-size:3rem;font-weight:700;
                   letter-spacing:0.05em;text-transform:uppercase;
                   background:linear-gradient(135deg,#f59e0b,#fbbf24,#fde68a);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   background-clip:text;margin:0 0 0.8rem'>
            🚦 Traffic Congestion<br>Detection System
        </h1>
        <p style='color:#64748b;font-size:0.95rem;font-family:"Inter",sans-serif;
                  letter-spacing:0.05em;margin:0'>
            Upload traffic footage &nbsp;·&nbsp; YOLOv5 Detection &nbsp;·&nbsp;
            Live congestion map &nbsp;·&nbsp; Email alerts
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        with st.container(border=True):
            tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

            with tab_login:
                username = st.text_input("Username", key="l_user", placeholder="Enter username")
                password = st.text_input("Password", type="password", key="l_pass")
                if st.button("Login", use_container_width=True, type="primary"):
                    user = login_user(username, password)
                    if user:
                        st.session_state.update({
                            "logged_in": True, "user_id": user[0],
                            "username": user[1], "role": user[2], "page": "detection"
                        })
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                #st.caption("🔑 Default admin → `admin` / `admin123`")

            with tab_register:
                nu   = st.text_input("Username",         key="r_user")
                ne   = st.text_input("Email",            key="r_email")
                np1  = st.text_input("Password",         type="password", key="r_p1")
                np2  = st.text_input("Confirm Password", type="password", key="r_p2")
                if st.button("Create Account", use_container_width=True, type="primary"):
                    if not nu or not np1:
                        st.error("Username and password required.")
                    elif np1 != np2:
                        st.error("Passwords do not match.")
                    elif len(np1) < 6:
                        st.error("Password must be ≥ 6 characters.")
                    else:
                        ok, msg = register_user(nu, np1, ne)
                        st.success(msg) if ok else st.error(msg)



PAGE_MAP = {
    "🎥  Detection":       "detection",
    "🗺️  Congestion Map":  "congestion_map",
    "📋  My History":      "my_history",
    "📧  Email Settings":  "email_settings",
    "👨‍💼  Admin Dashboard": "admin_dashboard",
}

def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a2236,#141c2e);
                    border:1px solid #1e2d45;border-left:3px solid #f59e0b;
                    border-radius:10px;padding:14px 16px;margin-bottom:10px'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                        letter-spacing:0.15em;color:#64748b;text-transform:uppercase;
                        margin-bottom:4px'>Logged in as</div>
            <div style='font-family:"Rajdhani",sans-serif;font-size:1.25rem;
                        font-weight:700;color:#f59e0b;letter-spacing:0.04em'>
                👤 {st.session_state.username.upper()}
            </div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:0.7rem;
                        color:#64748b;margin-top:2px'>
                ROLE: <span style='color:#e8edf5'>{st.session_state.role.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.email_enabled:
            st.success("📧 Email alerts: ON",  icon="✅")
        else:
            st.warning("📧 Email alerts: OFF", icon="⚠️")

        notifs  = st.session_state.notifications
        n_count = len(notifs)
        if n_count:
            st.markdown(
                f"### 🔔 Alerts <span class='notif-count'>{n_count}</span>",
                unsafe_allow_html=True)
            for note in notifs[-4:]:
                if "HIGH"     in note: st.error(note,   icon="🚨")
                elif "MODERATE" in note: st.warning(note, icon="⚠️")
                else:                  st.info(note,    icon="ℹ️")
            if st.button("✕ Clear alerts", use_container_width=True):
                st.session_state.notifications = []
                st.rerun()
        else:
            st.markdown("### 🔔 No active alerts")

        st.divider()
        st.markdown("### 📌 Navigation")
        for label, key in PAGE_MAP.items():
            if key == "admin_dashboard" and st.session_state.role != "admin":
                continue
            prefix = "🟢 " if st.session_state.page == key else ""
            if st.button(f"{prefix}{label}", use_container_width=True, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "login"
            st.rerun()



def show_email_settings():
    st.title("📧 Email Alert Settings")

    with st.container(border=True):
        st.subheader("📬 Configure Email Notifications")
        st.markdown("""
        Receive automatic emails when:
        - 🚨 Congestion is detected **during** video processing (throttled)
        - 📊 A full **summary report** is sent after processing completes
        """)
        enabled = st.toggle("Enable email alerts",
                            value=st.session_state.email_enabled)
        st.session_state.email_enabled = enabled
        if not enabled:
            st.info("Toggle ON to configure.")
            return

    st.divider()

    with st.container(border=True):
        st.subheader("⚙️ SMTP Configuration")
        col1, col2 = st.columns(2)

        with col1:
            provider = st.selectbox("Email Provider",
                                    ["gmail", "outlook", "yahoo"],
                                    index=0)
            sender_email = st.text_input(
                "Your Email Address",
                placeholder="yourname@gmail.com",
                value=st.session_state.email_cfg.get("sender_email", ""))
            sender_password = st.text_input(
                "App Password",
                type="password",
                placeholder="xxxx xxxx xxxx xxxx",
                value=st.session_state.email_cfg.get("sender_password", ""),
                help="Use an App Password — NOT your regular password")

        with col2:
            recipient_email = st.text_input(
                "Send Alerts To",
                placeholder="recipient@example.com",
                value=st.session_state.email_cfg.get("recipient_email", ""))
            alert_on_high     = st.checkbox("Alert on HIGH congestion (>50%)",    value=True)
            alert_on_moderate = st.checkbox("Alert on MODERATE congestion (>20%)", value=True)
            send_summary_opt  = st.checkbox("Send summary report after detection", value=True)

        if provider == "gmail":
            with st.expander("ℹ️ How to get a Gmail App Password (required for Gmail)"):
                st.markdown("""
                Gmail blocks regular passwords for SMTP. Follow these steps:

                1. Go to **Google Account → Security**
                2. Enable **2-Step Verification**
                3. Search for **"App Passwords"** in your Google Account
                4. Select **Mail** + **Windows Computer** → click **Generate**
                5. Copy the 16-character code → paste it above

                🔗 **Direct link:** https://myaccount.google.com/apppasswords
                """)

        col_save, col_test = st.columns(2)
        with col_save:
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                if not all([sender_email, sender_password, recipient_email]):
                    st.error("All fields are required.")
                else:
                    st.session_state.email_cfg = {
                        "provider":          provider,
                        "sender_email":      sender_email,
                        "sender_password":   sender_password,
                        "recipient_email":   recipient_email,
                        "alert_on_high":     alert_on_high,
                        "alert_on_moderate": alert_on_moderate,
                        "send_summary":      send_summary_opt,
                    }
                    st.success("✅ Settings saved!")

        with col_test:
            if st.button("📤 Send Test Email", use_container_width=True):
                if not st.session_state.email_cfg:
                    st.error("Save your settings first.")
                else:
                    with st.spinner("Sending…"):
                        ok, msg = send_congestion_alert(
                            st.session_state.email_cfg,
                            "Test Location", 20, 0, 15, "HIGH")
                    if ok:
                        st.success(
                            f"✅ Test email sent to "
                            f"`{st.session_state.email_cfg['recipient_email']}`!")
                    else:
                        st.error(f"❌ {msg}")


def show_detection():
    st.title("🎥 Traffic Congestion Detection")
    left, right = st.columns([1, 3])

    with left:
        with st.container(border=True):
            st.subheader("⚙️ Settings")
            threshold  = st.slider("Congestion threshold", 3, 50, 15)
            conf_score = st.slider("YOLO confidence",      0.1, 0.9, 0.4)
            skip_n     = st.slider("Process every N-th frame", 1, 5, 1)

        with st.container(border=True):
            st.subheader("📍 Location")
            city = st.selectbox("Select city", list(CITIES.keys()))
            if city == "Custom":
                lat           = st.number_input("Latitude",  value=22.5726, format="%.4f")
                lng           = st.number_input("Longitude", value=88.3639, format="%.4f")
                location_name = st.text_input("Location label", value="My Location")
            else:
                lat, lng      = CITIES[city]
                location_name = city
                st.caption(f"📌 {lat:.4f}, {lng:.4f}")

        with st.container(border=True):
            st.subheader("📧 Email Alerts")
            if st.session_state.email_enabled and st.session_state.email_cfg:
                st.success(
                    f"→ `{st.session_state.email_cfg.get('recipient_email','?')}`",
                    icon="✅")
            else:
                st.warning("Off", icon="⚠️")
                if st.button("⚙️ Configure Email", use_container_width=True):
                    st.session_state.page = "email_settings"
                    st.rerun()

    with right:
        uploaded = st.file_uploader("📤 Upload traffic video",
                                    type=["mp4","avi","mov"])
        run_btn  = st.button("▶ Run Detection", type="primary",
                             use_container_width=True)
        if uploaded and run_btn:
            _run_detection(uploaded, threshold, conf_score,
                           skip_n, lat, lng, location_name)
        elif not uploaded:
            st.info("⬆️ Upload a video and click **Run Detection**.")


def _run_detection(uploaded, threshold, conf_score,
                   skip_n, lat, lng, location_name):
    model = load_model()
    tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        st.error("❌ Cannot open video.")
        os.unlink(tmp.name)
        return

    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    st.info(f"⏳ Processing **{uploaded.name}** — {total_frames} frames …")

    frame_ph  = st.empty()
    prog_bar  = st.progress(0)
    alert_ph  = st.empty()
    status_ph = st.empty()
    email_ph  = st.empty()

    data             = []
    frame_id         = 0
    last_alert_frame = -30
    email_sent_count = 0

    e_cfg     = st.session_state.email_cfg
    e_enabled = st.session_state.email_enabled and bool(e_cfg)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        prog_bar.progress(min(frame_id / total_frames, 1.0))
        if frame_id % skip_n != 0:
            continue

        results = model(frame, conf=conf_score)[0]
        vehicle_count = 0
        if results.boxes is not None:
            for box in results.boxes:
                cls   = int(box.cls[0])
                label = model.names[cls]
                if label in VEHICLE_CLASSES:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    col = (0, 200, 80) if vehicle_count <= threshold else (0, 50, 220)
                    cv2.rectangle(frame, (x1,y1),(x2,y2), col, 2)
                    cv2.putText(frame, label, (x1,y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

        congestion = 1 if vehicle_count > threshold else 0
        severity   = ("HIGH"     if vehicle_count > threshold * 2 else
                      "MODERATE" if vehicle_count > threshold else "LOW")

        if congestion:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0),(frame.shape[1],70),(0,0,180),-1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            cv2.putText(frame, f"CONGESTION [{severity}]", (20,48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,60,255), 3)

            if frame_id - last_alert_frame >= 30:
                last_alert_frame = frame_id
                notif = (f"{'🚨' if severity=='HIGH' else '⚠️'} {severity} — "
                         f"Congestion at **{location_name}** "
                         f"(frame {frame_id}, {vehicle_count} vehicles)")
                st.session_state.notifications.append(notif)
                alert_ph.error(
                    f"🚨 {severity} CONGESTION — {vehicle_count} vehicles "
                    f"at **{location_name}**!", icon="🚨")

                if e_enabled:
                    send_it = (
                        (severity == "HIGH"     and e_cfg.get("alert_on_high", True)) or
                        (severity == "MODERATE" and e_cfg.get("alert_on_moderate", True))
                    )
                    if send_it:
                        ok, msg = send_congestion_alert(
                            e_cfg, location_name, vehicle_count,
                            frame_id, threshold, severity)
                        email_sent_count += 1
                        if ok:
                            email_ph.success(f"📧 Alert email #{email_sent_count} sent!", icon="✅")
                        else:
                            email_ph.warning(f"📧 Email failed: {msg}")
        else:
            alert_ph.empty()

        cv2.putText(frame, f"Vehicles: {vehicle_count}  |  Frame: {frame_id}",
                    (10, frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_container_width=True)
        status_ph.markdown(
            f"`Frame {frame_id}/{total_frames}` — **{vehicle_count}** vehicles — "
            f"{'🔴 CONGESTED' if congestion else '🟢 CLEAR'}")
        data.append([frame_id, vehicle_count, congestion])

    cap.release()
    os.unlink(tmp.name)

    if not data:
        st.warning("No frames processed.")
        return

    df = pd.DataFrame(data, columns=["frame","vehicle_count","congestion"])
    n_frames         = len(df)
    max_veh          = int(df["vehicle_count"].max())
    congested_frames = int(df["congestion"].sum())
    rate             = round(congested_frames / n_frames * 100, 1)

    log_detection(st.session_state.user_id, uploaded.name,
                  n_frames, max_veh, congested_frames, rate,
                  lat, lng, location_name, threshold)

    prog_bar.empty(); status_ph.empty(); email_ph.empty()

    if rate > 50:
        st.error(f"🚨 **HIGH CONGESTION** — {rate}% at {location_name}!", icon="🚨")
        final_sev = "HIGH"
    elif rate > 20:
        st.warning(f"⚠️ **MODERATE CONGESTION** — {rate}% at {location_name}.", icon="⚠️")
        final_sev = "MODERATE"
    else:
        st.success(f"✅ **LOW CONGESTION** — {rate}% at {location_name}.", icon="✅")
        final_sev = "LOW"

    # ── Summary email ──────────────────────────────────────────
    if e_enabled and e_cfg.get("send_summary", True):
        with st.spinner("📧 Sending summary email…"):
            ok, msg = send_summary_email(
                e_cfg, location_name, uploaded.name,
                n_frames, max_veh, congested_frames, rate, threshold)
        if ok:
            st.success(
                f"📧 Summary report sent to `{e_cfg['recipient_email']}`!",
                icon="✅")
        else:
            st.warning(f"📧 Summary email failed: {msg}")

    if email_sent_count:
        st.info(f"📧 {email_sent_count} congestion alert email(s) sent during processing.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Frames",     n_frames)
    c2.metric("Max Vehicles",     max_veh)
    c3.metric("Congested Frames", congested_frames)
    c4.metric("Congestion Rate",  f"{rate}%",
              delta=f"{'▲ High' if rate>50 else ('▼ Low' if rate<20 else '— Moderate')}",
              delta_color="inverse" if rate > 50 else "normal")

    st.subheader("📈 Vehicle Count Over Time")
    fig = px.line(df, x="frame", y="vehicle_count",
                  color_discrete_sequence=["#58a6ff"],
                  labels={"vehicle_count":"Vehicles","frame":"Frame"})
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ff4444",
                  annotation_text=f"Threshold ({threshold})",
                  annotation_position="top right")
    fig.update_layout(plot_bgcolor="#080b10", paper_bgcolor="#0e1420",
                      font_color="#e6edf3")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "📥 Download Detection Report (CSV)",
        df.to_csv(index=False).encode(),
        file_name=f"detection_{location_name.replace(' ','_')}.csv",
        mime="text/csv")



def show_map():
    st.title("🗺️ Traffic Congestion Map")
    logs = (get_all_logs() if st.session_state.role == "admin"
            else get_user_history(st.session_state.user_id))
    if logs.empty:
        st.info("No detection data yet. Run a detection first!")
        return

    if "congestion_rate" in logs.columns:
        high = logs[logs["congestion_rate"] > 50]
        mod  = logs[(logs["congestion_rate"] > 20) & (logs["congestion_rate"] <= 50)]
        if not high.empty:
            st.error(f"🚨 **{len(high)} HIGH-CONGESTION zone(s)** on this map!", icon="🚨")
            for _, row in high.iterrows():
                st.error(
                    f"📍 **{row.get('location_name','?')}** — "
                    f"{row.get('congestion_rate',0):.1f}% "
                    f"({int(row.get('max_vehicles',0))} max vehicles)")
        if not mod.empty:
            st.warning(f"⚠️ **{len(mod)} MODERATE zone(s)** detected.")

    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[22.0,78.0], zoom_start=5,
                   tiles="CartoDB dark_matter")

    def _color(rate):
        return ("red","exclamation-sign") if rate > 50 else \
               ("orange","warning-sign") if rate > 20 else \
               ("green","ok-sign")

    for _, row in logs.iterrows():
        lat  = row.get("lat",  22.5726)
        lng  = row.get("lng",  88.3639)
        rate = row.get("congestion_rate", 0)
        loc  = row.get("location_name", "Unknown")
        ts   = str(row.get("timestamp",""))[:19]
        if pd.isna(lat) or pd.isna(lng): continue
        color, icon_name = _color(rate)
        popup_html = f"""
        <div style="font-family:Arial;min-width:210px;padding:4px">
            <h4 style="margin:0 0 8px;color:#333">📍 {loc}</h4>
            <table style="width:100%;border-collapse:collapse">
                <tr><td><b>Congestion Rate</b></td>
                    <td style="color:{'red' if rate>50 else ('orange' if rate>20 else 'green')}">
                        <b>{rate:.1f}%</b></td></tr>
                <tr><td><b>Max Vehicles</b></td><td>{int(row.get('max_vehicles',0))}</td></tr>
                <tr><td><b>Total Frames</b></td><td>{int(row.get('total_frames',0))}</td></tr>
                <tr><td><b>Threshold</b></td><td>{int(row.get('threshold',15))}</td></tr>
                <tr><td><b>Time</b></td><td>{ts}</td></tr>
            </table>
            {"<p style='color:red;font-weight:bold;margin:6px 0 0'>⚠️ HIGH CONGESTION ZONE</p>" if rate>50 else ""}
        </div>"""
        folium.Marker([lat,lng],
                      popup=folium.Popup(popup_html, max_width=270),
                      tooltip=f"{loc} — {rate:.1f}%",
                      icon=folium.Icon(color=color, icon=icon_name,
                                       prefix="glyphicon")).add_to(m)
        if rate > 50:
            folium.Circle([lat,lng], radius=3000, color="red",
                          fill=True, fill_opacity=0.12).add_to(m)

    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:rgba(10,10,20,0.85);color:#eee;
                padding:12px 16px;border-radius:10px;font-family:Arial;
                border:1px solid #333">
        <b>🗺️ Congestion Level</b><br>
        🔴 High &gt;50%<br>🟠 Moderate 20–50%<br>🟢 Low &lt;20%
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    st_folium(m, use_container_width=True, height=560)

    st.subheader("📊 Summary by Location")
    summary = (logs.groupby("location_name")
               .agg(Detections=("id","count"),
                    Avg_Congestion_Pct=("congestion_rate","mean"),
                    Max_Vehicles=("max_vehicles","max"),
                    Total_Frames=("total_frames","sum"))
               .round(1).reset_index())
    summary.columns = ["Location","Detections","Avg Congestion %",
                       "Max Vehicles","Total Frames"]
    st.dataframe(summary, use_container_width=True)


def show_history():
    st.title("📋 My Detection History")
    df = get_user_history(st.session_state.user_id)
    if df.empty:
        st.info("No history yet. Run your first detection!")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Detections", len(df))
    c2.metric("Avg Congestion %", f"{df['congestion_rate'].mean():.1f}%")
    c3.metric("Peak Vehicles",    int(df["max_vehicles"].max()))
    c4.metric("Worst Location",   df.loc[df["congestion_rate"].idxmax(),"location_name"])

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.bar(df, x="timestamp", y="congestion_rate",
                      color="congestion_rate",
                      color_continuous_scale=["#2ea043","#d29922","#f85149"],
                      title="Congestion Rate per Run",
                      labels={"congestion_rate":"Rate %","timestamp":"Date"})
        fig1.update_layout(plot_bgcolor="#080b10", paper_bgcolor="#0e1420",
                           font_color="#e6edf3", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.scatter(df, x="max_vehicles", y="congestion_rate",
                          color="location_name", size="total_frames",
                          title="Max Vehicles vs Congestion Rate")
        fig2.update_layout(plot_bgcolor="#080b10", paper_bgcolor="#0e1420",
                           font_color="#e6edf3")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🗂️ Detection Log")
    wanted = ["filename","location_name","total_frames","max_vehicles",
              "congested_frames","congestion_rate","threshold","timestamp"]
    avail  = [c for c in wanted if c in df.columns]
    disp   = df[avail].copy()
    disp["timestamp"] = pd.to_datetime(
        disp["timestamp"], errors="coerce").dt.strftime("%d %b %Y  %H:%M")
    disp.columns = [c.replace("_"," ").title() for c in avail]
    st.dataframe(disp, use_container_width=True)
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode(),
                       "my_history.csv", "text/csv")


def show_admin():
    if st.session_state.role != "admin":
        st.error("🔒 Admins only.")
        return
    st.title("👨‍💼 Admin Dashboard")
    all_logs  = get_all_logs()
    all_users = get_all_users()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Users",      len(all_users))
    c2.metric("Total Detections", len(all_logs))
    if not all_logs.empty and "congestion_rate" in all_logs.columns:
        c3.metric("Avg Congestion %",  f"{all_logs['congestion_rate'].mean():.1f}%")
        high = all_logs[all_logs["congestion_rate"] > 50]
        c4.metric("High Alert Events", len(high),
                  delta=str(len(high)) if len(high) else None,
                  delta_color="inverse")
    else:
        c3.metric("Avg Congestion %", "N/A")
        c4.metric("High Alert Events", 0)

    tab_a, tab_u, tab_l = st.tabs(["📊 Analytics","👥 Users","📋 All Logs"])

    with tab_a:
        if all_logs.empty:
            st.info("No data yet.")
        else:
            cl, cr = st.columns(2)
            with cl:
                fig = px.histogram(all_logs, x="congestion_rate", nbins=20,
                                   title="Congestion Rate Distribution",
                                   color_discrete_sequence=["#58a6ff"])
                fig.update_layout(plot_bgcolor="#080b10", paper_bgcolor="#0e1420",
                                  font_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)
            with cr:
                loc_avg = (all_logs.groupby("location_name")["congestion_rate"]
                           .mean().reset_index())
                fig2 = px.bar(loc_avg, x="location_name", y="congestion_rate",
                              color="congestion_rate",
                              color_continuous_scale=["#2ea043","#d29922","#f85149"],
                              title="Avg Congestion by Location")
                fig2.update_layout(plot_bgcolor="#080b10", paper_bgcolor="#0e1420",
                                   font_color="#e6edf3", showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            if "username" in all_logs.columns:
                fig3 = px.line(all_logs.sort_values("timestamp"),
                               x="timestamp", y="congestion_rate",
                               color="username",
                               title="Congestion Rate Over Time (All Users)")
                fig3.update_layout(plot_bgcolor="#080b10", paper_bgcolor="#0e1420",
                                   font_color="#e6edf3")
                st.plotly_chart(fig3, use_container_width=True)

    with tab_u:
        st.subheader("👥 Registered Users")
        if not all_users.empty:
            st.dataframe(all_users, use_container_width=True)
            st.subheader("🗑️ Delete User")
            non_admins = all_users[all_users["role"] != "admin"]
            if non_admins.empty:
                st.info("No non-admin users.")
            else:
                to_del = st.selectbox("Select user", non_admins["username"].tolist())
                if st.button("Delete User", type="primary"):
                    uid = int(all_users.loc[all_users["username"]==to_del,"id"].values[0])
                    delete_user(uid)
                    st.success(f"User **{to_del}** deleted.")
                    st.rerun()

    with tab_l:
        st.subheader("📋 All Detection Logs")
        if not all_logs.empty:
            wanted = ["username","filename","location_name","total_frames",
                      "max_vehicles","congestion_rate","threshold","timestamp"]
            avail  = [c for c in wanted if c in all_logs.columns]
            disp   = all_logs[avail].copy()
            disp["timestamp"] = pd.to_datetime(
                disp["timestamp"], errors="coerce").dt.strftime("%d %b %Y  %H:%M")
            disp.columns = [c.replace("_"," ").title() for c in avail]
            st.dataframe(disp, use_container_width=True)
            st.download_button("📥 Export All Logs",
                               all_logs.to_csv(index=False).encode(),
                               "all_logs.csv", "text/csv")
        else:
            st.info("No logs found.")


if not st.session_state.logged_in:
    show_login()
else:
    show_sidebar()
    _page = st.session_state.page
    if   _page == "detection":       show_detection()
    elif _page == "congestion_map":  show_map()
    elif _page == "my_history":      show_history()
    elif _page == "email_settings":  show_email_settings()
    elif _page == "admin_dashboard": show_admin()
    else:                            show_detection()
