import streamlit as st
import base64
import os

# Page Config
st.set_page_config(page_title="Multimodal Driver Stress Analysis", layout="wide")
# Helper to convert local image to base64

# --- Hide Streamlit default UI elements ---
hide_streamlit_style = """
    <style>
    /* Hide deploy button, hamburger menu, footer, and top bar */
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {visibility: hidden !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stAppDeployButton {visibility: hidden !important;}
    /* Optional: hide rainbow top progress bar */
    div[data-testid="stProgressBar"] {visibility: hidden !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load base64 images (optional)
asset_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'assets'))

def safe_get_base64(image_path):
    try:
        return get_base64_image(image_path)
    except Exception:
        return ""

img_bg = safe_get_base64(os.path.join(asset_dir, "hero1.webp"))
img1 = safe_get_base64(os.path.join(asset_dir, "diagnosis.webp"))
img2 = safe_get_base64(os.path.join(asset_dir, "ecg.jpg"))
img3 = safe_get_base64(os.path.join(asset_dir, "reports.webp"))
img4 = safe_get_base64(os.path.join(asset_dir, "trends.webp"))

# Prepare a safe CSS snippet for background image (avoid backslashes inside f-strings)
if img_bg:
    bg_style = "background-image: url('data:image/png;base64," + img_bg + "'); background-size: cover; background-position: center; opacity: 0.5;"
else:
    bg_style = ""

# ---------- Premium + Sidebar Styling ----------
st.markdown(f"""
    <style>
    .stApp {{
        background-color: #0f172a;
        background-image: linear-gradient(to right, #1e293b, #0f172a);
        font-family: 'Segoe UI', sans-serif;
        animation: fadeIn 1s ease-in-out;
        color: #f8fafc;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    section[data-testid="stSidebar"] {{
        background-color: #f1f5f9;
        color: #0f172a;
        border-right: 1px solid #cbd5e1;
    }}
    section[data-testid="stSidebar"] * {{
        color: #0f172a !important;
    }}
    section[data-testid="stSidebar"] a {{
        color: #1e293b !important;
        text-decoration: none;
        font-weight: 500;
        border-radius: 0.4rem;
        padding: 0.4rem 1rem;
        display: block;
        transition: background 0.2s ease;
    }}
    section[data-testid="stSidebar"] a:hover {{
        background-color: #e2e8f0;
    }}

    .title-container {{
        background: linear-gradient(to right, #2563eb, #38bdf8);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        animation: slideUp 1s ease-in-out;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .title-container::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        {bg_style}
    }}
    .title-container h1,
    .title-container p {{
        position: relative;
        z-index: 1;
    }}

    /* -------- New Styles for Hero Text -------- */
    .hero-text {{
        color: #0f172a;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6);
    }}

    .button-style {{
        background-color: white;
        color: #2563eb;
        border: 2px solid #2563eb;
        border-radius: 5px;
        padding: 0.5rem 1.2rem;
        margin-right: 1rem;
        font-weight: 500;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
    }}
    .button-style:hover {{
        background-color: #2563eb;
        color: white;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}

    .section-header {{
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        justify-content: center;
        gap: 20px;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }}
    .feature-box {{
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: #f8fafc;
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }}
    .feature-box:hover {{
        transform: scale(1.05);
    }}
    .feature-box h4 {{
        color: #38bdf8;
        margin-top: 10px;
    }}

    .divider {{
        border-top: 2px solid #334155;
        margin: 2rem 0;
    }}

    .carousel-wrapper {{
        overflow: hidden;
        width: 100%;
        margin: 30px 0;
    }}
    .carousel-track {{
        display: flex;
        width: max-content;
        animation: scrollLeft 30s linear infinite;
    }}
    @keyframes scrollLeft {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    .card {{
        min-width: 300px;
        max-width: 300px;
        margin-right: 20px;
        background-color: #1e293b;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease;
        color: #f8fafc;
    }}
    .card:hover {{
        transform: scale(1.05);
    }}
    .card img {{
        width: 100%;
        height: 180px;
        object-fit: cover;
        border-radius: 10px;
    }}
    .card h4 {{
        margin-top: 15px;
        font-size: 20px;
        color: #38bdf8;
    }}
    .card p {{
        color: #cbd5e1;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- Hero Section ----------
st.markdown(f"""
<div class="title-container">
    <h1 class="hero-text">Welcome to the Multimodal Driver Stress Detection Dashboard</h1>
    <p style="font-size: 22px;" class="hero-text">This advanced tool integrates multiple physiological signals including 
            ECG, EMG, Respiration, Foot GSR and Hand GSR to enable real-time accurate detection of 
            Driver Stress Levels during driving. Experience intelligent analysis insights for accurate stress detection.</p>
</div>
""", unsafe_allow_html=True)



# # ---------- Navigation Tip ----------
# with st.expander("ℹ️ How to Navigate"):
#     st.markdown("""
#         Use the **sidebar on the left** to access:
#         - Signal Visualization & R-Peak Detection
#         - HRV Feature Extraction
#         - EEG/EMG/EOG Analysis
#         - Multi-Label Classification
#         - Final Reports
#     """)

# # ---------- Informational Section ----------
# st.markdown('<div class="section-header"><h2 id="learn">What is Sleep Apnea?</h2></div>', unsafe_allow_html=True)
# st.markdown("""
# Sleep apnea is a serious disorder where breathing repeatedly stops and starts during sleep. If untreated, it can lead to high blood pressure, cardiovascular issues, fatigue, and cognitive problems.
# """)

# # ---------- Feature Grid ----------
# st.markdown("""
# <div class="feature-grid">
#     <div class="feature-box">
#         <h4>ECG</h4>
#         <p>Tracks heart rhythm for apnea-linked arrhythmias.</p>
#     </div>
#     <div class="feature-box">
#         <h4>EEG</h4>
#         <p>Captures brain activity and sleep stage transitions.</p>
#     </div>
#     <div class="feature-box">
#         <h4>EOG</h4>
#         <p>Measures eye movement for REM/NREM tracking.</p>
#     </div>
#     <div class="feature-box">
#         <h4>EMG</h4>
#         <p>Detects muscle activity and atonia during sleep.</p>
#     </div>
#     <div class="feature-box">
#         <h4>SpO₂</h4>
#         <p>Monitors oxygen saturation for hypoxic episodes.</p>
#     </div>
#     <div class="feature-box" style="visibility: hidden;"></div>
# </div>
# """, unsafe_allow_html=True)

# # ---------- Feature Cards Carousel ----------
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
# st.markdown(f"""
# <div class="carousel-wrapper">
#   <div class="carousel-track">
#     <div class="card">
#       <img src="data:image/webp;base64,{img1}" />
#       <h4>Sleep Apnea Diagnosis</h4>
#       <p>Fusion of physiological signals to detect OSA, CSA, and mixed events in real-time.</p>
#     </div>
#     <div class="card">
#       <img src="data:image/jpeg;base64,{img2}" />
#       <h4>Signal Monitoring</h4>
#       <p>Visualize and process ECG, EEG, EMG, EOG, and SpO₂ with advanced algorithms.</p>
#     </div>
#     <div class="card">
#       <img src="data:image/webp;base64,{img3}" />
#       <h4>Smart Reports</h4>
#       <p>Generate rich, detailed reports tailored for physicians and clinical use.</p>
#     </div>
#     <div class="card">
#       <img src="data:image/webp;base64,{img4}" />
#       <h4>Trends & Insights</h4>
#       <p>Track apnea trends, stage transitions, and variability patterns over time.</p>
#     </div>
#     <div class="card">
#       <img src="data:image/webp;base64,{img1}" />
#       <h4>Sleep Apnea Diagnosis</h4>
#       <p>Fusion of physiological signals to detect OSA, CSA, and mixed events in real-time.</p>
#     </div>
#     <div class="card">
#       <img src="data:image/jpeg;base64,{img2}" />
#       <h4>Signal Monitoring</h4>
#       <p>Visualize and process ECG, EEG, EMG, EOG, and SpO₂ with advanced algorithms.</p>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# # ---------- Footer ----------
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
# st.markdown("""
# <center>
#     💤 Empowering Early Diagnosis through Data. <br>
#     <sub>© 2025 Multimodal Sleep Analytics | Built with ❤️ using AI & Biomedical Signals</sub>
# </center>
# """, unsafe_allow_html=True)
