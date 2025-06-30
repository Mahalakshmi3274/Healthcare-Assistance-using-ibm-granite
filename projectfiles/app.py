import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "1"

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import altair as alt

# ✅ Page Config
st.set_page_config(page_title="HealthAI", layout="wide", initial_sidebar_state="expanded")

# ✅ Styling
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #f0f4f8 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e0f7fa, #e1bee7);
            color: #000;
            border-right: 2px solid #90caf9;
            padding: 20px 10px;
            border-radius: 0 12px 12px 0;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        .stTextArea, .stTextInput, .stFileUploader, .stSelectbox, .stNumberInput {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #6a1b9a, #ab47bc);
            color: white !important;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #7b1fa2, #ce93d8);
        }
        .sidebar-nav button {
            width: 100%;
            margin-bottom: 10px;
            background: #6a1b9a;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        .sidebar-nav button:hover {
            background: #8e24aa;
        }
        .theme-toggle-button button {
            background: linear-gradient(90deg, #4a148c, #7b1fa2);
            color: white;
            padding: 8px 14px;
            font-size: 14px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Theme toggle
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light'

if st.sidebar.button("🌓 Toggle Theme", key="theme_toggle", help="Switch between Light and Dark mode"):
    st.session_state['theme'] = 'Dark' if st.session_state['theme'] == 'Light' else 'Light'
    st.rerun()

if st.session_state['theme'] == 'Dark':
    st.markdown("""
        <style>
        html, body {
            background-color: #1e1e1e !important;
            color: #f0f0f0 !important;
        }
        .stTextArea, .stTextInput, .stFileUploader, .stSelectbox, .stNumberInput {
            background-color: #2e2e2e !important;
            border: 1px solid #555 !important;
            color: #fff !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a237e, #4a148c);
        }
        </style>
    """, unsafe_allow_html=True)

# ✅ Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #4a148c;'>🧠 HealthAI</h2>", unsafe_allow_html=True)
    nav_pages = [
        "Dashboard", "Patient Chat", "Disease Prediction",
        "Treatment Plans", "Document Checkup", "Health Analytics", "Chat History"
    ]
    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    for p in nav_pages:
        if st.button(p):
            st.session_state["page"] = p
    st.markdown("</div>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

page = st.session_state["page"]

# ✅ Load model
@st.cache_resource(show_spinner="🧠 Loading Medical Model...")
def load_model():
    model_path = "./models/ibm-granite-3.3-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# ✅ Ask AI
def ask_ai(prompt, model, tokenizer, device):
    system_prompt = (
        "You are HealthAI, a responsible and ethical AI doctor assistant. "
        "\n🧪 Summary\n🔍 Possible Causes\n💊 Recommended Steps\n⚠️ Reminder to consult a real doctor\n\n"
    )
    inputs = tokenizer(system_prompt + prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split(system_prompt.strip())[-1].strip()

# ✅ Page Routing
if page == "Dashboard":
    st.markdown("<h1 style='text-align:center;'>🏥 Welcome to HealthAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Comprehensive AI platform for smarter healthcare.</p>", unsafe_allow_html=True)

    with st.expander("🩺 Patient Profile"):
        st.session_state['patient_name'] = st.text_input("Name")
        st.session_state['patient_age'] = st.number_input("Age", min_value=0, max_value=120)
        st.session_state['patient_gender'] = st.selectbox("Gender", ["Male", "Female", "Other"])

    cols = st.columns(3)
    features = [
        ("💬 Patient Chat", "Talk to AI for symptoms or advice."),
        ("🦠 Disease Prediction", "Get likely diseases from symptoms."),
        ("💊 Treatment Plans", "AI-curated medical suggestions."),
        ("📁 Document Checkup", "Upload health documents for review."),
        ("📊 Health Analytics", "Generate metrics and visualizations from reports."),
        ("📚 Chat History", "Access past medical Q&A.")
    ]
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.info(f"### {title}\n{desc}")

elif page == "Health Analytics":
    st.header("📊 Health Analytics")
    file = st.file_uploader("📎 Upload Health Report", type=["pdf", "jpg", "jpeg", "png"], key="analytics")
    extracted = ""
    if file:
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                extracted += page.get_text()
        else:
            image = Image.open(file)
            extracted = ask_ai("Summarize this health report.", model, tokenizer, device)

        if extracted:
            st.success("✅ Extracted health report data.")
            with st.expander("📄 Extracted Text"):
                st.code(extracted[:1500])
            prompt = f"Patient Age: {st.session_state.get('patient_age', 'Unknown')}\nAnalyze health metrics:\n{extracted}"
            result = ask_ai(prompt, model, tokenizer, device)
            st.subheader("🧠 AI Health Insights")
            st.markdown(result)

            chart_data = pd.DataFrame({
                'Metric': ['Cholesterol', 'Blood Sugar', 'BP Systolic', 'BP Diastolic'],
                'Past Value': [200, 130, 145, 95],
                'Present Value': [190, 120, 135, 90]
            })

            chart = alt.Chart(chart_data).transform_fold(
                ['Past Value', 'Present Value'],
                as_=['Report Type', 'Value']
            ).mark_bar().encode(
                x='Metric:N',
                y='Value:Q',
                color='Report Type:N',
                column='Metric:N'
            ).properties(width=80)

            st.altair_chart(chart, use_container_width=True)

# ✅ Additional pages here as needed (Patient Chat, Disease Prediction, Treatment Plans, etc.)
# ✅ Add rest of the pages below as needed (e.g., Patient Chat, Chat History, etc.)

elif page == "Patient Chat":
    st.header("💬 Patient Chat")
    query = st.text_area("Describe your health issue:")
    if st.button("Ask HealthAI") and query:
        response = ask_ai(query, model, tokenizer, device)
        st.markdown(f"### 🧠 AI Response:\n{response}")

elif page == "Disease Prediction":
    st.header("🦠 Disease Prediction")
    symptoms = st.text_input("Enter symptoms (comma-separated):")
    if st.button("Predict") and symptoms:
        prompt = (
            f"A patient is experiencing the following symptoms: {symptoms}. "
            "List possible medical conditions or diseases that could cause these symptoms. "
            "Provide a bullet-point list of the most likely diagnoses, and explain briefly why each might match."
        )
        with st.spinner("🧠 Analyzing symptoms..."):
            response = ask_ai(prompt, model, tokenizer, device)
        st.success("Prediction complete.")
        st.markdown(f"### 🧠 Possible Diseases\n{response}")

elif page == "Treatment Plans":
    st.header("💊 Treatment Plans")
    condition = st.text_input("Enter condition name:")
    if st.button("Suggest Treatment") and condition:
        prompt = f"Suggest treatment plan for {condition}."
        response = ask_ai(prompt, model, tokenizer, device)
        st.markdown(f"### 🩺 Suggested Treatment:\n{response}")

elif page == "Document Checkup":
    st.header("📁 Document Checkup")
    file = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
    extracted_text = ""
    if file:
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                extracted_text += page.get_text()
        else:
            image = Image.open(file)
            extracted_text = ask_ai("Extract health data from this image.", model, tokenizer, device)

        if extracted_text:
            with st.expander("📄 Extracted Text"):
                st.code(extracted_text[:1500])
            result = ask_ai(f"Analyze the health report:\n{extracted_text}", model, tokenizer, device)
            st.subheader("📈 Health Summary")
            st.markdown(result)
        else:
            st.error("No extractable data found.")


elif page == "Chat History":
    st.header("📚 Chat History")
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            history = json.load(f)
        for chat in reversed(history[-20:]):
            with st.expander(f"🕓 {chat['timestamp']}"):
                st.markdown(f"**User:** {chat['user']}")
                st.markdown(f"**AI:** {chat['ai']}")
    else:
        st.info("No past chats found.")
