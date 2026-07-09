import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="WESAD Stress Detection",
    page_icon="🫀",
    layout="centered"
)



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #EEF3F5 0%, #E4ECEF 100%);
}

/* Hero title */
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: #123C4D;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.02rem;
    color: #4A6572;
    max-width: 640px;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}

/* Heartbeat pulse divider */
.pulse-divider {
    width: 100%;
    height: 28px;
    margin: 1.2rem 0 1.6rem 0;
}

/* Section card */
.section-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 2px 14px rgba(18, 60, 77, 0.06);
    border: 1px solid rgba(18, 60, 77, 0.06);
    margin-bottom: 1.4rem;
}

.section-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2A9D8F;
    margin-bottom: 0.3rem;
}

.section-heading {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #123C4D;
    margin-bottom: 0.8rem;
}

/* Streamlit widget overrides */
div[data-testid="stRadio"] label, .stSlider label, .stFileUploader label {
    font-weight: 500 !important;
    color: #123C4D !important;
}

.stButton > button {
    background: linear-gradient(135deg, #146C6D 0%, #0F4C5C 100%);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 1.6rem;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(20, 108, 109, 0.25);
    transition: transform 0.15s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(20, 108, 109, 0.32);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #F4F9F9;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #2A9D8F;
}

div[data-testid="stMetricLabel"] {
    color: #4A6572 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #123C4D !important;
    font-family: 'Fraunces', serif !important;
}

/* Info / warning boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

footer, .stCaption {
    color: #7A8C94 !important;
}
</style>
""", unsafe_allow_html=True)



st.markdown('<div class="hero-title">🫀 WESAD Stress Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">A subject-independent CNN model that classifies psychological '
    'state — Baseline, Stress, or Amusement — from 8-channel physiological signals '
    '(ACC, ECG, EMG, EDA, Temperature, Respiration).</div>',
    unsafe_allow_html=True
)

# Heartbeat pulse SVG divider (signature element)
st.markdown("""
<svg class="pulse-divider" viewBox="0 0 600 28" xmlns="http://www.w3.org/2000/svg">
  <polyline points="0,14 180,14 200,4 215,24 230,14 600,14"
    fill="none" stroke="#2A9D8F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_wesad_model.h5")

model = load_model()

LABEL_NAMES = {0: "Baseline", 1: "Stress", 2: "Amusement"}
LABEL_COLORS = {0: "#2A9D8F", 1: "#E76F51", 2: "#F4A261"}
CHANNEL_NAMES = ["ACC_x", "ACC_y", "ACC_z", "ECG", "EMG", "EDA", "Temp", "Resp"]



st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Step 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Select Data Source</div>', unsafe_allow_html=True)

option = st.radio(
    "Input method",
    ["Use a sample from the existing dataset (demo)", "Upload a .npy file for a new signal window"],
    label_visibility="collapsed"
)

window = None

if option == "Use a sample from the existing dataset (demo)":
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    sample_idx = st.slider("Sample index", 0, len(X) - 1, 0)
    window = X[sample_idx]

    true_label_map = {1: "Baseline", 2: "Stress", 3: "Amusement"}
    st.info(f"Recorded true label for this sample: **{true_label_map.get(y[sample_idx], 'Unknown')}**")
else:
    uploaded_file = st.file_uploader("Upload a .npy file with shape (700, 8)", type=["npy"])
    if uploaded_file is not None:
        window = np.load(uploaded_file)
        if window.shape != (700, 8):
            st.error(f"File shape must be (700, 8), but got {window.shape}")
            window = None

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# SIGNAL PREVIEW + PREDICTION
# =====================================================

if window is not None:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Signal Preview</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    palette = ["#8AB6D6", "#B8A9C9", "#6FB3A0", "#E76F51", "#9A8C98", "#2A9D8F", "#F4A261", "#457B9D"]
    for i, ch_name in enumerate(CHANNEL_NAMES):
        ax.plot(window[:, i], label=ch_name, color=palette[i], linewidth=1.2)
    ax.set_xlabel("Time Steps", color="#4A6572")
    ax.set_ylabel("Value", color="#4A6572")
    ax.tick_params(colors="#4A6572")
    for spine in ax.spines.values():
        spine.set_color("#D6E0E3")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    st.pyplot(fig, transparent=True)

    predict_clicked = st.button("🔍 Predict Psychological State")

    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:

        input_data = np.expand_dims(window, axis=0)
        prediction = model.predict(input_data)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class] * 100

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Prediction</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted State", LABEL_NAMES[predicted_class])
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        fig2, ax2 = plt.subplots(figsize=(6, 2.8))
        fig2.patch.set_alpha(0.0)
        ax2.set_facecolor("none")
        bars = ax2.bar(
            [LABEL_NAMES[i] for i in range(3)],
            prediction * 100,
            color=[LABEL_COLORS[i] for i in range(3)],
            width=0.55
        )
        ax2.set_ylabel("Confidence (%)", color="#4A6572")
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors="#4A6572")
        for spine in ax2.spines.values():
            spine.set_color("#D6E0E3")
        for bar, val in zip(bars, prediction * 100):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%",
                      ha="center", color="#123C4D", fontweight="bold", fontsize=9)
        st.pyplot(fig2, transparent=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Select a sample or upload a file to run a prediction.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("WESAD Psychological State Classification · CNN · Subject-Independent Evaluation · 91% Test Accuracy")