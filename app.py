import streamlit as st
import numpy as np
import cv2
from PIL import Image
from gradcam import get_gradcam_heatmap, calculate_severity_percentage

# ===============================
# Multilingual UI Dictionary
# ===============================
UI = {
    "English": {
        "title": "Mulberry AI: Smart Disease Diagnosis",
        "up": "Upload Mulberry Leaf Image",
        "res": "Detection Result",
        "sev": "Infection Severity",
        "rec": "AI Treatment Protocol",
        "Healthy": "✅ **Status:** Leaf is healthy.\n\n🚜 Maintain routine spacing and irrigation.",
        "Leaf Rust": "💊 Spray **0.2% Kavach (Chlorothalonil 75% WP)**.\n⏳ Safety period: **5 days**.",
        "Leaf Spot": "💊 Spray **0.1% Bavistin (Carbendazim 50% WP)**.\n⏳ Safety period: **8 days**."
    },
    "Kannada": {
        "title": "ಹಿಪ್ಪುನೇರಳೆ AI: ಸ್ಮಾರ್ಟ್ ರೋಗ ಪತ್ತೆ",
        "up": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "res": "ಪತ್ತೆಯಾದ ಫಲಿತಾಂಶ",
        "sev": "ರೋಗದ ತೀವ್ರತೆ",
        "rec": "AI ಚಿಕಿತ್ಸಾ ಕ್ರಮ",
        "Healthy": "ಎಲೆ ಆರೋಗ್ಯವಾಗಿದೆ.",
        "Leaf Rust": "0.2% ಕವಚ ಸಿಂಪಡಿಸಿ.",
        "Leaf Spot": "0.1% ಬ್ಯಾವಿಸ್ಟಿನ್ ಸಿಂಪಡಿಸಿ."
    },
    "Telugu": {
        "title": "మల్బరీ AI: వ్యాధి నిర్ధారణ",
        "up": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "res": "ఫలితం",
        "sev": "తీవ్రత",
        "rec": "చికిత్స",
        "Healthy": "ఆకు ఆరోగ్యంగా ఉంది.",
        "Leaf Rust": "0.2% కవచ్ స్ప్రే చేయండి.",
        "Leaf Spot": "0.1% బావిస్టిన్ స్ప్రే చేయండి."
    },
    "Hindi": {
        "title": "शहतूत AI: रोग पहचान",
        "up": "पत्ती की फोटो अपलोड करें",
        "res": "परिणाम",
        "sev": "संक्रमण स्तर",
        "rec": "उपचार",
        "Healthy": "पत्ती स्वस्थ है।",
        "Leaf Rust": "0.2% कवच का छिड़काव करें।",
        "Leaf Spot": "0.1% बाविस्टिन का छिड़काव करें।"
    }
}

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Mulberry AI", layout="wide")

lang = st.sidebar.selectbox("🌐 Choose Language", ["English", "Kannada", "Telugu", "Hindi"])
t = UI[lang]

st.title(t["title"])
st.info("⚠️ Demo Mode: Model inference disabled for cloud deployment")

CLASS_NAMES = ["Healthy", "Leaf Rust", "Leaf Spot"]

uploaded = st.file_uploader(t["up"], type=["jpg", "png", "jpeg"])

if uploaded:
    col1, col2 = st.columns([1, 1.2])
    img = Image.open(uploaded).convert("RGB")

    with col1:
        st.image(img, caption="Uploaded Leaf", use_container_width=True)

    # -------------------------------
    # DEMO prediction (random)
    # -------------------------------
    label = np.random.choice(CLASS_NAMES)
    confidence = np.random.uniform(85, 97)

    with col2:
        st.subheader(f"{t['res']}: {label}")
        st.write(f"**AI Confidence:** {confidence:.2f}%")

        # Fake heatmap for demo
        heatmap = np.random.rand(img.height, img.width)

        pct, sev_text, emoji = calculate_severity_percentage(heatmap)
        st.metric(label=t["sev"], value=f"{emoji} {sev_text}", delta=f"{pct:.1f}% Area")

        heatmap_u8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption="AI Vision Analysis (Demo)",
                 use_container_width=True)

    st.markdown("---")
    st.subheader(f"📋 {t['rec']}")
    st.info(t[label])
