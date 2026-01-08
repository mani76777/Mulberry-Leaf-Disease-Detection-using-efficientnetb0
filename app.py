import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from gradcam import get_gradcam_heatmap, calculate_severity_percentage

# 1. Detailed Multilingual UI & Treatment Plans (English, Telugu, Hindi, Kannada)
UI = {
    "English": {
        "title": "Mulberry AI: Smart Disease Diagnosis",
        "up": "Upload Mulberry Leaf Image",
        "res": "Detection Result",
        "sev": "Infection Severity",
        "rec": "AI Treatment Protocol",
        "Healthy": "✅ **Status:** Leaf is healthy.\n\n🚜 **Action:** Maintain routine 90cm x 90cm spacing and regular irrigation. No chemical intervention needed.",
        "Leaf Rust": "💊 **Chemical:** Spray **0.2% Kavach (Chlorothalonil 75% WP)**.\n\n🚜 **Cultural:** Avoid delayed leaf harvest. Prune and burn infected branches.\n\n⏳ **Safety Period:** Wait **5 days** before feeding these leaves to silkworms.",
        "Leaf Spot": "💊 **Chemical:** Spray **0.1% Bavistin (Carbendazim 50% WP)** or 0.2% Dithane M-45.\n\n🚜 **Cultural:** Immediately remove and burn infected leaves. Improve field sanitation.\n\n⏳ **Safety Period:** Wait **8 days** before feeding these leaves to silkworms."
    },
    "Kannada": {
        "title": "ಹಿಪ್ಪುನೇರಳೆ AI: ಸ್ಮಾರ್ಟ್ ರೋಗ ಪತ್ತೆ",
        "up": "ಹಿಪ್ಪುನೇರಳೆ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "res": "ಪತ್ತೆಯಾದ ಫಲಿತಾಂಶ",
        "sev": "ರೋಗದ ತೀವ್ರತೆಯ ಪ್ರಮಾಣ",
        "rec": "AI ಚಿಕಿತ್ಸಾ ಕ್ರಮ",
        "Healthy": "✅ **ಸ್ಥಿತಿ:** ಎಲೆಯು ಆರೋಗ್ಯವಾಗಿದೆ.\n\n🚜 **ಕ್ರಮ:** ಕ್ರಮಬದ್ಧವಾಗಿ ನೀರಾವರಿ ಮತ್ತು 90cm x 90cm ಅಂತರವನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ.",
        "Leaf Rust": "💊 **ಔಷಧಿ:** **0.2% ಕವಚ (Chlorothalonil 75% WP)** ಸಿಂಪಡಿಸಿ.\n\n🚜 **ಕ್ರಮ:** ಎಲೆ ಕಟಾವು ವಿಳಂಬ ಮಾಡಬೇಡಿ. ರೋಗ ಪೀಡಿತ ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ ಸುಟ್ಟು ಹಾಕಿ.\n\n⏳ **ಮುನ್ನೆಚ್ಚರಿಕೆ:** ಔಷಧ ಸಿಂಪಡಿಸಿದ **5 ದಿನಗಳ** ವರೆಗೆ ರೇಷ್ಮೆ ಹುಳುಗಳಿಗೆ ಈ ಎಲೆಗಳನ್ನು ಹಾಕಬಾರದು.",
        "Leaf Spot": "💊 **ಔಷಧಿ:** **0.1% ಬ್ಯಾವಿಸ್ಟಿನ್ (Carbendazim 50% WP)** ಅಥವಾ 0.2% ಡೈಥೇನ್ M-45 ಸಿಂಪಡಿಸಿ.\n\n🚜 **ಕ್ರಮ:** ರಂಧ್ರಗಳಿರುವ ರೋಗಗ್ರಸ್ತ ಎಲೆಗಳನ್ನು ಕಿತ್ತು ಸುಟ್ಟು ಹಾಕಿ. ತೋಟವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ.\n\n⏳ **ಮುನ್ನೆಚ್ಚರಿಕೆ:** ಔಷಧ ಸಿಂಪಡಿಸಿದ **8 ದಿನಗಳ** ವರೆಗೆ ರೇಷ್ಮೆ ಹುಳುಗಳಿಗೆ ಈ ಎಲೆಗಳನ್ನು ಹಾಕಬಾರದು."
    },
    "Telugu": {
        "title": "మల్బరీ AI: స్మార్ట్ వ్యాధి నిర్ధారణ",
        "up": "మల్బరీ ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "res": "నిర్ధారణ ఫలితం",
        "sev": "ವ್ಯಾಧಿ తీవ్రత శాతం",
        "rec": "AI చికిత్స విధానం",
        "Healthy": "✅ **స్థితి:** ఆకు ఆరోగ్యంగా ఉంది.\n\n🚜 **చర్య:** క్రమం తప్పకుండా నీటి పారుదల మరియు 90cm x 90cm దూరం పాటించండి.",
        "Leaf Rust": "💊 **మందు:** **0.2% కవచ్ (Chlorothalonil 75% WP)** స్ప్రే చేయండి.\n\n🚜 **పద్ధతి:** ఆకు కోత ఆలస్యం చేయవద్దు. వ్యాధి సోకిన కొమ్మలను కత్తిరించి కాల్చివేయండి.\n\n⏳ **జಾಗ్రತ್ತ:** మందు చల్లిన **5 రోజుల** వరకు పట్టు పురుగులకు ఈ ఆకులను మేపకూడదు.",
        "Leaf Spot": "💊 **మందు:** **0.1% బావిస్టిన్ (Carbendazim 50% WP)** లేదా 0.2% డైథేన్ M-45 స్ప్రే చేయండి.\n\n🚜 **పద్ధತಿ:** రంధ్రాలు పడ్డ ఆకులను ఏరి కాల్చివేయండి. తోటను శుభ్రంగా ఉంచండి.\n\n⏳ **జాగ్రత్త:** మందు చల్లిన **8 రోజుల** వరకు పట్టు పురుగులకు ఈ ఆకులను మేపకూడదు."
    },
    "Hindi": {
        "title": "शहतूत AI: स्मार्ट रोग निदान",
        "up": "शहतूत की पत्ती की फोटो अपलोड करें",
        "res": "जांच का परिणाम",
        "sev": "संक्रमण की गंभीरता",
        "rec": "एआई उपचार योजना",
        "Healthy": "✅ **स्थिति:** पत्ती स्वस्थ है।\n\n🚜 **कार्रवाई:** नियमित सिंचाई और 90cm x 90cm की दूरी बनाए रखें।",
        "Leaf Rust": "💊 **दवा:** **0.2% कवच (Chlorothalonil 75% WP)** का छिड़काव करें।\n\n🚜 **तरीका:** पत्तियों की कटाई में देरी न करें। संक्रमित शाखाओं को जला दें।\n\n⏳ **सुरक्षा अवधि:** छिड़काव के **5 दिनों** तक रेशम कीटों को पत्ते न खिलाएं।",
        "Leaf Spot": "💊 **दवा:** **0.1% बाविस्टिन (Carbendazim 50% WP)** या 0.2% डायथेन M-45 का छिड़काव करें।\n\n🚜 **तरीका:** छेद वाली संक्रमित पत्तियों को हटाकर जला दें। खेत की सफाई रखें।\n\n⏳ **सुरक्षा अवधि:** छिड़काव के **8 दिनों** तक रेशम कीटों को पत्ते न खिलाएं।"
    }
}

st.set_page_config(page_title="Mulberry AI", layout="wide")

# Sidebar for Language Selection
lang = st.sidebar.selectbox("🌐 Choose Language / ಭಾಷೆ / భాష / भाषा", ["English", "Kannada", "Telugu", "Hindi"])
t = UI[lang]

st.title(t["title"])
st.markdown("---")

@st.cache_resource
def load_mulberry_model():
    return tf.keras.models.load_model("model/efficientnetb0_mulberry.keras", compile=False)

model = load_mulberry_model()
CLASS_NAMES = ["Healthy", "Leaf Rust", "Leaf Spot"]

uploaded = st.file_uploader(t["up"], type=["jpg", "png", "jpeg"])

if uploaded:
    col1, col2 = st.columns([1, 1.2])
    img = Image.open(uploaded).convert("RGB")
    
    with col1:
        st.image(img, caption="Uploaded Leaf", use_container_width=True)

    # Processing
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    confidence = np.max(preds) * 100
    
    heatmap = get_gradcam_heatmap(img_array, model)

    with col2:
        st.subheader(f"{t['res']}: {label}")
        st.write(f"**AI Confidence:** {confidence:.2f}%")
        
        if isinstance(heatmap, np.ndarray):
            try:
                pct, sev_text, emoji = calculate_severity_percentage(heatmap)
                st.metric(label=t["sev"], value=f"{emoji} {sev_text}", delta=f"{pct:.1f}% Area Infected")
                
                heatmap_res = cv2.resize(heatmap, (img.width, img.height))
                heatmap_u8 = np.uint8(255 * heatmap_res)
                heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
                
                orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)
                
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="AI Vision Analysis", use_container_width=True)
            except Exception as e:
                st.error(f"Visualization Error: {e}")
        else:
            st.warning("Heatmap not available.")

    st.markdown("---")
    st.subheader(f"📋 {t['rec']}")
    st.info(t[label])

    if label != "Healthy":
        st.warning("⚠️ **Safety Warning:** Ensure the 'Safety Period' is observed before feeding silkworms.")