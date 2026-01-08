# 🌿 Mulberry Leaf Disease Detection using EfficientNetB0

This project presents a deep learning–based system for detecting diseases in mulberry leaves using the EfficientNetB0 convolutional neural network.  
It includes a user-friendly web interface with multi-language support and Grad-CAM visualization to provide explainable and transparent predictions.

---

## 📌 Features
- Classification of mulberry leaf diseases: **Healthy, Rust, Leaf Spot**
- EfficientNetB0-based deep learning model
- Multi-language web interface
- Grad-CAM visualization to highlight affected regions
- Flask-based web application for real-time prediction

---

## 🧠 Model Used
- **EfficientNetB0**
- Transfer learning for improved accuracy
- Trained using categorical cross-entropy loss and Adam optimizer

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Flask
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure
leafdiseasedetection/
│── app.py
│── train.py
│── evaluate.py
│── gradcam.py
│── requirements.txt
│── dataset/ # Not included (large size)
│── model/ # Not included (large size)
│── screenshots/ # Optional (UI & Grad-CAM images)
│── README.md


---

## 📊 Dataset & Trained Model
Due to size limitations, the dataset and trained model files are not included in this repository.

- 📁 Dataset: Add Google Drive link here
- 🧠 Trained Model: Add Google Drive link here

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python app.py

🔍 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the regions of the mulberry leaf image that influence the model’s prediction, improving trust and explainability in the AI system.

🌐 Multi-Language Support

The web interface supports multiple languages, enabling broader accessibility for farmers and researchers.

🎯 Applications

Smart agriculture

Early disease detection

Farmer decision support systems

Explainable AI research

👤 Author

Manikanth K
Deep Learning | Computer Vision | Cybersecurity
