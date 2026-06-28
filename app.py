# ============================================================
# RICEGUARD AI - STREAMLIT APP (STREAMLIT CLOUD COMPATIBLE)
# Save as: app.py
# Run locally: streamlit run app.py
# ============================================================

import streamlit as st
import os
import sys

# ============================================================
# PAGE CONFIG (Must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="RiceGuard AI | Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ERROR HANDLING FOR IMPORTS
# ============================================================
try:
    import numpy as np
    import cv2
    from PIL import Image
    import joblib
    import requests
    from io import BytesIO
    from skimage.feature import hog, graycomatrix, graycoprops
    import matplotlib.pyplot as plt
    from datetime import datetime
    import time
except ImportError as e:
    st.error(f"""
    ❌ **Import Error: {e}**
    
    This usually means a required package is missing. 
    Please check that your `requirements.txt` includes:
    - opencv-python-headless
    - scikit-image
    - scikit-learn
    - xgboost
    - lightgbm
    - joblib
    """)
    st.stop()

# ============================================================
# GET BASE DIRECTORY (Works on both local and cloud)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# Verify models directory exists
if not os.path.exists(MODELS_DIR):
    st.error(f"""
    ❌ **Models directory not found!**
    
    Expected: `{MODELS_DIR}`
    
    Please ensure your `saved_models` folder is uploaded to GitHub 
    alongside `app.py`.
    """)
    st.stop()

# ============================================================
# CUSTOM CSS - MODERN DARK THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .disease-alert {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .healthy-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #10b981, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3) !important;
    }
    
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95) !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #10b981, #3b82f6) !important;
        border-radius: 10px !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    .stFileUploader > div > button {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 2px dashed rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
        color: #94a3b8 !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    .stMultiSelect > div > div {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 10px !important;
    }
    
    .metric-container {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0;
        font-size: 0.9rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_all_models():
    """Load all trained models and preprocessing objects."""
    models = {}
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl',
        'KNN': 'knn.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'XGBoost': 'xgboost.pkl',
        'AdaBoost': 'adaboost.pkl',
        'Extra Trees': 'extra_trees.pkl',
        'LightGBM': 'lightgbm.pkl'
    }
    
    missing_files = []
    
    for name, filename in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
        else:
            missing_files.append(filename)
    
    if missing_files:
        st.warning(f"⚠️ Missing model files: {', '.join(missing_files)}")
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    feature_params = joblib.load(os.path.join(MODELS_DIR, 'feature_params.pkl'))
    
    return models, scaler, label_encoder, feature_params

try:
    models, scaler, label_encoder, feature_params = load_all_models()
    models_loaded = True
    if len(models) == 0:
        st.error("❌ No models loaded! Check saved_models folder.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("Please ensure saved_models folder contains all .pkl files")
    models_loaded = False
    st.stop()

# ============================================================
# RICE PLANT DETECTION
# ============================================================

def is_rice_plant(image):
    """
    Simple heuristic to check if image likely contains a rice plant.
    Checks for dominant green color typical of plant leaves.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
    
    is_plant = green_ratio > 0.15
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    has_texture = texture_score > 50
    
    return is_plant and has_texture, green_ratio, texture_score

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):
    """Extract enhanced features (color + texture + HOG) from image."""
    img_size = feature_params['img_size']
    orientations = feature_params['hog_orientations']
    pixels_per_cell = feature_params['hog_pixels_per_cell']
    cells_per_block = feature_params['hog_cells_per_block']
    
    resized = cv2.resize(image, (img_size, img_size))
    normalized = resized / 255.0
    
    features = []
    
    # 1. Color Histograms
    for i in range(3):
        hist = cv2.calcHist([normalized.astype(np.float32)], [i], None, [16], [0, 1])
        features.extend(hist.flatten())
    
    # 2. GLCM Texture Features
    gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256, symmetric=True, normed=True)
    
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.extend(graycoprops(glcm, prop).flatten())
    
    # 3. HOG Features
    fd = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, visualize=False, feature_vector=True)
    features.extend(fd)
    
    return np.array(features).reshape(1, -1)

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_with_model(image, model_name):
    """Predict using a single model."""
    features = extract_features(image)
    features_scaled = scaler.transform(features)
    
    model = models[model_name]
    pred_proba = model.predict_proba(features_scaled)[0]
    pred_class = model.predict(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    
    class_probs = {
        label_encoder.inverse_transform([i])[0]: float(prob)
        for i, prob in enumerate(pred_proba)
    }
    
    return {
        'predicted_class': pred_label,
        'confidence': float(pred_proba[pred_class]),
        'all_probabilities': class_probs
    }

def predict_with_models(image, selected_models):
    """Predict using multiple models."""
    results = {}
    for model_name in selected_models:
        results[model_name] = predict_with_model(image, model_name)
    return results

# ============================================================
# DISEASE RECOMMENDATIONS
# ============================================================

def get_recommendations(disease):
    """Get management recommendations for detected disease."""
    recommendations = {
        "Bacterial Blight Disease": {
            "icon": "🦠",
            "color": "#ef4444",
            "management": [
                "Use resistant rice varieties (e.g., IR20, IR24)",
                "Avoid excessive nitrogen fertilization",
                "Apply copper-based bactericides early",
                "Remove and destroy infected plants immediately",
                "Maintain proper field drainage"
            ],
            "prevention": [
                "Use certified disease-free seeds",
                "Practice crop rotation with non-host crops",
                "Avoid water stagnation in fields"
            ]
        },
        "Blast Disease": {
            "icon": "🍂",
            "color": "#f59e0b",
            "management": [
                "Apply fungicides: tricyclazole or isoprothiolane",
                "Maintain proper water management (intermittent irrigation)",
                "Use resistant varieties (e.g., IR64, MTU1010)",
                "Avoid dense planting - maintain proper spacing",
                "Apply silica-based fertilizers to strengthen cell walls"
            ],
            "prevention": [
                "Balanced fertilization (avoid excess nitrogen)",
                "Proper field sanitation - remove crop residues",
                "Early planting to avoid peak disease season"
            ]
        },
        "Brown Spot Disease": {
            "icon": "🟤",
            "color": "#8b5cf6",
            "management": [
                "Apply fungicides: iprodione or propiconazole",
                "Improve soil fertility with balanced NPK",
                "Use certified, treated seeds",
                "Apply micronutrients (especially zinc)",
                "Maintain proper plant spacing for air circulation"
            ],
            "prevention": [
                "Practice 2-3 year crop rotation",
                "Use resistant varieties",
                "Proper seed treatment before sowing"
            ]
        },
        "False Smut Disease": {
            "icon": "⚫",
            "color": "#ec4899",
            "management": [
                "Apply fungicides at flowering stage (propiconazole)",
                "Use resistant varieties",
                "Avoid high humidity conditions",
                "Ensure proper field drainage",
                "Remove infected panicles before spore release"
            ],
            "prevention": [
                "Avoid excessive nitrogen application",
                "Proper spacing to reduce humidity",
                "Timely irrigation management"
            ]
        },
        "Healthy": {
            "icon": "✅",
            "color": "#10b981",
            "management": [
                "Continue regular monitoring",
                "Maintain good agricultural practices",
                "Ensure proper irrigation scheduling",
                "Apply balanced fertilization",
                "Monitor for early signs of disease"
            ],
            "prevention": [
                "Regular field scouting",
                "Maintain optimal plant health",
                "Proper water and nutrient management"
            ]
        }
    }
    return recommendations.get(disease, recommendations["Healthy"])

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Animated header
    st.markdown("""
    <div class="animate-in">
        <h1 class="main-header">🌾 RiceGuard AI</h1>
        <p class="sub-header">Advanced Rice Disease Detection using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #10b981; font-size: 1.5rem;">⚙️ Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Selection Mode
        st.markdown("### 🤖 Model Selection")
        selection_mode = st.radio(
            "Choose prediction mode:",
            ["Single Model", "Multiple Models", "All 10 Models"],
            index=2,
            help="Select how many ML models to use for prediction"
        )
        
        # Model selection based on mode
        if selection_mode == "Single Model":
            selected_models = [st.selectbox(
                "Select one model:",
                list(models.keys()),
                index=6
            )]
        elif selection_mode == "Multiple Models":
            selected_models = st.multiselect(
                "Select 2 or more models:",
                list(models.keys()),
                default=['XGBoost', 'Random Forest', 'Extra Trees'],
                help="Select at least 2 models"
            )
            if len(selected_models) < 2:
                st.warning("⚠️ Please select at least 2 models")
                selected_models = ['XGBoost']
        else:
            selected_models = list(models.keys())
            st.success(f"✅ Using all {len(models)} models")
        
        # Input Method
        st.markdown("### 📤 Input Method")
        input_method = st.radio(
            "Choose input type:",
            ["📁 Upload Image", "🌐 Image URL"],
            index=0
        )
        
        # About section
        st.markdown("---")
        with st.expander("ℹ️ About RiceGuard AI"):
            st.markdown("""
            **RiceGuard AI** uses 10 Machine Learning algorithms to detect rice diseases from leaf images.
            
            **Supported Diseases:**
            - 🦠 Bacterial Blight Disease
            - 🍂 Blast Disease
            - 🟤 Brown Spot Disease
            - ⚫ False Smut Disease
            - ✅ Healthy Plants
            
            **Models Available:**
            """ + "\n".join([f"- {m}" for m in models.keys()]))
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    image = None
    img_array = None
    
    # Left column - Input
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0; margin-bottom: 1rem;">📸 Image Input</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Drag and drop or browse",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                help="Upload a clear image of a rice leaf"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                img_array = np.array(image)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            url = st.text_input(
                "Enter image URL:",
                placeholder="https://example.com/rice_leaf.jpg",
                help="Paste a direct link to a rice leaf image"
            )
            if url:
                try:
                    with st.spinner("Loading image from URL..."):
                        response = requests.get(url, timeout=15)
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                        img_array = np.array(image)
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(image, caption="Image from URL", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error loading image: {e}")
        
        # Image validation info
        if img_array is not None:
            is_plant, green_ratio, texture = is_rice_plant(img_array)
            
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #94a3b8; margin-bottom: 0.5rem;">🔍 Image Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Green Ratio", f"{green_ratio:.1%}")
            with col_b:
                st.metric("Texture Score", f"{texture:.0f}")
            
            if not is_plant:
                st.error("""
                ⚠️ **This doesn't appear to be a rice plant image!**
                
                The image lacks sufficient green color or plant-like texture. 
                Please upload a clear image of a rice leaf.
                """)
    
    # Right column - Results
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0; margin-bottom: 1rem;">🎯 Prediction Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if img_array is not None and selected_models:
            is_plant, _, _ = is_rice_plant(img_array)
            
            if not is_plant:
                st.warning("👆 Upload a valid rice plant image to get predictions")
            else:
                if st.button("🚀 Analyze Disease", type="primary"):
                    with st.spinner("🔬 Analyzing with AI models..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        predictions = predict_with_models(img_array, selected_models)
                        
                        all_votes = [pred['predicted_class'] for pred in predictions.values()]
                        vote_counts = {}
                        for vote in all_votes:
                            vote_counts[vote] = vote_counts.get(vote, 0) + 1
                        
                        final_prediction = max(vote_counts, key=vote_counts.get)
                        agreement = vote_counts[final_prediction] / len(selected_models)
                        
                        st.markdown("---")
                        
                        rec = get_recommendations(final_prediction)
                        
                        if final_prediction == "Healthy":
                            st.markdown(f"""
                            <div class="healthy-card animate-in">
                                <div style="font-size: 4rem; margin-bottom: 1rem;">{rec['icon']}</div>
                                <h2 style="color: #10b981; margin: 0;">PLANT IS HEALTHY</h2>
                                <p style="color: #94a3b8; margin-top: 0.5rem;">
                                    {agreement*100:.0f}% of models agree ({vote_counts[final_prediction]}/{len(selected_models)})
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="disease-alert animate-in">
                                <div style="font-size: 4rem; margin-bottom: 1rem;">{rec['icon']}</div>
                                <h2 style="color: {rec['color']}; margin: 0;">{final_prediction.upper()}</h2>
                                <p style="color: #94a3b8; margin-top: 0.5rem;">
                                    {agreement*100:.0f}% of models agree ({vote_counts[final_prediction]}/{len(selected_models)})
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### 🤝 Model Agreement")
                        
                        agree_cols = st.columns(len(vote_counts))
                        for idx, (cls, count) in enumerate(vote_counts.items()):
                            pct = count / len(selected_models)
                            with agree_cols[idx]:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <div class="metric-value" style="color: {get_recommendations(cls)['color']}">{count}</div>
                                    <div class="metric-label">{cls}</div>
                                    <div style="color: #64748b; font-size: 0.8rem;">{pct:.0%}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("### 📊 Individual Model Results")
                        
                        sorted_results = sorted(
                            predictions.items(),
                            key=lambda x: x[1]['confidence'],
                            reverse=True
                        )
                        
                        for model_name, result in sorted_results:
                            probs = result['all_probabilities']
                            top_class = result['predicted_class']
                            confidence = result['confidence']
                            
                            with st.expander(
                                f"**{model_name}**: {top_class} ({confidence*100:.1f}%)",
                                expanded=(model_name == sorted_results[0][0])
                            ):
                                for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                                    st.progress(prob, text=f"{cls}: {prob*100:.1f}%")
                        
                        st.markdown("---")
                        st.markdown("### 💡 Management Recommendations")
                        
                        rec = get_recommendations(final_prediction)
                        
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #10b981;">🛡️ Immediate Actions</h4>
                            <ul style="color: #e2e8f0;">
                                {''.join([f'<li>{item}</li>' for item in rec['management']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #3b82f6;">🔮 Prevention</h4>
                            <ul style="color: #e2e8f0;">
                                {''.join([f'<li>{item}</li>' for item in rec['prevention']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="footer">
                            Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                        """, unsafe_allow_html=True)
        
        elif not selected_models:
            st.warning("⚠️ Please select at least one model from the sidebar")
        else:
            st.info("""
            👆 **Get Started**
            
            1. Upload a rice leaf image or enter an image URL
            2. Select your preferred ML models
            3. Click **Analyze Disease** to get results
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌾 RiceGuard AI | Powered by 10 Machine Learning Models | IFT512 Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
