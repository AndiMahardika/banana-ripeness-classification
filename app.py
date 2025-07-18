import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import base64
import io
import streamlit.components.v1 as components

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        # Pastikan path model sudah benar untuk lingkungan Anda
        return tf.keras.models.load_model('banana_ripeness_model-uas.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Pastikan file 'banana_ripeness_model-uas.h5' berada di dalam direktori './model/'.")
        return None

with st.spinner('Model is being loaded...'):
    model = load_model()

# --- CLASS NAMES ---
class_names = ['overripe', 'ripe', 'rotten', 'unripe']

# --- HELPER FUNCTIONS ---
def import_and_predict(image_data, model):
    """Preprocesses an image and makes a prediction."""
    size = (224, 224)
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.asarray(image) / 255.0
    img_reshape = image_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def pil_image_to_base64(image):
    """Converts a PIL image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- STYLING (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main app container */
    .stApp {
        background-color: white;
        color: black;
        /* Add padding to the bottom to prevent footer from overlapping content */
        padding-bottom: 5rem; 
    }

    /* Page Titles */
    .main-title { font-size: 6em; font-weight: 700; color: #FFA500; text-align: center; margin-bottom: 0.5em; }
    .subtitle { font-size: 1.2em; color: blue; text-align: center; margin-bottom: 2em; }
    .section-header { font-size: 1.5em; font-weight: 600; color: #000; text-align: center; margin-bottom: 1em; }

    /* Custom Upload Box */
    .upload-box-visual {
        border: 2px dashed #00008B; border-radius: 10px; padding: 20px; min-height: 250px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        text-align: center; margin-bottom: 20px; position: relative;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .upload-box-visual:hover { background-color: #f0f2f6; }
    .plus-icon { font-size: 5em; color: #FFD700; margin-bottom: 10px; }

    /* Button Styling */
    .stButton > button {
        background-color: #00008B; color: white; border-radius: 5px; padding: 10px 20px;
        font-weight: 600; border: none; cursor: pointer; margin-top: 10px; width: 100%;
    }
    .stButton > button:hover { background-color: #0000CD; color: white; }

    /* Result Box */
    .result-box-container {
        border: 2px solid #00008B; border-radius: 10px; padding: 20px; min-height: 250px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        text-align: center; margin-bottom: 20px;
    }
    .result-image-wrapper {
        width: 100%; display: flex; justify-content: center; align-items: center;
        overflow: hidden; border-radius: 5px; margin-bottom: 15px;
    }
    .result-image-wrapper img { height: auto; max-width: 100%; border-radius: 5px; }

    /* Status Display */
    .status-container {
        display: flex; align-items: center; justify-content: space-between;
        width: 100%; margin-top: 10px; font-size: 1.2em; font-weight: 600; color: #333;
    }
    .status-dot { width: 15px; height: 15px; border-radius: 50%; margin-right: 10px; display: inline-block; }

    /* MODIFIED: Sticky Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #00008B;
        color: white;
        text-align: center;
        padding: 20px;
        font-size: 0.9em;
        z-index: 100; /* Ensures footer stays on top */
    }

    /* Hides the default Streamlit file uploader widget */
    [data-testid="stFileUploader"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<p class='main-title' style='font-size: 3em;'>BANANA RIPENESS CLASSIFICATION</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>INSTANTLY PREDICT BANANA RIPENESS — FROM UNRIPE TO OVERRIPE OR SPOILED, ALL WITH JUST ONE CLICK!</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle' style='color: black;'>Use this app in laptop or desktop!</p>", unsafe_allow_html=True)

# --- MAIN LAYOUT (2 COLUMNS) ---
col1, col2 = st.columns(2)

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# --- COLUMN 1: UPLOAD ---
with col1:
    st.markdown("<h3 class='section-header'>UPLOAD A PICTURE OF A BANANA</h3>", unsafe_allow_html=True)
    
    # This div is the clickable area for uploading
    st.markdown("""
    <div id="upload-trigger" class="upload-box-visual">
        <div class="plus-icon">+</div>
        <p>Click here to upload a banana image (JPG, PNG, JPEG)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # The actual file uploader, hidden by CSS
    uploaded_file = st.file_uploader(
        "Upload", type=["jpg", "png", "jpeg"], 
        label_visibility="hidden", key="file_uploader_key"
    )

    # MODIFIED: Logic for automatic analysis
    # If a new file is uploaded, update session state and rerun the script to trigger analysis
    if uploaded_file is not None:
        # Check if it's a new file to avoid re-running on every minor interaction
        if st.session_state.uploaded_file is None or uploaded_file.getvalue() != st.session_state.uploaded_file.getvalue():
            st.session_state.uploaded_file = uploaded_file
            st.rerun() # Rerun to trigger the automatic analysis

    # The button now serves to re-analyze the current image
    if st.button("IMAGE ANALYST", key="image_analyst_button"):
        if st.session_state.uploaded_file is not None and model is not None:
            # Just rerun to re-trigger the analysis in col2
            st.rerun()
        elif model is None:
            st.error("Model is not loaded. Cannot perform analysis.")
        else:
            st.warning("Please upload an image first.")

# --- COLUMN 2: RESULTS ---
with col2:
    st.markdown("<h3 class='section-header'>RESULT CLASSIFICATION</h3>", unsafe_allow_html=True)
    result_placeholder = st.empty()

    # MODIFIED: Logic to display results automatically
    if st.session_state.uploaded_file is None:
        # Default message when no image is uploaded
        result_placeholder.markdown("""
        <div class='result-box-container'>
            <p>Upload an image to see classification results automatically.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # A file has been uploaded, so analyze it immediately.
        image = Image.open(st.session_state.uploaded_file)
        
        if model is not None:
            with st.spinner('Analyzing image...'):
                # Perform prediction
                predictions = import_and_predict(image, model)
                score = tf.nn.softmax(predictions[0])
                predicted_class_name = class_names[np.argmax(score)]
                
                # Determine dot color based on prediction
                dot_color_map = {'unripe': 'green', 'ripe': 'gold', 'overripe': 'orange', 'rotten': 'red'}
                dot_color = dot_color_map.get(predicted_class_name, "gray")
                
                # Convert image for display
                image_base64 = pil_image_to_base64(image)

                # Build HTML for the result box
                result_html = f"""
                <div class='result-box-container'>
                    <div class='result-image-wrapper'>
                        <img src='data:image/png;base64,{image_base64}' alt='Uploaded Banana Image'>
                    </div>
                    <div class="status-container">
                        <span>STATUS</span>
                        <span><span class="status-dot" style="background-color: {dot_color};"></span> {predicted_class_name.upper()}</span>
                    </div>
                </div>
                """
                result_placeholder.markdown(result_html, unsafe_allow_html=True)
        else:
            # Show error in result box if the model failed to load
            result_placeholder.markdown("""
            <div class='result-box-container'>
                <p style='color: red;'>Error: Model could not be loaded. Analysis is not available.</p>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
# The footer is now styled by the CSS block above to be sticky
st.markdown('<div class="footer">© 2024 BANANA RIPENESS CLASSIFICATION. ALL RIGHTS RESERVED.</div>', unsafe_allow_html=True)

# --- JAVASCRIPT TO TRIGGER FILE UPLOAD ---
# This script links the click on our custom div to the hidden file input.
components.html("""
<script>
function setupTrigger() {
    const doc = window.parent.document;
    const trigger = doc.getElementById("upload-trigger");
    const fileUploader = doc.querySelector('[data-testid="stFileUploader"]');
    
    if (trigger && fileUploader) {
        const input = fileUploader.querySelector("input[type=file]");
        if (input) {
            trigger.addEventListener("click", () => {
                input.click();
            });
        } else {
            setTimeout(setupTrigger, 250);
        }
    } else {
        setTimeout(setupTrigger, 250);
    }
}
window.addEventListener('load', setupTrigger);
</script>
""", height=0)
