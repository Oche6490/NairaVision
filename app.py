
import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd
import json
import hashlib
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps

# ---------------------------------
# PAGE SETTINGS
# ---------------------------------
st.set_page_config(
    page_title="NairaVision | Banknote Recognition",
    page_icon="🇳🇬",
    layout="centered"
)

# ---------------------------------
# GREEN AND WHITE DESIGN
# ---------------------------------
st.markdown("""
<style>
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #075E35 !important;
    }

    .hero {
        background: linear-gradient(135deg, #006B3C, #13965B);
        padding: 35px 25px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        text-align: center;
    }

    .hero h1 {
        color: white !important;
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #E7FFF1;
        font-size: 17px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #087B46;
        color: white;
        border-radius: 12px;
        font-size: 21px;
        font-weight: bold;
        min-height: 65px;
        width: 100%;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #065D35;
        color: white;
    }

    .result-card {
        background: #E5F5EB;
        border: 2px solid #087B46;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 15px;
    }

    .result-card h2 {
        font-size: 45px;
        color: #006B3C !important;
    }

    .footer {
        text-align: center;
        color: #52715E;
        padding: 25px 5px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD MACHINE LEARNING MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "naira_model.pkl"
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load recognition model: {error}")
    st.stop()

# ---------------------------------
# INITIALIZE SESSION HISTORY
# ---------------------------------
if "recognition_history" not in st.session_state:
    st.session_state.recognition_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "result_number" not in st.session_state:
    st.session_state.result_number = 0

# ---------------------------------
# DENOMINATION NAMES
# ---------------------------------
naira_names = {
    "50": "fifty",
    "100": "one hundred",
    "200": "two hundred",
    "500": "five hundred",
    "1000": "one thousand"
}

# ---------------------------------
# APPLICATION HEADER
# ---------------------------------
st.markdown("""
<div class="hero">
    <h1>🇳🇬 NairaVision</h1>
    <p>Automated Nigerian Banknote Recognition System</p>
    <p>Helping visually impaired persons identify Nigerian currency</p>
</div>
""", unsafe_allow_html=True)

st.subheader("💵 Recognize a Nigerian Banknote")

st.write(
    "Upload a banknote image or use your camera. "
    "NairaVision will attempt to identify the denomination "
    "and announce the result."
)

# ---------------------------------
# INPUT METHOD
# ---------------------------------
option = st.radio(
    "Choose your input method:",
    ["📁 Upload Image", "📷 Use Camera"],
    horizontal=True
)

image_file = None

if option == "📁 Upload Image":
    image_file = st.file_uploader(
        "Upload a Nigerian banknote picture",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input(
        "Take a clear photograph of your banknote"
    )

# ---------------------------------
# IMAGE RECOGNITION
# ---------------------------------
if image_file is not None:
    try:
        image = Image.open(image_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Banknote image preview",
            use_container_width=True
        )

        if st.button(
            "🔍 Recognize Banknote",
            type="primary",
            use_container_width=True
        ):

            processed_image = ImageOps.fit(
                image,
                (100, 100)
            )

            image_array = (
                np.asarray(
                    processed_image,
                    dtype=np.float32
                ).flatten() / 255.0
            ).reshape(1, -1)

            prediction = str(
                model.predict(image_array)[0]
            )

            # Measure nearest-neighbour agreement.
            # This is NOT a calibrated confidence score.
            _, neighbour_indices = model.kneighbors(
                image_array
            )

            neighbour_labels = model._y[
                neighbour_indices[0]
            ]

            predicted_index = list(
                model.classes_
            ).index(prediction)

            agreement = float(
                np.mean(
                    neighbour_labels == predicted_index
                )
            )

            if agreement < 0.8:
                status = "Uncertain"
                displayed_value = "Not identified"

                voice_message = (
                    "Unable to identify the banknote reliably. "
                    "Please take another picture."
                )
            else:
                status = "Recognized"
                displayed_value = f"₦{prediction}"

                spoken_value = naira_names.get(
                    prediction,
                    prediction
                )

                voice_message = (
                    f"The predicted banknote is "
                    f"{spoken_value} naira. "
                    "Please verify the result."
                )

            # Save result for display
            st.session_state.result_number += 1

            st.session_state.last_result = {
                "status": status,
                "value": displayed_value,
                "voice": voice_message,
                "number": st.session_state.result_number
            }

            # Record recognition history
            st.session_state.recognition_history.append({
                "Date and Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Predicted Banknote": displayed_value,
                "Status": status
            })

    except Exception as error:
        st.error(
            f"Image processing failed: {error}"
        )

# ---------------------------------
# RECOGNITION RESULT AND VOICE
# ---------------------------------
if st.session_state.last_result is not None:
    result = st.session_state.last_result

    st.divider()
    st.subheader("✅ Recognition Result")

    if result["status"] == "Recognized":
        st.markdown(
            f"""
            <div class="result-card">
                <p>Predicted Banknote</p>
                <h2>{result["value"]}</h2>
                <p>Recognition completed</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(
            "Experimental prediction. "
            "Please verify the denomination before use."
        )
    else:
        st.warning(
            "⚠️ Recognition uncertain. "
            "Please try another photograph with better lighting."
        )

    voice_message = result["voice"]

    # The unique key permits a new announcement
    # after each recognition attempt.
    components.html(
        f"""
        <button onclick="speakResult()" style="
            background: #087B46;
            color: white;
            padding: 12px 18px;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
        ">
            🔊 Hear Result Again
        </button>

        <script>
            function speakResult() {{
                const speech = new SpeechSynthesisUtterance(
                    {json.dumps(voice_message)}
                );
                speech.lang = "en-NG";
                speech.rate = 0.85;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(speech);
            }}

            speakResult();
        </script>
        """,
        height=80,
        )

# ---------------------------------
# RECOGNITION HISTORY
# ---------------------------------
st.divider()
st.subheader("📋 Recognition History")

history = st.session_state.recognition_history

if history:
    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 Download Recognition History",
        data=csv_data,
        file_name="nairavision_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):
        st.session_state.recognition_history = []
        st.session_state.last_result = None
        st.rerun()

else:
    st.info(
        "No banknotes have been recognized in this session."
    )

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("""
<div class="footer">
    🇳🇬 NairaVision — Nigerian Banknote Recognition System
    <p>Developed for academic research and accessibility.</p>
    <p>Prototype system — predictions should be verified.</p>
</div>
""", unsafe_allow_html=True)
