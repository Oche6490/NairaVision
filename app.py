
import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Nigerian Banknote Recognition",
    page_icon="🇳🇬",
    layout="centered"
)

# Improve readability
st.markdown("""
<style>
.stButton > button {
    font-size: 24px;
    min-height: 65px;
    width: 100%;
    font-weight: bold;
}
h1 {
    font-size: 36px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🇳🇬 Nigerian Banknote Recognition System")
st.write("Accessible currency recognition for visually impaired persons.")

@st.cache_resource
def load_model():
    return joblib.load("naira_model.pkl")

model = load_model()


# Store recognition history during the current session
if "recognition_history" not in st.session_state:
    st.session_state.recognition_history = []


naira_names = {
    "50": "fifty",
    "100": "one hundred",
    "200": "two hundred",
    "500": "five hundred",
    "1000": "one thousand"
}

option = st.radio(
    "Choose an input method:",
    ["Upload Image", "Use Camera"]
)

image_file = None

if option == "Upload Image":
    image_file = st.file_uploader(
        "Upload a banknote image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input(
        "Take a picture of the banknote"
    )

if image_file is not None:
    image = Image.open(image_file)

    st.image(
        image,
        caption="Banknote Image",
        use_container_width=True
    )

    if st.button("🔍 Recognize Banknote"):
        processed_image = ImageOps.fit(
            ImageOps.exif_transpose(image).convert("RGB"),
            (100, 100)
        )

        image_array = (
            np.array(processed_image).flatten() / 255.0
        ).reshape(1, -1)

        prediction = str(model.predict(image_array)[0])

        # Agreement among nearest neighbours
        neighbour_distances, neighbour_indices = (
            model.kneighbors(image_array)
        )
        neighbour_labels = model._y[
            neighbour_indices[0]
        ]

        agreement = np.mean(
            neighbour_labels == model.classes_.tolist().index(prediction)
        )

        if agreement < 0.8:
            status = "Uncertain"
            st.warning(
                "⚠️ Recognition uncertain. Please try "
                "another picture with better lighting."
            )
            voice_message = (
                "Unable to identify the banknote reliably. "
                "Please try again."
            )
        else:
            status = "Recognized"
            st.success(
                f"Predicted Banknote: ₦{prediction}"
            )
            st.caption(
                "Experimental prediction. Verify before use."
            )
            spoken_value = naira_names.get(
                prediction, prediction
            )
            voice_message = (
                f"The predicted banknote is "
                f"{spoken_value} naira. "
                "Please verify the result."
            )

        # Add the prediction to recognition history
        st.session_state.recognition_history.append({
            "Date and Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Predicted Banknote": (
                f"₦{prediction}"
                if status == "Recognized"
                else "Not identified"
            ),
            "Status": status
        })


        # Automatically announce result
        components.html(
            f"""
            <script>
            const message = new SpeechSynthesisUtterance(
                {json.dumps(voice_message)}
            );
            message.lang = "en-NG";
            message.rate = 0.85;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(message);
            </script>
            """,
            height=0
        )

# Recognition history section
st.divider()
st.subheader("📋 Banknote Recognition History")

if st.session_state.recognition_history:
    history_df = pd.DataFrame(
        st.session_state.recognition_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Recognition History",
        data=csv_data,
        file_name="banknote_recognition_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear History"):
        st.session_state.recognition_history = []
        st.rerun()
else:
    st.info("No banknotes have been recognized in this session.")
            
