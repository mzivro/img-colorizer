from model import Model
from io import BytesIO
from PIL import Image

import streamlit as st
import requests
import base64

API_URL = "http://localhost:8000/colorize"

st.title("Image Colorizer - Client")

# loading image
uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
)

# colorize image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original image")
    st.image(image, width="stretch")

    if st.button("Colorize", width="stretch"):
        with st.spinner("Colorizing image..."):
            try:
                # encode image into base64
                image_bytes = uploaded_file.getvalue()
                image_encoded = base64.b64encode(image_bytes).decode("utf-8")

                payload = {
                    "image_encoded": image_encoded,
                }

                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    colorized_image_encoded = response.json()["colorized_image_encoded"]

                    colorized_image_bytes = base64.b64decode(colorized_image_encoded)
                    colorized_image = Image.open(
                        BytesIO(colorized_image_bytes)
                    ).convert("RGB")

                    st.session_state.colorized_image = colorized_image
                else:
                    st.error(response.text)
            except Exception as e:
                st.error(f"Colorization failed: {e}")

    # display image
    if st.session_state.get("colorized_image") is not None:
        result = st.session_state.colorized_image

        st.subheader("Colorized image")
        st.image(result, width="stretch")

        # prepare image to download
        buffer = BytesIO()
        result.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="Download colorized image",
            data=buffer.getvalue(),
            file_name="colorized_image.png",
            mime="image/png",
            width="stretch",
        )
else:
    st.info("Upload an image to get started.")
