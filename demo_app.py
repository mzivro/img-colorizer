from src.model import Model
from io import BytesIO
from PIL import Image

import streamlit as st

st.title("Image Colorizer - Demo")

# init model
if "model" not in st.session_state:
    st.session_state.model = Model()

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
                colorized_image = st.session_state.model.colorize(image)

                if not isinstance(colorized_image, Image.Image):
                    colorized_image = Image.fromarray(colorized_image)

                colorized_image = colorized_image.convert("RGB")

                st.session_state.colorized_image = colorized_image
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
