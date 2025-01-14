import streamlit as st
from PIL import Image
import io
import base64
from utils.app import ImageColorizer
import torch


def get_image_download_link(img, filename, text):
    """Generate a link to download an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def colorize(grayscale_image, colorizer):
    return colorizer.colorize(grayscale_image)[1]


def initialize_colorizer():
    """Initialize the ImageColorizer if it doesn't exist in session state"""
    if "colorizer" not in st.session_state:
        st.session_state.colorizer = ImageColorizer("results/experiment_2/final.pt", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def main():
    st.set_page_config(page_title="Image Colorization App", layout="wide")
    with st.spinner("Loading model..."):
        initialize_colorizer()

    # Custom CSS for styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
            margin-bottom: 1rem;
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem;
        }
        .download-link {
            text-align: center;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🎨 Image Colorization App")
    st.write("Upload a grayscale image and watch it come to life with colors!")

    # File uploader
    uploaded_file = st.file_uploader("Choose a grayscale image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        # Colorize button
        if st.button("🎨 Colorize Image"):
            with col2:
                st.subheader("Colorized Result")
                # Process the image
                colorized_image = colorize(image, st.session_state.get("colorizer"))
                st.image(colorized_image, use_container_width=True)

                # Download button
                st.markdown("<div class='download-link'>", unsafe_allow_html=True)
                st.markdown(
                    get_image_download_link(
                        colorized_image,
                        "colorized_image.png",
                        "📥 Download Colorized Image"
                    ),
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
