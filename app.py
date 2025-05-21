import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import io
import base64
import os


# Styling and theme setup
def set_professional_theme():
    # White background with subtle pale green gradient and professional styling
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #f5f9f5 50%, #f0f8f0 100%);
            background-size: cover;
        }

        /* Subtle animated gradient background */
        @keyframes gradientAnimation {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .gradient-bg {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(-45deg, #ffffff, #f8fcf8, #f0f7f0, #e8f5e8);
            background-size: 400% 400%;
            animation: gradientAnimation 15s ease infinite;
            z-index: -1;
        }

        /* Professional card styling */
        .css-1kyxreq, .css-1r6slb0, .css-12w0qpk {
            border-radius: 8px !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
            background-color: rgba(255, 255, 255, 0.95) !important;
        }

        /* Professional button styling */
        .stButton button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-weight: 500 !important;
            border-radius: 4px !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.2s ease !important;
        }

        .stButton button:hover {
            background-color: #43a047 !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15) !important;
        }

        /* Header styling - IMPROVED */
        h1, h2, h3 {
            color: #2E7D32 !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            font-weight: 600 !important;
            text-align: center !important;
        }

        h1 {
            font-size: 2.5rem !important;
            margin-bottom: 1.5rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            background: linear-gradient(120deg, #2E7D32, #388E3C, #43A047);
            -webkit-background-clip: text !important;
            padding: 10px 0 !important;
            text-shadow: 0px 0px 15px rgba(46, 125, 50, 0.15) !important;
        }

        /* Main title banner */
        .main-title {
            text-align: center !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
            padding: 20px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
            margin-bottom: 30px !important;
            border-bottom: 3px solid #4CAF50 !important;
        }

        .subtitle {
            text-align: center !important;
            color: #4d4d4d !important;
            font-size: 1.2rem !important;
            margin-bottom: 25px !important;
            max-width: 80% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            line-height: 1.6 !important;
        }

        .logo-text {
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 3px !important;
            color: #388E3C !important;
            margin-bottom: 5px !important;
            font-weight: 700 !important;
        }

        h2 {
            font-size: 1.7rem !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            border-bottom: 2px solid rgba(76, 175, 80, 0.2) !important;
            padding-bottom: 5px !important;
        }

        h3 {
            font-size: 1.3rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* Upload box styling */
        .uploadedFileData {
            background-color: rgba(255, 255, 255, 0.7) !important;
            border-radius: 4px !important;
            padding: 8px !important;
            border: 1px dashed rgba(76, 175, 80, 0.3) !important;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }

        .stTabs [data-baseweb="tab"] {
            background-color: rgba(240, 248, 240, 0.6) !important;
            border-radius: 4px 4px 0 0 !important;
            padding: 8px 16px !important;
            border: none !important;
            font-weight: 500 !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-bottom: 2px solid #4CAF50 !important;
        }

        /* Info panels */
        .info-panel {
            background-color: rgba(255, 255, 255, 0.9);
            border-left: 4px solid #4CAF50;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        /* Footer styling */
        .footer {
            margin-top: 30px;
            padding: 15px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 4px;
            border-top: 1px solid rgba(0, 0, 0, 0.05);
            text-align: center;
            font-size: 0.9rem;
            color: #555;
        }
        </style>

        <div class="gradient-bg"></div>
        """,
        unsafe_allow_html=True
    )


# Preprocessing functions
def normalize_batch(images):
    """Performs channel-wise z-score normalization"""
    return (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])


def denormalize_batch(images, should_clip=True):
    """Denormalize the images for prediction"""
    images = (images * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    if should_clip:
        images = np.clip(images, 0, 1)
    return images


# Image to base64 for display
def get_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# Hide operation function
def hide_image(cover_image, secret_image, model_path):
    model = load_model(model_path)

    # Convert to RGB and resize to 224x224
    secret_image = secret_image.convert('RGB')
    cover_image = cover_image.convert('RGB')

    # Resize images to 224x224
    if secret_image.size != (224, 224):
        secret_image = secret_image.resize((224, 224))
    if cover_image.size != (224, 224):
        cover_image = cover_image.resize((224, 224))

    # Convert to numpy arrays and normalize
    secret_image_in = np.array(secret_image).reshape(1, 224, 224, 3) / 255.0
    cover_image_in = np.array(cover_image).reshape(1, 224, 224, 3) / 255.0

    # Make prediction
    steg_image_out = model.predict([normalize_batch(secret_image_in), normalize_batch(cover_image_in)])
    steg_image_out = denormalize_batch(steg_image_out)
    steg_image_out = np.squeeze(steg_image_out) * 255.0
    steg_image_out = np.uint8(steg_image_out)

    # Convert back to PIL Image
    steg_pil = Image.fromarray(steg_image_out)
    return steg_pil


# Reveal operation function
def reveal_image(stego_image, model_path):
    model = load_model(model_path, compile=False)

    # Convert to RGB and resize to 224x224
    stego_image = stego_image.convert('RGB')

    # Resize the image to 224x224
    if stego_image.size != (224, 224):
        stego_image = stego_image.resize((224, 224))

    # Convert to numpy arrays and normalize
    stego_image_in = np.array(stego_image).reshape(1, 224, 224, 3) / 255.0

    # Make prediction
    secret_image_out = model.predict([normalize_batch(stego_image_in)])
    secret_image_out = denormalize_batch(secret_image_out)
    secret_image_out = np.squeeze(secret_image_out) * 255.0
    secret_image_out = np.uint8(secret_image_out)

    # Convert back to PIL Image
    secret_pil = Image.fromarray(secret_image_out)
    return secret_pil


# Main app
def main():
    set_professional_theme()

    # Updated centered and attractive header
    st.markdown("""
    <div class="main-title">
        <p class="logo-text">Deep learning based invisible watermarking system </p>
        <h1>INVISIBLE WATERMARK</h1>
        <p class="subtitle">Secure your valuable images with  invisible watermarks that protect your intellectual property while remaining completely undetectable to the human eye.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
        <p style='color: #2E7D32; font-size: 16px; margin-bottom: 5px; font-weight: 500;'>
            Image Steganography
        </p>
        <p style='color: #555; font-size: 14px; margin: 0;'>
            Securely hide confidential images within ordinary images using advanced deep learning technology.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Paths to models - Configure these to your model paths
    hide_model_path = "/teamspace/studios/this_studio/models/hide.h5"
    reveal_model_path = "/teamspace/studios/this_studio/models/reveal.h5"

    # Create tabs
    tab1, tab2 = st.tabs(["Hide Image", "Reveal Image"])

    # Hide Image Tab
    with tab1:
        st.header("Hide a Secret Image")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload Cover Image")
            st.markdown("This is the visible image that will hide your secret.")
            cover_file = st.file_uploader("Choose a cover image", type=["jpg", "jpeg", "png"], key="cover")
            if cover_file is not None:
                cover_image = Image.open(cover_file)
                st.image(cover_image, caption="Cover Image", use_column_width=True)

        with col2:
            st.subheader("Upload Secret Image")
            st.markdown("This is the image you want to hide.")
            secret_file = st.file_uploader("Choose a secret image", type=["jpg", "jpeg", "png"], key="secret")
            if secret_file is not None:
                secret_image = Image.open(secret_file)
                st.image(secret_image, caption="Secret Image", use_column_width=True)

        if st.button("Process", key="hide_button"):
            if cover_file is not None and secret_file is not None:
                with st.spinner("Processing... Please wait."):
                    try:
                        # Perform hiding operation
                        steg_image = hide_image(cover_image, secret_image, hide_model_path)

                        # Save the steganographic image
                        output_filepath = os.path.join("steg_image.png")
                        steg_image.save(output_filepath)

                        # Display the result
                        st.success(f"Success! Steganographic image created and saved to {output_filepath}")
                        st.image(steg_image, caption="Steganographic Image", use_column_width=True)

                        # Option to download
                        img_base64 = get_image_base64(steg_image)
                        href = f'<a href="data:image/png;base64,{img_base64}" download="steg_image.png" style="background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block; margin-top: 10px; font-size: 14px;">Download Image</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please upload both cover and secret images first.")

    # Reveal Image Tab
    with tab2:
        st.header("Reveal a Hidden Image")

        st.subheader("Upload Steganographic Image")
        st.markdown("Upload an image that contains a hidden secret.")
        stego_file = st.file_uploader("Choose a steganographic image", type=["jpg", "jpeg", "png"], key="stego")
        if stego_file is not None:
            stego_image = Image.open(stego_file)
            st.image(stego_image, caption="Steganographic Image", use_column_width=True)

        if st.button("Process", key="reveal_button"):
            if stego_file is not None:
                with st.spinner("Analyzing image..."):
                    try:
                        # Perform revealing operation
                        secret_image = reveal_image(stego_image, reveal_model_path)

                        # Save the revealed image
                        output_filepath = os.path.join("secret_out.png")
                        secret_image.save(output_filepath)

                        # Display the result
                        st.success(f"Success! Secret image revealed and saved to {output_filepath}")
                        st.image(secret_image, caption="Revealed Secret Image", use_column_width=True)

                        # Option to download
                        img_base64 = get_image_base64(secret_image)
                        href = f'<a href="data:image/png;base64,{img_base64}" download="secret_out.png" style="background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block; margin-top: 10px; font-size: 14px;">Download Image</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please upload a steganographic image first.")

    # Footer
    st.markdown("""
    <div class="footer">
        <h3 style='color: #2E7D32; margin-bottom: 10px; font-size: 16px;'>About Steganography Technology</h3>
        <p style='margin-bottom: 5px;'>This application utilizes deep neural networks to conceal images within other images with minimal visual distortion.</p>
        <p>The concealed image can be later extracted using our proprietary neural network technology.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()