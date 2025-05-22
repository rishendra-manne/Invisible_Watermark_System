import streamlit as st
import requests
import base64
from PIL import Image
import io
import numpy as np
from typing import Optional

# Set page config
st.set_page_config(
    page_title="Image Watermarking System",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


class WatermarkingAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def check_health(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def hide_watermark(self, cover_image: bytes, watermark_image: bytes) -> Optional[str]:
        """Hide watermark in cover image"""
        try:
            files = {
                'cover_image': ('cover.png', cover_image, 'image/png'),
                'watermark_image': ('watermark.png', watermark_image, 'image/png')
            }

            response = requests.post(
                f"{self.base_url}/hide-watermark",
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('encoded_image')
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None

    def reveal_watermark(self, watermarked_image: bytes) -> Optional[str]:
        """Reveal watermark from watermarked image"""
        try:
            files = {
                'watermarked_image': ('watermarked.png', watermarked_image, 'image/png')
            }

            response = requests.post(
                f"{self.base_url}/reveal-watermark",
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('revealed_watermark')
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to bytes"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]

    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


def main():
    # Apply theme
    set_professional_theme()

    # Initialize API client
    api = WatermarkingAPI()

    # Header
    st.markdown("""
    <div class="main-title">
        <div class="logo-text">Advanced AI</div>
        <h1>Image Watermarking System</h1>
        <div class="subtitle">
            Securely hide and reveal watermarks in images using deep learning technology
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API Status Check
    with st.spinner("Checking API connection..."):
        api_healthy = api.check_health()

    if not api_healthy:
        st.error("⚠️ API Server is not running. Please start the FastAPI server first.")
        st.code("uvicorn main:app --reload", language="bash")
        return
    else:
        st.success("✅ API Server is running")

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        api_url = st.text_input("API URL", value="http://localhost:8000")
        if api_url != api.base_url:
            api.base_url = api_url

        st.markdown("### 📋 Instructions")
        st.markdown("""
        **Hide Watermark:**
        1. Upload a cover image
        2. Upload a watermark image
        3. Click 'Hide Watermark'

        **Reveal Watermark:**
        1. Upload a watermarked image
        2. Click 'Reveal Watermark'
        """)

        st.markdown("### ℹ️ Tips")
        st.info("For best results, use images of similar sizes and high quality.")

    # Main content
    tab1, tab2 = st.tabs(["🔒 Hide Watermark", "🔓 Reveal Watermark"])

    with tab1:
        st.markdown("## Hide Watermark in Image")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cover Image")
            cover_file = st.file_uploader(
                "Upload cover image",
                type=['png', 'jpg', 'jpeg'],
                key="cover_upload"
            )

            if cover_file:
                cover_image = Image.open(cover_file)
                st.image(cover_image, caption="Cover Image", use_column_width=True)

        with col2:
            st.markdown("### Watermark Image")
            watermark_file = st.file_uploader(
                "Upload watermark image",
                type=['png', 'jpg', 'jpeg'],
                key="watermark_upload"
            )

            if watermark_file:
                watermark_image = Image.open(watermark_file)
                st.image(watermark_image, caption="Watermark Image", use_column_width=True)

        # Process button
        if st.button("🔒 Hide Watermark", type="primary", use_container_width=True):
            if cover_file and watermark_file:
                with st.spinner("Hiding watermark... This may take a few moments."):
                    cover_bytes = image_to_bytes(cover_image)
                    watermark_bytes = image_to_bytes(watermark_image)

                    result = api.hide_watermark(cover_bytes, watermark_bytes)

                    if result:
                        st.success("✅ Watermark hidden successfully!")

                        # Display result
                        result_image = base64_to_image(result)

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown("### 🖼️ Watermarked Image")
                            st.image(result_image, caption="Watermarked Image", use_column_width=True)

                            # Download button
                            result_bytes = image_to_bytes(result_image)
                            st.download_button(
                                label="📥 Download Watermarked Image",
                                data=result_bytes,
                                file_name="watermarked_image.png",
                                mime="image/png",
                                use_container_width=True
                            )
            else:
                st.warning("⚠️ Please upload both cover and watermark images.")

    with tab2:
        st.markdown("## Reveal Watermark from Image")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("### Watermarked Image")
            watermarked_file = st.file_uploader(
                "Upload watermarked image",
                type=['png', 'jpg', 'jpeg'],
                key="watermarked_upload"
            )

            if watermarked_file:
                watermarked_image = Image.open(watermarked_file)
                st.image(watermarked_image, caption="Watermarked Image", use_column_width=True)

        # Process button
        if st.button("🔓 Reveal Watermark", type="primary", use_container_width=True):
            if watermarked_file:
                with st.spinner("Revealing watermark... This may take a few moments."):
                    watermarked_bytes = image_to_bytes(watermarked_image)

                    result = api.reveal_watermark(watermarked_bytes)

                    if result:
                        st.success("✅ Watermark revealed successfully!")

                        # Display result
                        revealed_image = base64_to_image(result)

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown("### 🔍 Revealed Watermark")
                            st.image(revealed_image, caption="Revealed Watermark", use_column_width=True)

                            # Download button
                            result_bytes = image_to_bytes(revealed_image)
                            st.download_button(
                                label="📥 Download Revealed Watermark",
                                data=result_bytes,
                                file_name="revealed_watermark.png",
                                mime="image/png",
                                use_container_width=True
                            )
            else:
                st.warning("⚠️ Please upload a watermarked image.")

    # Footer
    st.markdown("""
    <div class="footer">
        <strong>Image Watermarking System</strong> | 
        Powered by Deep Learning & FastAPI | 
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()