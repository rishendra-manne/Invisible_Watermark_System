from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import List
import uvicorn

# Import your prediction pipeline
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.exception import CustomException
from src.logger import logging

app = FastAPI(
    title="Image Watermarking API",
    description="API for hiding and revealing watermarks in images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction pipeline
try:
    pipeline = PredictionPipeline()
    logging.info("Prediction pipeline initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize prediction pipeline: {e}")
    pipeline = None


def process_image(uploaded_file: UploadFile) -> np.ndarray:
    """Process uploaded file and convert to numpy array"""
    try:
        # Read image file
        image_data = uploaded_file.file.read()

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    try:
        # Remove batch dimension if present
        if len(image_array.shape) == 4:
            image_array = image_array[0]

        # Ensure values are in 0-255 range
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(image_array)

        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting image: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Image Watermarking API is running!"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/hide-watermark")
async def hide_watermark(
        cover_image: UploadFile = File(..., description="Cover image file"),
        watermark_image: UploadFile = File(..., description="Watermark image file")
):
    """Hide watermark in cover image"""

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction pipeline not available")

    try:
        # Validate file types
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if cover_image.content_type not in allowed_types or watermark_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

        # Process images
        cover_array = process_image(cover_image)
        watermark_array = process_image(watermark_image)

        # Ensure images have same dimensions
        if cover_array.shape[1:3] != watermark_array.shape[1:3]:
            # Resize watermark to match cover image
            h, w = cover_array.shape[1:3]
            watermark_resized = cv2.resize(watermark_array[0], (w, h))
            watermark_array = np.expand_dims(watermark_resized, axis=0)

        # Hide watermark
        encoded_image = pipeline.hide_watermark(cover_array, watermark_array)

        # Convert result to base64
        result_base64 = numpy_to_base64(encoded_image)

        return JSONResponse({
            "success": True,
            "message": "Watermark hidden successfully",
            "encoded_image": f"data:image/png;base64,{result_base64}"
        })

    except CustomException as e:
        logging.error(f"Custom exception in hide_watermark: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    except Exception as e:
        logging.error(f"Unexpected error in hide_watermark: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/reveal-watermark")
async def reveal_watermark(
        watermarked_image: UploadFile = File(..., description="Watermarked image file")
):
    """Reveal watermark from watermarked image"""

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction pipeline not available")

    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if watermarked_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

        # Process image
        watermarked_array = process_image(watermarked_image)

        # Reveal watermark
        revealed_watermark = pipeline.reveal_watermark(watermarked_array)

        # Convert result to base64
        result_base64 = numpy_to_base64(revealed_watermark)

        return JSONResponse({
            "success": True,
            "message": "Watermark revealed successfully",
            "revealed_watermark": f"data:image/png;base64,{result_base64}"
        })

    except CustomException as e:
        logging.error(f"Custom exception in reveal_watermark: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    except Exception as e:
        logging.error(f"Unexpected error in reveal_watermark: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/batch-hide-watermark")
async def batch_hide_watermark(
        cover_images: List[UploadFile] = File(..., description="Multiple cover image files"),
        watermark_images: List[UploadFile] = File(..., description="Multiple watermark image files")
):
    """Hide watermarks in multiple images"""

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction pipeline not available")

    if len(cover_images) != len(watermark_images):
        raise HTTPException(status_code=400, detail="Number of cover images must match number of watermark images")

    try:
        results = []

        for i, (cover_img, watermark_img) in enumerate(zip(cover_images, watermark_images)):
            try:
                # Process images
                cover_array = process_image(cover_img)
                watermark_array = process_image(watermark_img)

                # Ensure images have same dimensions
                if cover_array.shape[1:3] != watermark_array.shape[1:3]:
                    h, w = cover_array.shape[1:3]
                    watermark_resized = cv2.resize(watermark_array[0], (w, h))
                    watermark_array = np.expand_dims(watermark_resized, axis=0)

                # Hide watermark
                encoded_image = pipeline.hide_watermark(cover_array, watermark_array)

                # Convert result to base64
                result_base64 = numpy_to_base64(encoded_image)

                results.append({
                    "index": i,
                    "success": True,
                    "encoded_image": f"data:image/png;base64,{result_base64}",
                    "cover_filename": cover_img.filename,
                    "watermark_filename": watermark_img.filename
                })

            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "cover_filename": cover_img.filename,
                    "watermark_filename": watermark_img.filename
                })

        return JSONResponse({
            "success": True,
            "message": f"Processed {len(results)} image pairs",
            "results": results
        })

    except Exception as e:
        logging.error(f"Unexpected error in batch_hide_watermark: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )