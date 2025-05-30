from src.constants.constants import MODEL_PATH, FRUIT_MAP, FRUIT_RENAMED
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.api.schemas import PredictionResponse
from src.api.process import process_image
from src.utils.helper import load_model
from PIL import UnidentifiedImageError
from src import logger

router = APIRouter()

try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Classify an uploaded fruit image as fresh or rotten.

    This endpoint processes an uploaded image of a fruit (peach, pomegranate, or strawberry)
    and classifies it into one of six categories: fresh or rotten versions of each fruit type.
    The model returns both the predicted class and a confidence score.

    Args:
        file: An image file uploaded by the client (JPG, PNG, or other common image formats)

    Returns:
        PredictionResponse: JSON object containing:
            - filename: Original uploaded filename
            - prediction_class: Classified fruit category (e.g., "FreshPeaches", "RottenStrawberries")
            - prediction: Confidence score

    Raises:
        HTTPException(400): If the uploaded file is not a valid image
        HTTPException(500): If model prediction fails for any reason
        HTTPException(503): If the classification model is not available

    Example:
        Upload a fruit image to receive a classification like:
        {"filename": "strawberry.jpg", "prediction_class": "FreshStrawberries", "prediction": 0.9876}
    """
    try:
        image_content = await file.read()

        processed_image = process_image(image_content)

        prediction = round(model.predict(processed_image)[0][0], 4)

        prediction_class = FRUIT_RENAMED[FRUIT_MAP[round(prediction)]]

        response = PredictionResponse(
            FileName=file.filename,
            PredictionClass=prediction_class,
            prediction=prediction,
        )

        logger.info(f"Successfully processed prediction for {file.filename}")
        return response

    except UnidentifiedImageError as e:
        logger.error(f"Invalid image format: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
