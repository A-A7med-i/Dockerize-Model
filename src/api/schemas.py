from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """
    Response model for image classification results.

    This model represents the prediction output for a multi-class image classifier
    that categorizes images into one of six possible classes.

    Attributes:
        FileName: Name of the uploaded image file
        PredictionClass: Classification result (one of six possible classes)
        prediction: Confidence score representing prediction probability
    """

    FileName: str
    PredictionClass: str
    prediction: float
