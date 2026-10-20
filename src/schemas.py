from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Schema for image colorization requests.

    The request contains an image encoded as a Base64 string, typically
    serialized as part of a JSON payload sent to the API.

    Attributes
    ----------
    image_encoded : str
        Base64-encoded representation of the input image.
    """

    image_encoded: str


class PredictionResponse(BaseModel):
    """Schema for image colorization responses.

    The response contains the colorized image encoded as a Base64 string,
    allowing the generated image to be transmitted in a JSON payload.

    Attributes
    ----------
    colorized_image_encoded : str
        Base64-encoded representation of the colorized image, serialized
        in PNG format by the API endpoint.
    """

    colorized_image_encoded: str
