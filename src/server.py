from src.schemas import PredictionRequest, PredictionResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.model import Model
from PIL import Image

import base64
import io


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Manage the lifecycle of the FastAPI application.

    Initialize the colorization model during application startup and
    release the application state reference during shutdown.

    Parameters
    ----------
    app : fastapi.FastAPI
        FastAPI application instance whose lifecycle is being managed.

    Yields
    ------
    None
        Yields control after the model has been initialized, allowing
        the application to serve requests until shutdown.
    """
    app.state.model = Model()

    yield

    del app.state.model


app = FastAPI(title="Image Colorizer API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Return the basic health status of the API.

    This endpoint indicates that the health-check handler is reachable.
    It does not verify whether the colorization model has been loaded.

    Returns
    -------
    dict
        Dictionary containing the status key with the value "ok".
    """
    return {
        "status": "ok",
    }


@app.get("/ready")
def ready() -> dict:
    """Check whether the colorization model has been initialized.

    Verify that the application state contains a model instance.

    Returns
    -------
    dict
        Dictionary containing the status key with the value
        "ready" when the model is available.

    Raises
    ------
    fastapi.HTTPException
        If the model has not been loaded. The exception has HTTP status
        code 503 and the detail message "Model not loaded".
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {"status": "ready"}


@app.post("/colorize", response_model=PredictionResponse)
def colorize(request: PredictionRequest) -> PredictionResponse:
    """Colorize an image submitted through the REST API.

    Decode the Base64-encoded input image, convert it to RGB, run the
    colorization model, serialize the resulting image as PNG, and encode
    the output as Base64.

    Parameters
    ----------
    request : PredictionRequest
        Validated request containing the Base64-encoded input image.

    Returns
    -------
    PredictionResponse
        Response containing the Base64-encoded colorized PNG image.

    Raises
    ------
    fastapi.HTTPException
        If the input string is empty or contains only whitespace.
        In this case, HTTP status code 400 is returned.
    binascii.Error
        If the input contains invalid Base64 data.
    PIL.UnidentifiedImageError
        If the decoded data cannot be identified as a supported image.
    """
    if not request.image_encoded.strip():
        raise HTTPException(status_code=400, detail="No image!")

    # decode image into PIL image
    image_bytes = base64.b64decode(request.image_encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # send to model
    model = app.state.model
    colorized_image = model.colorize(image)

    # convert PIL Image to bytes
    buffer = io.BytesIO()
    colorized_image.save(buffer, format="PNG")
    colorized_image_bytes = buffer.getvalue()

    colorized_image_encoded = base64.b64encode(colorized_image_bytes).decode("utf-8")

    return PredictionResponse(colorized_image_encoded=colorized_image_encoded)
