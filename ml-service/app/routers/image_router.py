from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from app.services.face_detection import detect_faces_in_image
from app.utils.logger import logger

router = APIRouter(
    prefix="/process",
    tags=["Image Processing"],
)

@router.post(
    "/",
    summary="Process an image and detect faces",
    description="Uploads an image and returns the image with detected faces highlighted.",
    responses={
        200: {"content": {"image/jpeg": {}}},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def process_image(
    image: UploadFile = File(...),
    scale_factor: float = Form(1.1),
    min_neighbors: int = Form(4),
):
    try:
        # Input validation
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        if not (1.0 <= scale_factor <= 2.0):
            raise HTTPException(
                status_code=400,
                detail="scale_factor must be between 1.0 and 2.0",
            )

        if not (1 <= min_neighbors <= 10):
            raise HTTPException(
                status_code=400,
                detail="min_neighbors must be between 1 and 10",
            )

        contents = await image.read()
        processed_image = detect_faces_in_image(contents, scale_factor, min_neighbors)

        if processed_image is None:
            raise HTTPException(status_code=500, detail="Error processing image")

        return Response(content=processed_image, media_type="image/jpeg")
    except HTTPException as http_err:
        logger.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.exception("An unexpected error occurred")
        raise HTTPException(status_code=500, detail="Internal server error")