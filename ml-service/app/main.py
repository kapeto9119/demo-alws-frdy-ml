import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routers import image_router
from app.utils.logger import logger
from app.utils.config import settings

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Machine Learning Service",
    description="An API for face detection in images",
    version="1.0.0",
    redirect_slashes=False  # Disable automatic redirects
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(image_router.router)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ML service is healthy"}