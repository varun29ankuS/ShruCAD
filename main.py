# main.py
# This is the core backend server for our Forge AI prototype.

# Import necessary libraries
import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Initialize the FastAPI Application
app = FastAPI(
    title="Forge AI Backend",
    description="A prototype backend to convert 2D drawings to 3D models.",
    version="0.1.0",
)

# Define a temporary directory for uploads
TEMP_UPLOAD_DIR = "temp_uploads"

@app.on_event("startup")
async def startup_event():
    """Create the temporary directory when the server starts."""
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    print(f"Server startup complete. Temporary directory '{TEMP_UPLOAD_DIR}' is ready.")

# Create the API Endpoint for /generate
@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    """
    This endpoint receives an image file from the user, saves it,
    and returns a success message.
    """
    try:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Successfully received and saved file: {file.filename}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": file.filename,
                "message": "File uploaded successfully. Ready for next steps.",
            },
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )

# This block allows running the server directly for local development
if __name__ == "__main__":
    print("Starting Forge AI server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)