# main.py
# This is the complete backend server code for Day 2.
# It includes the crucial CORS middleware to allow the frontend to connect.

# --- 1. Imports ---
# Import all the necessary libraries
import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- 2. FastAPI App Initialization ---
# Create the main application instance
app = FastAPI(
    title="Forge AI Backend",
    description="A prototype backend to convert 2D drawings to 3D models.",
    version="0.1.0",
)

# --- 3. CORS Middleware Configuration (The "Guest List") ---
# This section is the fix for the browser's NetworkError.
# It tells the server which frontend URLs are allowed to make requests.
origins = [
    "http://localhost",
    "http://localhost:5173",  # The default address for the Vite React dev server
    "http://localhost:3000",  # A common address for Create React App
    # IMPORTANT: Later, when you deploy your frontend, you will add its URL here.
    # For example: "https://forge-ai-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Allow requests from the origins list
    allow_credentials=True,     # Allow cookies to be included in requests
    allow_methods=["*"],        # Allow all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],        # Allow all headers
)

# --- 4. Temporary Directory Setup ---
# Define a directory to store user uploads temporarily
TEMP_UPLOAD_DIR = "temp_uploads"

@app.on_event("startup")
async def startup_event():
    """This function runs when the server starts. It creates the upload directory."""
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    print(f"Server startup complete. Temporary directory '{TEMP_UPLOAD_DIR}' is ready.")


# --- 5. The API Endpoint ---
# This is the main endpoint that the frontend will call.
@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    """
    Receives an image file from the user, saves it, and returns a success message.
    """
    try:
        # Define the full path where the file will be saved
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

        # Save the uploaded file from memory to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Log to the server console and return a success response to the frontend
        print(f"Successfully received and saved file: {file.filename}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": file.filename,
                "message": "File uploaded successfully. Awaiting AI processing.",
            },
        )
    except Exception as e:
        # If anything goes wrong, log the error and return a 500 status code
        print(f"An error occurred: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )

# --- 6. Local Development Server ---
# This block allows you to run the server directly on your machine using "python main.py"
if __name__ == "__main__":
    print("Starting Forge AI server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
