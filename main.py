# main.py - Day 3 Version with AI Core

# --- 1. Imports ---
import os
import shutil
import base64  # NEW: To encode the image for the API
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import OpenAI # NEW: Import the OpenAI library

# --- 2. FastAPI App and CORS Configuration ---
app = FastAPI(title="Forge AI Backend")
origins = ["http://localhost", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. OpenAI Client Initialization ---
# NEW: Initialize the OpenAI client. It will automatically find the
# OPENAI_API_KEY we set in the Render environment.
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI client initialized.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# --- 4. Directory Setup ---
TEMP_UPLOAD_DIR = "temp_uploads"
@app.on_event("startup")
async def startup_event():
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    print("Temporary directory is ready.")


# --- 5. The Upgraded API Endpoint ---
@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    if not client:
        return JSONResponse(status_code=500, content={"message": "OpenAI client not initialized."})

    try:
        # Save the file first
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Successfully received and saved file: {file.filename}")

        # --- NEW: AI Processing Logic ---
        # 1. Encode the image to base64
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 2. Call the OpenAI API
        print("Sending image to OpenAI for analysis...")
        response = client.chat.completions.create(
            model="gpt-4o", # The best model for vision tasks
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert 3D modeling assistant. Your task is to analyze a 2D line drawing and generate a Python script using the CadQuery library to create a 3D model of it.

                    Guidelines:
                    - Analyze the primary shapes in the image.
                    - Assume the drawing is a 2D representation to be extruded into a simple 3D shape.
                    - Use a default extrusion depth of 10 units.
                    - Your ONLY output must be a complete, runnable Python script.
                    - Do NOT include any explanations, greetings, or markdown formatting like ```python.
                    - The script must start with 'import cadquery as cq' and end with the final CadQuery chain. It should NOT save to a file yet.
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this drawing and generate the CadQuery script."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        # 3. Extract the script from the response
        generated_script = response.choices.message.content
        print("Successfully received script from OpenAI.")
        print(f"--- Generated Script ---\n{generated_script}\n------------------------")

        # Return the generated script to the frontend
        return JSONResponse(status_code=200, content={"script": generated_script})

    except Exception as e:
        print(f"An error occurred during AI processing: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

# --- 6. Local Development Server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
