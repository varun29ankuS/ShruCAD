# main.py - Day 3 Version (Switched to Google Gemini API)

# --- 1. Imports ---
import os
import shutil
import mimetypes # NEW: To guess the image type
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai # NEW: Import the Google library

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

# --- 3. Google AI Client Initialization ---
# NEW: Configure the client with the API key from Render's environment
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    # Initialize the specific model we want to use
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Google AI Gemini client initialized.")
except Exception as e:
    print(f"Error initializing Google AI client: {e}")
    model = None

# --- 4. Directory Setup ---
TEMP_UPLOAD_DIR = "temp_uploads"
@app.on_event("startup")
async def startup_event():
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    print("Temporary directory is ready.")


# --- 5. The Upgraded API Endpoint (Using Google AI) ---
@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"message": "Google AI client not initialized."})

    try:
        # Save the file first
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        print(f"Successfully received and saved file: {file.filename}")

        # --- NEW: Google AI Processing Logic ---
        # 1. Prepare the image for the API
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream" # Default if type can't be guessed

        image_file = genai.upload_file(path=file_path, mime_type=mime_type)
        print(f"Uploaded file to Google: {image_file.name}")

        # 2. Define the prompt
        prompt = """
        You are an expert 3D modeling assistant. Your task is to analyze a 2D line drawing and generate a Python script using the CadQuery library to create a 3D model of it.

        Guidelines:
        - Analyze the primary shapes in the image.
        - Assume the drawing is a 2D representation to be extruded into a simple 3D shape.
        - Use a default extrusion depth of 10 units.
        - Your ONLY output must be a complete, runnable Python script.
        - Do NOT include any explanations, greetings, or markdown formatting like ```python.
        - The script must start with 'import cadquery as cq' and end with the final CadQuery chain. It should NOT save to a file yet.
        """

        # 3. Call the Google AI API
        print("Sending image to Google AI for analysis...")
        response = model.generate_content([prompt, image_file])

        # 4. Extract the script from the response
        # We remove the markdown backticks just in case the AI adds them
        generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        print("Successfully received script from Google AI.")
        print(f"--- Generated Script ---\n{generated_script}\n------------------------")

        # Clean up the uploaded file on Google's side
        genai.delete_file(image_file.name)
        
        return JSONResponse(status_code=200, content={"script": generated_script})

    except Exception as e:
        print(f"An error occurred during AI processing: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

# --- 6. Local Development Server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
