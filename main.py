# main.py - Day 4 Final Robust Version

# --- 1. Imports ---
import os
import shutil
import mimetypes
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
import cadquery as cq

# --- 2. FastAPI App and CORS Configuration ---
app = FastAPI(title="Forge AI Backend")
origins = ["http://localhost", "http://localhost:5173", "https://forge-ai-frontend.vercel.app"] # Added a placeholder for deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Google AI Client Initialization ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
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


# --- 5. The FINAL Backend Endpoint Logic ---
@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"message": "Google AI client not initialized."})

    try:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        image_file = genai.upload_file(path=file_path)
        
        # --- NEW: SMARTER PROMPT ---
        # We've added more specific rules to prevent common geometry errors.
        prompt = """
        You are an expert 3D modeling assistant. Your task is to analyze a 2D line drawing and generate a Python script using the CadQuery library to create a robust, valid 3D model.

        CRITICAL GEOMETRY GUIDELINES:
        - To avoid 'TopoDS_Builder::Add' errors, you MUST ensure that all `union` and `cut` operations involve shapes that have a clear, overlapping volume. Do NOT create shapes that only touch at a single edge or point.
        - Ensure all cuts pass completely through the object if they are intended to be holes.
        - Avoid creating zero-thickness features or self-intersecting geometry.

        OUTPUT REQUIREMENTS:
        - Your ONLY output must be a complete, runnable Python script.
        - The script must start with 'import cadquery as cq'.
        - The script MUST assign the final 3D object to a variable named 'result'.
        - Example: 'result = cq.Workplane("XY").box(10, 10, 10)'
        - Do NOT include any explanations, greetings, or markdown formatting.
        """

        response = model.generate_content([prompt, image_file])
        generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        print(f"--- Generated Script ---\n{generated_script}\n------------------------")
        genai.delete_file(image_file.name)
        
        # --- NEW: ERROR HANDLING FOR GEOMETRY ---
        try:
            script_locals = {}
            exec(generated_script, {"cq": cq}, script_locals)
            cadquery_object = script_locals.get("result")

            if cadquery_object and isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
                output_stl_path = os.path.join(TEMP_UPLOAD_DIR, "output.stl")
                cq.exporters.export(cadquery_object, output_stl_path)
                print(f"Successfully exported STL file to {output_stl_path}")
                return FileResponse(path=output_stl_path, filename="generated_model.stl", media_type="application/octet-stream")
            else:
                raise ValueError("Script did not produce a valid CadQuery object in the 'result' variable.")

        except Exception as geometry_error:
            # This block now catches errors from the geometry kernel (like TopoDS_Builder)
            print(f"GEOMETRY ERROR: {geometry_error}")
            # Return a user-friendly error message to the frontend
            return JSONResponse(
                status_code=422, # Unprocessable Entity - a good code for this type of error
                content={"message": f"The AI generated a model that was geometrically invalid. Please try a simpler or cleaner drawing. (Error: {geometry_error})"}
            )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

# --- 6. Local Development Server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
