# main.py - Day 4 Final Backend Version

# --- 1. Imports ---
import os
import shutil
import mimetypes
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse # NEW: Import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
import cadquery as cq # NEW: Import CadQuery itself

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
        
        prompt = """
        You are an expert 3D modeling assistant. Your task is to analyze a 2D line drawing and generate a Python script using the CadQuery library to create a 3D model of it.

        Guidelines:
        - Analyze the primary shapes in the image.
        - Assume the drawing is a 2D representation to be extruded into a simple 3D shape.
        - Use a default extrusion depth of 10 units.
        - Your ONLY output must be a complete, runnable Python script.
        - Do NOT include any explanations, greetings, or markdown formatting like ```python.
        - The script must start with 'import cadquery as cq' and assign the final object to a variable named 'result'. For example: 'result = cq.Workplane("XY").box(10, 10, 10)'
        """

        response = model.generate_content([prompt, image_file])
        generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        print(f"--- Generated Script ---\n{generated_script}\n------------------------")
        genai.delete_file(image_file.name)
        
        # --- NEW: Execute the script and generate the STL file ---
        # 1. Define a dictionary to hold the result of the executed script
        script_locals = {}
        # 2. SECURITY NOTE: exec() is powerful but dangerous in a real production app.
        #    For our prototype, it's the fastest way to get results.
        exec(generated_script, {"cq": cq}, script_locals)
        # 3. Get the final 3D object from the 'result' variable the AI created
        cadquery_object = script_locals.get("result")

        if cadquery_object:
            # 4. Define the output path and export the STL file
            output_stl_path = os.path.join(TEMP_UPLOAD_DIR, "output.stl")
            cq.exporters.export(cadquery_object, output_stl_path)
            print(f"Successfully exported STL file to {output_stl_path}")
            
            # 5. Return the generated file itself as the response
            return FileResponse(
                path=output_stl_path,
                filename="generated_model.stl",
                media_type="application/octet-stream"
            )
        else:
            raise ValueError("The executed script did not produce a 'result' object.")

    except Exception as e:
        print(f"An error occurred during STL generation: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

# --- 6. Local Development Server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
