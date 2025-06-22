# main.py - The Quality Overhaul Version

import os
import shutil
import base64
import cv2  # NEW: Import OpenCV
import numpy as np # NEW: Import numpy
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
import cadquery as cq

app = FastAPI(title="Forge AI Backend")
origins = ["http://localhost", "http://localhost:5173"] 
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Google AI Gemini client initialized.")
except Exception as e:
    model = None
    print(f"Error initializing Google AI client: {e}")

TEMP_UPLOAD_DIR = "temp_uploads"
@app.on_event("startup")
async def startup_event():
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- NEW: Image Pre-processing Function ---
def preprocess_image(image_path: str) -> str:
    """Loads an image, converts it to pure black and white, and returns the path."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Invert the image if it has a white background
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    # Apply a threshold to make it pure black and white
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # Define the output path for the processed image
    processed_path = os.path.join(TEMP_UPLOAD_DIR, "processed_" + os.path.basename(image_path))
    cv2.imwrite(processed_path, thresh)
    print(f"Image processed and saved to {processed_path}")
    return processed_path


@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"message": "Google AI client not initialized."})

    try:
        # Save the original file
        original_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(original_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # --- NEW: Pre-process the image first ---
        processed_image_path = preprocess_image(original_file_path)

        # Upload the CLEANED image to the AI
        image_file = genai.upload_file(path=processed_image_path)
        
        # --- NEW: THE "GENIUS" PROMPT ---
        prompt = """
        You are a hyper-logical, precision-focused CAD conversion bot. Your task is to analyze a simplified, pure black-and-white 2D drawing and generate a CadQuery Python script. Follow these steps meticulously:

        1.  **Analyze Silhouette:** Identify the primary, outermost continuous shape in the image. This will be your base extrusion.
        2.  **Identify Internal Cutouts:** Identify any separate, enclosed white spaces *inside* the main silhouette. These are holes or cuts.
        3.  **Translate to CadQuery:**
            *   Start with a workplane: `cq.Workplane("XY")`
            *   Draw the main silhouette using a chain of `.line()`, `.threePointArc()`, etc., then `.close()` to form a wire.
            *   Extrude the main silhouette: `.extrude(10)`
            *   For EACH internal cutout, find its center, create a new workplane at that center, draw its shape, and use `.cutThruAll()`.
        4.  **Final Object:** Assign the final object to a variable named `result`.

        **CRITICAL RULES:**
        - You are looking at a CLEAN, black-and-white image. Do not invent features. Be extremely literal.
        - DO NOT use `box()`, `circle()`, or `rect()` for complex outlines. Build the outline from lines and arcs. Use `sketch()` if necessary for complex shapes.
        - Your ONLY output is the Python script. No explanations.
        - Example for a simple shape: `result = cq.Workplane("XY").sketch().rect(10, 20).finalize().extrude(10)`
        """
        response = model.generate_content([prompt, image_file])
        generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        genai.delete_file(image_file.name)
        
        try:
            script_locals = {}
            exec(generated_script, {"cq": cq}, script_locals)
            cadquery_object = script_locals.get("result")

            if cadquery_object and isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
                output_stl_path = os.path.join(TEMP_UPLOAD_DIR, "output.stl")
                cq.exporters.export(cadquery_object, output_stl_path)
                
                with open(output_stl_path, "rb") as stl_file:
                    encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
                
                return JSONResponse(status_code=200, content={"script": generated_script, "stl_data": encoded_stl})
            else:
                raise ValueError("Script did not produce a valid CadQuery object.")
        except Exception as geometry_error:
            return JSONResponse(status_code=422, content={"message": f"Geometry Error: {geometry_error}", "script": generated_script})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
