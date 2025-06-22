# main.py - Now with a Debug-Ready Response

import os
import shutil
import mimetypes
import base64 # NEW: Needed to encode the STL file
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
        You are a precise 3D modeling assistant. Your task is to analyze a 2D line drawing and generate a Python script using the CadQuery library to create a robust, valid 3D model.

        CRITICAL GEOMETRY GUIDELINES:
        - Analyze the shapes meticulously. If you see a circle, use `circle()`. If you see a rectangle, use `rect()` or `box()`. Be as literal as possible.
        - To avoid 'TopoDS_Builder::Add' errors, you MUST ensure that all `union` and `cut` operations involve shapes that have a clear, overlapping volume.
        - Extrude the base shape by a default of 10 units.

        OUTPUT REQUIREMENTS:
        - Your ONLY output must be a complete, runnable Python script.
        - The script must start with 'import cadquery as cq'.
        - The script MUST assign the final 3D object to a variable named 'result'.
        - Do NOT include any explanations, greetings, or markdown formatting.
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
                
                # --- NEW: Encode the STL file instead of sending it directly ---
                with open(output_stl_path, "rb") as stl_file:
                    encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
                
                # --- NEW: Return a JSON object with both script and STL data ---
                return JSONResponse(status_code=200, content={
                    "script": generated_script,
                    "stl_data": encoded_stl
                })
            else:
                raise ValueError("Script did not produce a valid CadQuery object.")

        except Exception as geometry_error:
            # Also return the faulty script for debugging
            return JSONResponse(status_code=422, content={
                "message": f"Geometry Error: The AI's script was invalid. (Error: {geometry_error})",
                "script": generated_script
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
