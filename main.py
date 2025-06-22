# main.py - The Final "Extractor" Architecture

import os
import shutil
import base64
import cv2
import numpy as np
import json # NEW: To parse the AI's JSON output
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

def preprocess_image(image_path: str) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    processed_path = os.path.join(TEMP_UPLOAD_DIR, "processed_" + os.path.basename(image_path))
    cv2.imwrite(processed_path, thresh)
    return processed_path

@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"message": "Google AI client not initialized."})

    try:
        original_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(original_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        processed_image_path = preprocess_image(original_file_path)
        image_file = genai.upload_file(path=processed_image_path)
        
        # --- NEW "EXTRACTOR" PROMPT ---
        prompt = """
        You are a computer vision data extractor. Your task is to analyze a pure black-and-white image and extract the coordinates of its main silhouette.

        1.  Find the contour of the primary white shape.
        2.  Trace this contour and provide a list of (x, y) coordinates along its path.
        3.  Your ONLY output must be a single, valid JSON array of arrays, where each inner array is an [x, y] coordinate.
        4.  Do NOT include any explanations, greetings, or markdown formatting.
        5.  The first and last points should be the same to indicate a closed loop.

        Example output for a simple square:
        [[0,0], [10,0], [10,10], [0,10], [0,0]]
        """
        response = model.generate_content([prompt, image_file])
        # Clean up the AI's response to be valid JSON
        json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        genai.delete_file(image_file.name)
        
        # --- NEW: PYTHON BUILDS THE MODEL, NOT THE AI ---
        try:
            # 1. Parse the JSON data from the AI
            points = json.loads(json_text)
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError("AI did not return a valid list of points.")

            # 2. Our Python code builds the CadQuery object
            # This is more robust than executing AI-written code.
            result = cq.Workplane("XY").polyline(points).close().extrude(10)
            
            # 3. Export the STL
            output_stl_path = os.path.join(TEMP_UPLOAD_DIR, "output.stl")
            cq.exporters.export(result, output_stl_path)
            
            with open(output_stl_path, "rb") as stl_file:
                encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
            
            # The "script" is now just the JSON data, for debugging
            return JSONResponse(status_code=200, content={"script": json_text, "stl_data": encoded_stl})

        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(status_code=422, content={"message": f"AI returned invalid data: {e}", "script": json_text})
        except Exception as geometry_error:
            return JSONResponse(status_code=422, content={"message": f"Geometry Error: The extracted points created an invalid shape. (Error: {geometry_error})", "script": json_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
