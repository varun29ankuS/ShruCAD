# main.py - Final Version using EasyOCR (Pure Python)

import os
import shutil
import base64
import cv2
import numpy as np
import json
import easyocr # NEW: The EasyOCR library
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
import cadquery as cq

app = FastAPI(title="Forge AI Backend")
origins = ["http://localhost", "http://localhost:5173"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- NEW: Initialize the EasyOCR reader ---
# This line will run once when the server starts.
# It might download the language models on the first run.
print("Initializing EasyOCR Reader...")
ocr_reader = easyocr.Reader(['en'])
print("EasyOCR Reader initialized.")

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
            
        img = cv2.imread(file_path)

        # --- NEW: Specialist 2 - Text Scribe using EasyOCR ---
        ocr_results = ocr_reader.readtext(file_path)
        text_annotations = []
        for (bbox, text, prob) in ocr_results:
            if prob > 0.6: # Confidence threshold
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x, y, w, h = int(top_left[0]), int(top_left[1]), int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
                text_annotations.append({"text": text, "location": [x, y, w, h]})
        
        # (The shape detection part is omitted for this simplified example, focusing on the fix)
        engineering_report = {"text_annotations": text_annotations}
        report_json = json.dumps(engineering_report, indent=2)
        print(f"--- Engineering Report (from EasyOCR) ---\n{report_json}\n--------------------------")
        
        # (The rest of the logic remains the same, sending the report to the LLM)
        prompt = f"""
        You are a senior CAD engineer. You will receive a JSON object containing a pre-analyzed engineering report of a technical drawing. Your task is to interpret this report and generate a final, precise CadQuery Python script. Based on the text and its location, create a plausible object.

        Here is the engineering report:
        {report_json}

        Generate a single, runnable CadQuery Python script. The final object must be assigned to a variable named 'result'. Your ONLY output is the Python script.
        """
        response = model.generate_content(prompt)
        generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        print(f"--- AI Generated Script ---\n{generated_script}\n---------------------------")

        # Execute the script
        script_locals = {}
        exec(generated_script, {"cq": cq}, script_locals)
        cadquery_object = script_locals.get("result")
        if cadquery_object and isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
            output_stl_path = os.path.join(TEMP_UPLOAD_.py
DIR, "output.stl")
            cq.exporters.export(cadquery_object, output_stl_path)
            with open(output_stl_path, "rb") as stl_file:
                encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
            return JSONResponse(status_code=200, content={"script": generated_script, "stl_data": encoded_stl})
        else:
            raise ValueError("Script did not produce a valid CadQuery object.")

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
