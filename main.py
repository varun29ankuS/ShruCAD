# main.py - The Hybrid Vision Pipeline

import os
import shutil
import base64
import cv2
import numpy as np
import json
import pytesseract # NEW: The OCR Scribe
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
            
        # --- STAGE 1: SPECIALIST ANALYSIS ---
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Specialist 1: Shape Detector (finds circles)
        detected_circles = []
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected_circles.append({"center": [int(i[0]), int(i[1])], "radius": int(i[2])})
        
        # Specialist 2: Text Scribe (OCR)
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        text_annotations = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 60: # Confidence threshold
                text = ocr_data['text'][i].strip()
                if text:
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    text_annotations.append({"text": text, "location": [x, y, w, h]})
        
        # Compile the engineering report
        engineering_report = {
            "image_resolution": [img.shape[1], img.shape[0]],
            "detected_shapes": {"circles": detected_circles},
            "text_annotations": text_annotations
        }
        report_json = json.dumps(engineering_report, indent=2)
        print(f"--- Engineering Report ---\n{report_json}\n--------------------------")

        # --- STAGE 2: LEAD ENGINEER (AI REASONING) ---
        prompt = f"""
        You are a senior CAD engineer. You will receive a JSON object containing a pre-analyzed engineering report of a technical drawing. Your task is to interpret this report and generate a final, precise CadQuery Python script.

        Here is the engineering report:
        {report_json}

        INSTRUCTIONS:
        1. Analyze the report. The report contains the image size, detected circles (with center and radius in pixels), and text annotations (with their content and location).
        2. Associate text annotations with nearby shapes. For example, a text label 'ø50' near a circle indicates a diameter of 50 units. 'R25' indicates a radius of 25.
        3. Assume a 1:1 pixel-to-unit mapping unless dimensions contradict this.
        4. Create a logical construction plan. Start with the largest objects and then add smaller features or cutouts.
        5. Generate a single, runnable CadQuery Python script. The final object must be assigned to a variable named 'result'.
        6. Your ONLY output is the Python script. Do not add any other text or formatting.
        """

        response = model.generate_content(prompt)
        generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        print(f"--- AI Generated Script ---\n{generated_script}\n---------------------------")

        # --- STAGE 3: EXECUTION ---
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
