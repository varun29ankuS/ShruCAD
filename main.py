# main.py - Fixed Version with Critical Issues Resolved

import os
import shutil
import base64
import cv2
import numpy as np
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
import cadquery as cq

app = FastAPI(title="RasterShape AI Backend")

# CORS setup - consider restricting origins in production
origins = ["*"]  # For development - restrict in production
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Global variables
ocr_reader = None

# Initialize Google AI
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY environment variable not set")
        model = None
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Google AI Gemini client initialized successfully.")
except Exception as e:
    model = None
    print(f"Error initializing Google AI client: {e}")

TEMP_UPLOAD_DIR = "temp_uploads"

@app.on_event("startup")
async def startup_event():
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    print("Server startup complete. TEMP_UPLOAD_DIR is ready.")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "ai_model_available": model is not None,
        "ocr_ready": ocr_reader is not None,
        "temp_dir_exists": os.path.exists(TEMP_UPLOAD_DIR),
        "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY"))
    }

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

async def initialize_ocr():
    """Initialize OCR reader with better error handling"""
    global ocr_reader
    if ocr_reader is None:
        try:
            print("First request received. Importing and initializing EasyOCR...")
            import easyocr
            ocr_reader = easyocr.Reader(['en'], gpu=False)  # Explicitly disable GPU for Render
            print("EasyOCR Reader initialized and ready for future requests.")
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"OCR service initialization failed: {str(e)}"
            )
    return ocr_reader

@app.post("/generate")
async def generate_model(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(
            status_code=503, 
            detail="AI service unavailable. GOOGLE_API_KEY may not be set."
        )

    # Validate file
    validate_file(file)
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Reset file pointer
    await file.seek(0)
    
    # Initialize OCR
    reader = await initialize_ocr()
    
    file_path = None
    try:
        # Save uploaded file
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Process with OCR
        ocr_results = reader.readtext(file_path)
        text_annotations = []
        
        for (bbox, text, prob) in ocr_results:
            if prob > 0.6:
                (top_left, _, bottom_right, _) = bbox
                x, y = int(top_left[0]), int(top_left[1])
                w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
                text_annotations.append({
                    "text": text, 
                    "location": [x, y, w, h],
                    "confidence": float(prob)
                })
        
        if not text_annotations:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in image. Please ensure image contains clear technical drawings or text."
            )
        
        engineering_report = {
            "text_annotations": text_annotations,
            "total_annotations": len(text_annotations)
        }
        report_json = json.dumps(engineering_report, indent=2)
        print(f"--- Engineering Report (from EasyOCR) ---\n{report_json}\n--------------------------")
        
        # Generate with AI
        prompt = f"""
        You are a senior CAD engineer. You will receive a JSON object containing a pre-analyzed engineering report of a technical drawing. Your task is to interpret this report and generate a final, precise CadQuery Python script. Based on the text and its location, create a plausible object.
        
        Here is the engineering report:
        {report_json}
        
        Generate a single, runnable CadQuery Python script. The final object must be assigned to a variable named 'result'. Your ONLY output is the Python script.
        """
        
        try:
            response = model.generate_content(prompt)
            generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
        
        print(f"--- AI Generated Script ---\n{generated_script}\n---------------------------")

        # Execute CadQuery script
        script_locals = {}
        try:
            exec(generated_script, {"cq": cq}, script_locals)
            cadquery_object = script_locals.get("result")
            
            if not cadquery_object or not isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
                raise ValueError("Script did not produce a valid CadQuery object.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {str(e)}")
        
        # Export STL
        output_stl_path = os.path.join(TEMP_UPLOAD_DIR, "output.stl")
        try:
            cq.exporters.export(cadquery_object, output_stl_path)
            with open(output_stl_path, "rb") as stl_file:
                encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"STL export failed: {str(e)}")
        finally:
            # Clean up STL file
            if os.path.exists(output_stl_path):
                os.remove(output_stl_path)
        
        return JSONResponse(
            status_code=200, 
            content={
                "script": generated_script, 
                "stl_data": encoded_stl,
                "engineering_report": engineering_report
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Always clean up input file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")

@app.post("/generate-from-prompt")
async def generate_from_prompt(request_data: dict):
    """Generate 3D model from text prompt"""
    prompt_text = request_data.get("prompt")
    if not prompt_text:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    if not model:
        raise HTTPException(
            status_code=503, 
            detail="AI service unavailable. GOOGLE_API_KEY may not be set."
        )

    try:
        system_prompt = f"""
        You are an expert CAD engineer that translates natural language descriptions into precise CadQuery Python scripts.

        USER'S REQUEST: "{prompt_text}"

        INSTRUCTIONS:
        1. Read the user's request carefully.
        2. Translate the description into a logical sequence of CadQuery operations.
        3. Generate a single, runnable CadQuery Python script.
        4. The final object MUST be assigned to a variable named 'result'.
        5. Your ONLY output is the Python script. Do not add any explanations, greetings, or markdown formatting.
        """
        
        print(f"--- Sending Text Prompt to AI ---\n{prompt_text}\n--------------------------")
        
        try:
            response = model.generate_content(system_prompt)
            generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
        
        print(f"--- AI Generated Script ---\n{generated_script}\n---------------------------")

        # Execute script
        script_locals = {}
        try:
            exec(generated_script, {"cq": cq}, script_locals)
            cadquery_object = script_locals.get("result")
            
            if not cadquery_object or not isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
                raise ValueError("Script did not produce a valid CadQuery object.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {str(e)}")
        
        # Export STL
        output_stl_path = os.path.join(TEMP_UPLOAD_DIR, "output_from_prompt.stl")
        try:
            cq.exporters.export(cadquery_object, output_stl_path)
            
            with open(output_stl_path, "rb") as stl_file:
                encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
            
            return JSONResponse(
                status_code=200, 
                content={
                    "script": generated_script, 
                    "stl_data": encoded_stl,
                    "prompt": prompt_text
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"STL export failed: {str(e)}")
        finally:
            # Clean up STL file
            if os.path.exists(output_stl_path):
                try:
                    os.remove(output_stl_path)
                except Exception as e:
                    print(f"Error cleaning up STL file: {e}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# CRITICAL FIX: Use $PORT environment variable
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
