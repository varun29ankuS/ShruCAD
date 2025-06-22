# Enhanced main.py with improvements

import os
import shutil
import base64
import cv2
import numpy as np
import json
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
import cadquery as cq

app = FastAPI(title="RasterShape AI Backend", version="2.0.0")

# CORS setup
origins = ["*"]  # In production, specify exact origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables
ocr_reader = None
model = None

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
TEMP_UPLOAD_DIR = "temp_uploads"
FILE_CLEANUP_HOURS = 24

# Request models
class PromptRequest(BaseModel):
    prompt: str
    complexity: Optional[str] = "medium"  # simple, medium, complex

# Initialize AI model
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Google AI Gemini client initialized successfully.")
except Exception as e:
    model = None
    print(f"Error initializing Google AI client: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize directories and cleanup old files"""
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    
    # Schedule periodic cleanup
    asyncio.create_task(periodic_cleanup())
    print("Server startup complete. TEMP_UPLOAD_DIR is ready.")

async def periodic_cleanup():
    """Clean up old temporary files"""
    while True:
        try:
            cleanup_old_files()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            print(f"Cleanup error: {e}")

def cleanup_old_files():
    """Remove files older than FILE_CLEANUP_HOURS"""
    cutoff_time = datetime.now() - timedelta(hours=FILE_CLEANUP_HOURS)
    
    for file_path in Path(TEMP_UPLOAD_DIR).glob("*"):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                try:
                    file_path.unlink()
                    print(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    print(f"Error cleaning up {file_path}: {e}")

def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

async def get_ocr_reader():
    """Initialize OCR reader with lazy loading"""
    global ocr_reader
    if ocr_reader is None:
        print("Initializing EasyOCR...")
        import easyocr
        ocr_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if available
        print("EasyOCR initialized successfully.")
    return ocr_reader

def cleanup_temp_file(file_path: str):
    """Background task to clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_model_available": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Generate 3D model from uploaded image"""
    if not model:
        raise HTTPException(
            status_code=503, 
            detail="AI service unavailable. Please check server configuration."
        )
    
    # Validate file
    validate_image_file(file)
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(TEMP_UPLOAD_DIR, f"{timestamp}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, file_path)
        
        # Get OCR reader
        reader = await get_ocr_reader()
        
        # Process image with OCR
        ocr_results = reader.readtext(file_path)
        text_annotations = []
        
        for (bbox, text, confidence) in ocr_results:
            if confidence > 0.6:  # Filter low-confidence results
                (top_left, _, bottom_right, _) = bbox
                x = int(top_left[0])
                y = int(top_left[1])
                w = int(bottom_right[0] - top_left[0])
                h = int(bottom_right[1] - top_left[1])
                
                text_annotations.append({
                    "text": text,
                    "location": [x, y, w, h],
                    "confidence": float(confidence)
                })
        
        if not text_annotations:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in image. Please ensure image contains clear technical drawings or text."
            )
        
        engineering_report = {
            "text_annotations": text_annotations,
            "image_filename": file.filename,
            "total_annotations": len(text_annotations)
        }
        
        report_json = json.dumps(engineering_report, indent=2)
        print(f"Engineering Report:\n{report_json}")
        
        # Enhanced AI prompt
        prompt = f"""
        You are a senior CAD engineer specializing in interpreting technical drawings.
        
        TASK: Generate a precise CadQuery Python script based on this engineering analysis.
        
        ENGINEERING REPORT:
        {report_json}
        
        REQUIREMENTS:
        1. Analyze the text annotations and their spatial relationships
        2. Infer dimensions, features, and geometric relationships
        3. Create a logical 3D object that matches the drawing
        4. Use appropriate CadQuery operations (box, cylinder, hole, fillet, etc.)
        5. Include comments explaining your design decisions
        6. Assign the final object to variable 'result'
        
        OUTPUT: Only the Python script, no explanations or markdown.
        """
        
        # Generate script with AI
        try:
            response = model.generate_content(prompt)
            generated_script = response.text.strip()
            
            # Clean up script formatting
            generated_script = generated_script.replace("```python", "").replace("```", "").strip()
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"AI generation failed: {str(e)}"
            )
        
        # Execute CadQuery script
        script_locals = {}
        try:
            exec(generated_script, {"cq": cq, "math": __import__("math")}, script_locals)
            cadquery_object = script_locals.get("result")
            
            if not cadquery_object or not isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
                raise ValueError("Script did not produce a valid CadQuery object")
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Script execution failed: {str(e)}"
            )
        
        # Export to STL
        output_stl_path = os.path.join(TEMP_UPLOAD_DIR, f"{timestamp}_output.stl")
        try:
            cq.exporters.export(cadquery_object, output_stl_path)
            
            with open(output_stl_path, "rb") as stl_file:
                encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
            
            # Schedule STL cleanup
            background_tasks.add_task(cleanup_temp_file, output_stl_path)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"STL export failed: {str(e)}"
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "script": generated_script,
                "stl_data": encoded_stl,
                "engineering_report": engineering_report,
                "processing_time": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/generate-from-prompt")
async def generate_from_prompt(request_data: PromptRequest):
    """Generate 3D model from text prompt"""
    if not model:
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable"
        )
    
    if not request_data.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )
    
    # Complexity-based prompting
    complexity_instructions = {
        "simple": "Create a basic geometric shape with minimal features.",
        "medium": "Create a moderately detailed object with multiple features.",
        "complex": "Create a detailed object with advanced features like fillets, chamfers, and complex geometries."
    }
    
    system_prompt = f"""
    You are an expert CAD engineer that translates natural language into CadQuery Python scripts.
    
    USER REQUEST: "{request_data.prompt}"
    COMPLEXITY LEVEL: {request_data.complexity} - {complexity_instructions.get(request_data.complexity, "")}
    
    INSTRUCTIONS:
    1. Parse the user's description carefully
    2. Identify key dimensions, shapes, and features
    3. Plan the construction sequence logically
    4. Use appropriate CadQuery methods
    5. Add meaningful comments
    6. Ensure the final object is assigned to 'result'
    
    AVAILABLE METHODS: box(), cylinder(), sphere(), hole(), fillet(), chamfer(), cut(), union(), etc.
    
    OUTPUT: Only the Python script, no markdown formatting.
    """
    
    try:
        response = model.generate_content(system_prompt)
        generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        
        # Execute script
        script_locals = {}
        exec(generated_script, {"cq": cq, "math": __import__("math")}, script_locals)
        cadquery_object = script_locals.get("result")
        
        if not cadquery_object or not isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
            raise ValueError("Script did not produce a valid CadQuery object")
        
        # Export STL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_stl_path = os.path.join(TEMP_UPLOAD_DIR, f"{timestamp}_prompt_output.stl")
        
        cq.exporters.export(cadquery_object, output_stl_path)
        
        with open(output_stl_path, "rb") as stl_file:
            encoded_stl = base64.b64encode(stl_file.read()).decode('utf-8')
        
        # Clean up STL file
        os.remove(output_stl_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "script": generated_script,
                "stl_data": encoded_stl,
                "prompt": request_data.prompt,
                "complexity": request_data.complexity,
                "processing_time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
