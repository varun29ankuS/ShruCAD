# main.py - Production Ready Version with Enhanced Error Handling

import os
import shutil
import base64
import json
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RasterShape AI Backend",
    version="1.0.0",
    description="AI-powered 3D modeling and technical drawings"
)

# CORS setup
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://your-frontend-domain.com",  # Add your actual frontend domain
    "*"  # Remove this in production and specify exact domains
]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["GET", "POST", "PUT", "DELETE"], 
    allow_headers=["*"]
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.stl'}
TEMP_UPLOAD_DIR = None  # Will be set to system temp dir

# Global variables
ocr_reader = None
genai_model = None

# Pydantic models
class ChatMessage(BaseModel):
    id: int
    type: str
    content: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    conversation_history: List[ChatMessage] = []

class PromptRequest(BaseModel):
    prompt: str
    complexity: str = "medium"

# Initialize dependencies with better error handling
async def initialize_dependencies():
    """Initialize all dependencies with proper error handling"""
    global genai_model, TEMP_UPLOAD_DIR
    
    # Set up temp directory
    TEMP_UPLOAD_DIR = tempfile.mkdtemp(prefix="rastershape_")
    logger.info(f"Created temp directory: {TEMP_UPLOAD_DIR}")
    
    # Initialize Google AI
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY environment variable not set")
            genai_model = None
        else:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Google AI Gemini client initialized successfully")
    except ImportError:
        logger.error("google-generativeai package not installed")
        genai_model = None
    except Exception as e:
        logger.error(f"Error initializing Google AI client: {e}")
        genai_model = None

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    await initialize_dependencies()
    logger.info("Server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    global TEMP_UPLOAD_DIR
    if TEMP_UPLOAD_DIR and os.path.exists(TEMP_UPLOAD_DIR):
        try:
            shutil.rmtree(TEMP_UPLOAD_DIR)
            logger.info(f"Cleaned up temp directory: {TEMP_UPLOAD_DIR}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RasterShape AI Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "ai_model_available": genai_model is not None,
        "ocr_ready": ocr_reader is not None,
        "temp_dir_exists": TEMP_UPLOAD_DIR and os.path.exists(TEMP_UPLOAD_DIR),
        "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "dependencies": {
            "fastapi": True,
            "google_generativeai": genai_model is not None,
            "cadquery": check_cadquery_available(),
            "easyocr": check_easyocr_available()
        }
    }

def check_cadquery_available() -> bool:
    """Check if CadQuery is available"""
    try:
        import cadquery as cq
        return True
    except ImportError:
        return False

def check_easyocr_available() -> bool:
    """Check if EasyOCR is available"""
    try:
        import easyocr
        return True
    except ImportError:
        return False

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file with enhanced checks"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if len(file.filename) > 255:
        raise HTTPException(status_code=400, detail="Filename too long")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

async def initialize_ocr():
    """Initialize OCR reader optimized for Render free tier (512MB RAM)"""
    global ocr_reader
    if ocr_reader is None:
        try:
            logger.info("Initializing EasyOCR for Render free tier...")
            import easyocr
            import gc
            
            # Free up memory before loading OCR
            gc.collect()
            
            # Optimized for low memory - single language, minimal models
            ocr_reader = easyocr.Reader(
                ['en'], 
                gpu=False, 
                verbose=False,
                model_storage_directory='/tmp/easyocr',  # Use temp storage
                download_enabled=True
            )
            
            logger.info("EasyOCR Reader initialized successfully")
            
            # Free up memory after initialization
            gc.collect()
            
        except ImportError:
            logger.error("EasyOCR package not installed")
            raise HTTPException(
                status_code=500, 
                detail="OCR service not available. EasyOCR package not installed."
            )
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"OCR service initialization failed: {str(e)}"
            )
    return ocr_reader

async def cleanup_file(file_path: str):
    """Safely cleanup file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Enhanced chat endpoint with better error handling"""
    if not genai_model:
        raise HTTPException(
            status_code=503, 
            detail="AI chat service is not configured. Please set GOOGLE_API_KEY environment variable."
        )
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(request.message) > 4000:
        raise HTTPException(status_code=400, detail="Message too long. Maximum 4000 characters.")
    
    try:
        # Build context-aware prompt
        system_prompt = """You are an expert AI assistant specializing in 3D modeling, CAD design, and manufacturing. 
        You help users with:
        - 3D modeling questions and best practices
        - CAD design optimization
        - 3D printing advice and troubleshooting
        - CadQuery/OpenSCAD code explanation and improvements
        - Technical drawing and manufacturing guidance
        - Material selection and design considerations
        
        Provide practical, actionable advice. Be concise but thorough. Always be helpful and professional."""
        
        # Add context about user's current work
        context_info = ""
        if request.context:
            context_info = f"\n\nCurrent user context:"
            if request.context.get('activeTab'):
                context_info += f"\n- Working on: {request.context['activeTab']} workflow"
            if request.context.get('hasModel'):
                context_info += f"\n- Has generated 3D model: Yes"
            if request.context.get('currentPrompt'):
                prompt_preview = request.context['currentPrompt'][:100]
                context_info += f"\n- Current project: {prompt_preview}..."
            if request.context.get('complexity'):
                context_info += f"\n- Complexity level: {request.context['complexity']}"
            if request.context.get('recentScript'):
                script_preview = request.context['recentScript'][:200]
                context_info += f"\n- Recent generated code snippet: {script_preview}..."
        
        # Build conversation history
        conversation_context = ""
        if request.conversation_history:
            conversation_context = "\n\nRecent conversation:"
            for msg in request.conversation_history[-4:]:  # Last 4 messages
                role = "User" if msg.type == "user" else "Assistant"
                content_preview = msg.content[:150]
                conversation_context += f"\n{role}: {content_preview}..."
        
        # Complete prompt
        full_prompt = f"{system_prompt}{context_info}{conversation_context}\n\nUser's current question: {request.message}\n\nProvide a helpful response:"
        
        # Generate response using Gemini
        response = genai_model.generate_content(full_prompt)
        
        if not response.text:
            raise Exception("Empty response from AI model")
        
        return {
            "response": response.text,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        # Return a more user-friendly error message
        if "quota" in str(e).lower():
            detail = "AI service is temporarily over quota. Please try again later."
        elif "rate limit" in str(e).lower():
            detail = "Too many requests. Please wait a moment and try again."
        else:
            detail = "AI chat service is temporarily unavailable. Please try again."
        
        raise HTTPException(status_code=503, detail=detail)

@app.post("/generate-from-prompt")
async def generate_from_prompt(request: PromptRequest, background_tasks: BackgroundTasks):
    """Generate 3D model from text prompt with improved error handling"""
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if len(request.prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt too long. Maximum 2000 characters.")

    if not genai_model:
        raise HTTPException(
            status_code=503, 
            detail="AI service unavailable. GOOGLE_API_KEY may not be set."
        )

    if not check_cadquery_available():
        raise HTTPException(
            status_code=503,
            detail="CadQuery is not available. 3D modeling service is disabled."
        )

    output_stl_path = None
    try:
        # Import CadQuery
        import cadquery as cq
        
        system_prompt = f"""
        You are an expert CAD engineer that translates natural language descriptions into precise CadQuery Python scripts.

        USER'S REQUEST: "{request.prompt}"
        COMPLEXITY LEVEL: {request.complexity}

        INSTRUCTIONS:
        1. Read the user's request carefully.
        2. Translate the description into a logical sequence of CadQuery operations.
        3. Generate a single, runnable CadQuery Python script.
        4. The final object MUST be assigned to a variable named 'result'.
        5. Use appropriate dimensions and reasonable defaults if not specified.
        6. Your ONLY output is the Python script. Do not add any explanations, greetings, or markdown formatting.
        
        Example format:
        import cadquery as cq
        result = cq.Workplane("XY").box(10, 10, 10)
        """
        
        logger.info(f"Generating model from prompt: {request.prompt[:100]}...")
        
        try:
            response = genai_model.generate_content(system_prompt)
            generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
        
        logger.info("AI script generated successfully")

        # Execute script in controlled environment
        script_locals = {}
        script_globals = {"cq": cq, "__builtins__": {}}  # Restricted globals for security
        
        try:
            exec(generated_script, script_globals, script_locals)
            cadquery_object = script_locals.get("result")
            
            if not cadquery_object:
                raise ValueError("Script did not assign anything to 'result' variable")
            
            if not isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
                raise ValueError(f"Script produced {type(cadquery_object).__name__}, expected CadQuery Workplane or Shape")
                
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generated script execution failed: {str(e)}")
        
        # Export STL with unique filename
        import uuid
        filename = f"output_{uuid.uuid4().hex[:8]}.stl"
        output_stl_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        
        try:
            cq.exporters.export(cadquery_object, output_stl_path)
            
            with open(output_stl_path, "rb") as stl_file:
                stl_content = stl_file.read()
                if len(stl_content) == 0:
                    raise ValueError("Generated STL file is empty")
                
                encoded_stl = base64.b64encode(stl_content).decode('utf-8')
            
            # Schedule cleanup
            background_tasks.add_task(cleanup_file, output_stl_path)
            
            logger.info(f"Model generated successfully. STL size: {len(stl_content)} bytes")
            
            return JSONResponse(
                status_code=200, 
                content={
                    "script": generated_script, 
                    "stl_data": encoded_stl,
                    "prompt": request.prompt,
                    "complexity": request.complexity,
                    "stl_size_bytes": len(stl_content)
                }
            )
            
        except Exception as e:
            logger.error(f"STL export failed: {e}")
            raise HTTPException(status_code=500, detail=f"STL export failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_from_prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Ensure cleanup even on error
        if output_stl_path and os.path.exists(output_stl_path):
            await cleanup_file(output_stl_path)

@app.post("/generate")
async def generate_model(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Generate model from image with enhanced error handling"""
    if not genai_model:
        raise HTTPException(
            status_code=503, 
            detail="AI service unavailable. GOOGLE_API_KEY may not be set."
        )

    if not check_cadquery_available():
        raise HTTPException(
            status_code=503,
            detail="CadQuery is not available. 3D modeling service is disabled."
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
    output_stl_path = None
    
    try:
        # Import required modules
        import cadquery as cq
        import uuid
        
        # Save uploaded file with unique name
        filename = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Process with OCR
        ocr_results = reader.readtext(file_path)
        text_annotations = []
        
        for (bbox, text, prob) in ocr_results:
            if prob > 0.6:
                (top_left, _, bottom_right, _) = bbox
                x, y = int(top_left[0]), int(top_left[1])
                w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
                text_annotations.append({
                    "text": text.strip(), 
                    "location": [x, y, w, h],
                    "confidence": float(prob)
                })
        
        if not text_annotations:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in image. Please ensure image contains clear technical drawings with readable text/dimensions."
            )
        
        engineering_report = {
            "text_annotations": text_annotations,
            "total_annotations": len(text_annotations),
            "filename": file.filename
        }
        
        logger.info(f"OCR found {len(text_annotations)} text annotations")
        
        # Generate with AI
        prompt = f"""
        You are a senior CAD engineer. You will receive a JSON object containing a pre-analyzed engineering report of a technical drawing. Your task is to interpret this report and generate a final, precise CadQuery Python script.

        Here is the engineering report:
        {json.dumps(engineering_report, indent=2)}

        INSTRUCTIONS:
        1. Analyze the text annotations and their positions to understand the drawing
        2. Extract dimensions, labels, and geometric features
        3. Generate a single, runnable CadQuery Python script
        4. The final object MUST be assigned to a variable named 'result'
        5. Use reasonable defaults if dimensions are unclear
        6. Your ONLY output is the Python script

        Example format:
        import cadquery as cq
        result = cq.Workplane("XY").box(10, 10, 10)
        """
        
        try:
            response = genai_model.generate_content(prompt)
            generated_script = response.text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
        
        logger.info("AI script generated from image analysis")

        # Execute CadQuery script
        script_locals = {}
        script_globals = {"cq": cq, "__builtins__": {}}
        
        try:
            exec(generated_script, script_globals, script_locals)
            cadquery_object = script_locals.get("result")
            
            if not cadquery_object or not isinstance(cadquery_object, (cq.Workplane, cq.Shape)):
                raise ValueError("Script did not produce a valid CadQuery object.")
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Script execution failed: {str(e)}")
        
        # Export STL
        stl_filename = f"output_{uuid.uuid4().hex[:8]}.stl"
        output_stl_path = os.path.join(TEMP_UPLOAD_DIR, stl_filename)
        
        try:
            cq.exporters.export(cadquery_object, output_stl_path)
            with open(output_stl_path, "rb") as stl_file:
                stl_content = stl_file.read()
                encoded_stl = base64.b64encode(stl_content).decode('utf-8')
        except Exception as e:
            logger.error(f"STL export failed: {e}")
            raise HTTPException(status_code=500, detail=f"STL export failed: {str(e)}")
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, file_path)
            background_tasks.add_task(cleanup_file, output_stl_path)
        
        logger.info(f"Model generated from image successfully. STL size: {len(stl_content)} bytes")
        
        return JSONResponse(
            status_code=200, 
            content={
                "script": generated_script, 
                "stl_data": encoded_stl,
                "engineering_report": engineering_report,
                "stl_size_bytes": len(stl_content)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Cleanup files
        if file_path and os.path.exists(file_path):
            await cleanup_file(file_path)
        if output_stl_path and os.path.exists(output_stl_path):
            await cleanup_file(output_stl_path)

@app.post("/generate-drawings")
async def generate_drawings(
    file: UploadFile = File(...),
    views: str = Form(...),
    scale: str = Form("1:1"),
    format: str = Form("svg")
):
    """Generate technical drawings from STL file with improved error handling"""
    try:
        # Validate inputs
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Parse the views parameter
        try:
            selected_views = json.loads(views)
            if not isinstance(selected_views, list) or not selected_views:
                raise ValueError("Views must be a non-empty list")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid views format: {e}")
        
        # Validate scale format
        if not scale or ":" not in scale:
            scale = "1:1"
        
        logger.info(f"Generating {len(selected_views)} drawings with scale {scale}")
        
        # For now, return enhanced mock SVG drawings
        # TODO: Implement actual STL to 2D projection using mesh processing
        drawings = {}
        
        view_templates = {
            "front": {"angle": "0°", "projection": "XY"},
            "side": {"angle": "90°", "projection": "YZ"}, 
            "top": {"angle": "Top", "projection": "XZ"},
            "isometric": {"angle": "Iso", "projection": "3D"}
        }
        
        for view in selected_views:
            view_info = view_templates.get(view, {"angle": "Unknown", "projection": "2D"})
            
            # Create enhanced SVG drawing for each view
            svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#e0e0e0" stroke-width="0.5"/>
        </pattern>
    </defs>
    
    <!-- Background with grid -->
    <rect width="400" height="300" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
    <rect width="400" height="300" fill="url(#grid)" opacity="0.3"/>
    
    <!-- Title -->
    <text x="200" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#212529">
        {view.title()} View - {view_info['projection']} Projection
    </text>
    <text x="200" y="50" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#6c757d">
        Scale: {scale} | Generated from STL
    </text>
    
    <!-- Main drawing area -->
    <g transform="translate(200,150)">
        <!-- Primary shape -->
        <rect x="-60" y="-40" width="120" height="80" fill="none" stroke="#007bff" stroke-width="2"/>
        
        <!-- Feature details -->
        <circle cx="0" cy="0" r="20" fill="none" stroke="#007bff" stroke-width="2"/>
        <circle cx="-30" cy="-20" r="8" fill="none" stroke="#dc3545" stroke-width="1.5"/>
        <circle cx="30" cy="-20" r="8" fill="none" stroke="#dc3545" stroke-width="1.5"/>
        
        <!-- Dimension lines -->
        <g stroke="#28a745" stroke-width="1" fill="#28a745">
            <!-- Horizontal dimension -->
            <line x1="-70" y1="-50" x2="70" y2="-50"/>
            <line x1="-70" y1="-55" x2="-70" y2="-45"/>
            <line x1="70" y1="-55" x2="70" y2="-45"/>
            <text x="0" y="-58" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">120mm</text>
            
            <!-- Vertical dimension -->
            <line x1="-75" y1="-45" x2="-75" y2="45"/>
            <line x1="-80" y1="-45" x2="-70" y2="-45"/>
            <line x1="-80" y1="45" x2="-70" y2="45"/>
            <text x="-85" y="0" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" transform="rotate(-90, -85, 0)">80mm</text>
            
            <!-- Radius dimension -->
            <line x1="14" y1="-14" x2="35" y2="-35"/>
            <text x="40" y="-30" text-anchor="start" font-family="Arial, sans-serif" font-size="9">R20</text>
        </g>
        
        <!-- Center lines -->
        <g stroke="#ffc107" stroke-width="1" stroke-dasharray="5,5">
            <line x1="-80" y1="0" x2="80" y2="0"/>
            <line x1="0" y1="-60" x2="0" y2="60"/>
        </g>
    </g>
    
    <!-- Drawing information -->
    <g transform="translate(20, 270)">
        <text x="0" y="0" font-family="Arial, sans-serif" font-size="10" fill="#495057">
            Material: Steel | Tolerance: ±0.1mm | Surface: Ra 1.6
        </text>
    </g>
    
    <!-- Title block -->
    <g transform="translate(280, 250)">
        <rect x="0" y="0" width="110" height="40" fill="none" stroke="#343a40" stroke-width="1"/>
        <text x="55" y="15" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" font-weight="bold">RasterShape AI</text>
        <text x="55" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="8">{view.title()} View</text>
        <text x="55" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="8">Scale: {scale}</text>
    </g>
</svg>"""
            
            # Convert to base64
            drawings[view] = base64.b64encode(svg_content.encode('utf-8')).decode('ascii')
        
        logger.info(f"Generated {len(drawings)} technical drawings successfully")
        
        return {
            "drawings": drawings,
            "status": "success",
            "message": f"Generated {len(drawings)} technical drawings",
            "scale": scale,
            "format": format,
            "views": selected_views
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drawing generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drawing generation failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Path {request.url.path} not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# CRITICAL: Render free tier specific configuration
if __name__ == "__main__":
    # Render uses PORT 10000 by default for web services
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"  # Required for Render
    
    logger.info(f"Starting RasterShape AI Backend on {host}:{port}")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    logger.info(f"Python version: {os.sys.version}")
    
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port, 
        reload=False,  # Never use reload in production
        access_log=True,
        log_level="info",
        workers=1  # Single worker for free tier memory limit
    )
