import os
from typing import List, Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
from pydantic import BaseModel
import json

app = FastAPI(title="Follow-Up Email Generator")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration for open-webui API
WEBUI_ENABLED = True  # Set to use open-webui API
WEBUI_BASE_URL = "https://chat.ivislabs.in/api"
API_KEY = "sk-61d2af07a85145c4966c2b1bceba9c34"  # Replace with your actual API key if needed
# Default model based on available models
DEFAULT_MODEL = "gemma2:2b"  # Update to one of the available models

# Fallback to local Ollama API if needed
OLLAMA_ENABLED = True  # Set to False to use only the web UI API
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api"

class EmailRequest(BaseModel):
    prospect_name: str
    last_interaction: str
    previous_feedback: str
    next_step: Optional[str] = "Follow-up on the previous discussion"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_email(
    prospect_name: str = Form(...),
    last_interaction: str = Form(...),
    previous_feedback: str = Form(...),
    next_step: str = Form("Follow-up on the previous discussion"),
    model: str = Form(DEFAULT_MODEL)
):
    try:
        # Build the prompt using the template
        prompt = f"Generate a personalized follow-up email for {prospect_name}. The last interaction was: {last_interaction}. The prospect gave the following feedback: {previous_feedback}. The next step is: {next_step}. Create a professional tone email."

        # Try using the open-webui API first if enabled
        if WEBUI_ENABLED:
            try:
                # Prepare message for API format
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                # Debug: Print request payload
                request_payload = {
                    "model": model,
                    "messages": messages
                }
                print(f"Attempting open-webui API with payload: {json.dumps(request_payload)}")
                
                # Call Chat Completions API
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{WEBUI_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json=request_payload,
                        timeout=60.0
                    )
                    
                    # Debug: Print response details
                    print(f"Open-webui API response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Extract generated text from the response with fallback options
                        generated_text = ""
                        
                        # Try different possible response formats
                        if "choices" in result and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                generated_text = choice["message"]["content"]
                            elif "text" in choice:
                                generated_text = choice["text"]
                        elif "response" in result:
                            generated_text = result["response"]
                        
                        if generated_text:
                            return {"generated_email": generated_text}
            except Exception as e:
                print(f"Open-webui API attempt failed: {str(e)}")
        
        # Fallback to direct Ollama API if enabled and web UI failed
        if OLLAMA_ENABLED:
            print("Falling back to direct Ollama API")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_API_URL}/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to generate email from Ollama API")
                
                result = response.json()
                generated_text = result.get("response", "")
                
                return {"generated_email": generated_text}
                
        # If we get here, both attempts failed
        raise HTTPException(status_code=500, detail="Failed to generate email from any available LLM API")
            
    except Exception as e:
        import traceback
        print(f"Error generating follow-up email: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating email: {str(e)}")

@app.get("/models")
async def get_models():
    try:
        # Try the open-webui models API first if enabled
        if WEBUI_ENABLED:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{WEBUI_BASE_URL}/models",
                        headers={
                            "Authorization": f"Bearer {API_KEY}"
                        }
                    )
                    
                    if response.status_code == 200:
                        models_data = response.json()
                        
                        # Handle the specific response format we received
                        if "data" in models_data and isinstance(models_data["data"], list):
                            model_names = []
                            for model in models_data["data"]:
                                if "id" in model:
                                    model_names.append(model["id"])
                            

                            # If models found, return them
                            if model_names:
                                return {"models": model_names}
            except Exception as e:
                print(f"Error fetching models from open-webui API: {str(e)}")
        
        # Fallback to Ollama's API if enabled
        if OLLAMA_ENABLED:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{OLLAMA_API_URL}/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [model.get("name") for model in models]
                        return {"models": model_names}
            except Exception as e:
                print(f"Error fetching models from Ollama: {str(e)}")
        
        # If all attempts fail, return default model and some common models
        fallback_models = [DEFAULT_MODEL, "gemma2:2b", "qwen2.5:0.5b", "deepseek-r1:1.5b", "deepseek-coder:latest"]
        return {"models": fallback_models}
    except Exception as e:
        print(f"Unepected error in get_models: {str(e)}")
        return {"models": [DEFAULT_MODEL]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=5500, reload=True)
