import os
import base64
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_OK = True
except Exception as e:
    log.warning("google-genai import failed")
    genai = None
    genai_types = None
    GENAI_OK = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-1.5-flash")

client = None
if GEMINI_API_KEY and GENAI_OK:
    client = genai.Client(api_key=GEMINI_API_KEY)
    log.info("Gemini client OK")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LAYER_HINTS = {
    "scene": "A complete cinematic scene combining all elements.",
    "object": "The main subject only, isolated on a clean background.",
    "background": "Background environment only, no foreground subjects.",
    "light": "Dramatic lighting and atmosphere overlay for a scene.",
    "mood": "Color grading, mood, and emotional tone overlay.",
}


class ImproveRequest(BaseModel):
    layer: str
    text: str


class LayerRequest(BaseModel):
    layer: str
    prompt: str


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "configured": bool(client),
        "has_key": bool(GEMINI_API_KEY),
        "text_model": TEXT_MODEL,
    }


@app.post("/api/improve")
async def improve(req: ImproveRequest):
    if client is None:
        return JSONResponse(status_code=500, content={"error": "API not configured."})
    text = (req.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Empty prompt."})
    hint = LAYER_HINTS.get(req.layer, "")
    system = (
        "You are an expert AI image prompt engineer. "
        "Improve this prompt for the layer type: " + hint + " "
        "Make it vivid, specific, and cinematically detailed. "
        "Return ONLY the improved prompt text, nothing else."
    )
    try:
        resp = await client.aio.models.generate_content(
            model=TEXT_MODEL,
            contents=[text],
            config=genai_types.GenerateContentConfig(system_instruction=system),
        )
        return {"text": resp.text.strip()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


async def gen_image(prompt, layer):
    log.info("Generating layer: " + layer)
    try:
        resp = await client.aio.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(number_of_images=1),
        )
        if resp.generated_images:
            raw = resp.generated_images[0].image.image_bytes
            b64 = base64.b64encode(raw).decode("utf-8")
            return {"image_base64": b64, "mime_type": "image/png", "layer": layer}
    except Exception as e:
        log.warning("Imagen failed: " + str(e))
    resp = await client.aio.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=[prompt],
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for cand in resp.candidates:
        for part in cand.content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                raw = part.inline_data.data
                b64 = base64.b64encode(raw).decode("utf-8")
                return {
                    "image_base64": b64,
                    "mime_type": part.inline_data.mime_type or "image/png",
                    "layer": layer,
                }
    raise Exception("No image returned")


@app.post("/api/generate_layer")
async def generate_layer(req: LayerRequest):
    if client is None:
        return JSONResponse(status_code=500, content={"error": "API not configured."})
    prompt = (req.prompt or "").strip()
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "Empty prompt."})
    try:
        return await gen_image(prompt, req.layer)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
