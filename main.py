from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
import os
import requests
import fitz  # PyMuPDF

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectRoomRequest(BaseModel):
    pdf_url: str
    click_x: float
    click_y: float
    page_number: int = 1
    crop_radius: float = 300  # rayon en points PDF autour du clic

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/debug/{filename}")
def get_debug_image(filename: str):
    path = f"/tmp/{filename}"
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Fichier non trouvé")

@app.post("/detect-room")
def detect_room(req: DetectRoomRequest):
    try:
        print(f"CLICK: x={req.click_x}, y={req.click_y}, page={req.page_number}")

        # Télécharger le PDF
        response = requests.get(req.pdf_url, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Impossible de télécharger le PDF")

        # Ouvrir avec PyMuPDF
        pdf = fitz.open(stream=response.content, filetype="pdf")
        page = pdf[req.page_number - 1]
        page_rect = page.rect
        print(f"PAGE SIZE: {page_rect.width}x{page_rect.height} points")

        # Définir le crop autour du clic
        r = req.crop_radius
        clip = fitz.Rect(
            max(0, req.click_x - r),
            max(0, req.click_y - r),
            min(page_rect.width, req.click_x + r),
            min(page_rect.height, req.click_y + r),
        )

        crop_x = clip.x0
        crop_y = clip.y0

        # Rendu uniquement du crop à zoom 2x
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

        print(f"CROP IMAGE: {pix.width}x{pix.height}px")

        # Convertir en OpenCV
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # Coordonnées du clic dans le crop
        click_px = int((req.click_x - crop_x) * zoom)
        click_py = int((req.click_y - crop_y) * zoom)

        print(f"CLICK IN CROP: {click_px},{click_py}")

        if not (0 <= click_px < w and 0 <= click_py < h):
            raise HTTPException(status_code=400, detail="Coordonnées hors crop")

        gray = cv2.cvtColo
