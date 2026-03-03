from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectRoomRequest(BaseModel):
    image_base64: str
    click_x: int
    click_y: int

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
        img_bytes = base64.b64decode(req.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")

        h, w = img.shape[:2]
        print(f"IMAGE: {w}x{h}, CLICK: {req.click_x},{req.click_y}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Seuillage : garder seulement les traits sombres
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Supprimer les traits fins (meubles, cotations, texte)
        # Ne garde que les murs épais
        kernel_thin = np.ones((2, 2), np.uint8)
        walls_only = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_thin, iterations=1)

        # Fermer les portes
        kernel_close = np.ones((20, 20), np.uint8)
        closed = cv2.morphologyEx(walls_only, cv2.MORPH_CLOSE, kernel_close)

        # Sauvegarder debug
        cv2.imwrite("/tmp/debug_crop.png", img)
        cv2.imwrite("/tmp/debug_binary.png", binary)
        cv2.imwrite("/tmp/debug_walls.png", walls_only)
        cv2.imwrite("/tmp/debug_closed.png", closed)

        # Flood fill pour isoler la pièce
        flood = cv2.bitwise_not(closed)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (req.click_x, req.click_y), 128)
        room_mask = np.uint8(flood == 128) * 255

        cv2.imwrite("/tmp/debug_flood.png", room_mask)

        if cv2.countNonZero(room_mask) < 500:
            raise HTTPException(status_code=404, detail="Aucune pièce détectée")

        # Éroder légèrement pour rentrer dans les murs
        kernel_erode = np.ones((10, 10), np.uint8)
        room_eroded = cv2.erode(room_mask, kernel_erode, iterations=1)

        # Simplification agressive pour ne garder que les coins
        contours, _ = cv2.findContours(room_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise HTTPException(status_code=404, detail="Aucun contour trouvé")

        contour = max(contours, key=cv2.contourArea)

        epsilon = 0.05 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(p[0][0]), int(p[0][1])] for p in polygon]

        print(f"POINTS POLYGONE: {len(points)} -> {points}")

        return {"polygon": points, "point_count": len(points)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
