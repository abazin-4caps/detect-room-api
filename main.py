from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import base64

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

@app.post("/detect-room")
def detect_room(req: DetectRoomRequest):
    try:
        # Décoder l'image
        img_bytes = base64.b64decode(req.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")

        h, w = img.shape[:2]

        # Vérifier que le clic est dans l'image
        if not (0 <= req.click_x < w and 0 <= req.click_y < h):
            raise HTTPException(status_code=400, detail="Coordonnées hors image")

        # Binarisation : murs en noir, espace en blanc
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Fermer les gaps (portes, ouvertures)
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Flood fill depuis le point cliqué
        flood = closed.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (req.click_x, req.click_y), 128)

        # Isoler la zone remplie
        room_mask = np.uint8(flood == 128) * 255

        # Extraire le contour
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise HTTPException(status_code=404, detail="Aucune pièce détectée")

        # Prendre le plus grand contour
        contour = max(contours, key=cv2.contourArea)

        # Simplifier en polygone
        epsilon = 0.005 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Convertir en liste de points
        points = [[int(p[0][0]), int(p[0][1])] for p in polygon]

        return {"polygon": points, "point_count": len(points)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
