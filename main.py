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
        img_bytes = base64.b64decode(req.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")

        h, w = img.shape[:2]

        if not (0 <= req.click_x < w and 0 <= req.click_y < h):
            raise HTTPException(status_code=400, detail="Coordonnées hors image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Seuillage adaptatif — meilleur sur plans architecte
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=4
        )

        # Fermer les portes avec un kernel plus grand
        kernel_close = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # Dilater les murs pour bien fermer les gaps
        kernel_dilate = np.ones((3, 3), np.uint8)
        closed = cv2.dilate(closed, kernel_dilate, iterations=2)

        # Flood fill
        flood = closed.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (req.click_x, req.click_y), 128)

        room_mask = np.uint8(flood == 128) * 255

        # Filtrer les petites zones parasites
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise HTTPException(status_code=404, detail="Aucune pièce détectée")

        # Prendre le contour le plus proche du point cliqué
        def dist_to_click(c):
            M = cv2.moments(c)
            if M["m00"] == 0:
                return float('inf')
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (cx - req.click_x)**2 + (cy - req.click_y)**2

        contour = min(contours, key=dist_to_click)

        # Surface minimale pour éviter le bruit
        if cv2.contourArea(contour) < 500:
            raise HTTPException(status_code=404, detail="Zone trop petite")

        epsilon = 0.008 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(p[0][0]), int(p[0][1])] for p in polygon]

        return {"polygon": points, "point_count": len(points)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
