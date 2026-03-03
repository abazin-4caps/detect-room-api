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

        print(f"IMAGE REÇUE: {len(img_bytes)} bytes")
        print(f"IMAGE SHAPE: {img.shape if img is not None else 'None'}")
        print(f"CLICK: x={req.click_x}, y={req.click_y}")

        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")

        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Seuillage : murs sombres → blanc, espace vide → noir
        # C'est l'inverse de avant !
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Fermer les petits gaps
        kernel_close = np.ones((25, 25), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # Fermer aussi les portes en dilatant davantage
        kernel_dilate = np.ones((5, 5), np.uint8)
        closed = cv2.dilate(closed, kernel_dilate, iterations=3)

        # Sauvegarder debug
        cv2.imwrite("/tmp/debug_crop.png", img)
        cv2.imwrite("/tmp/debug_binary.png", binary)
        cv2.imwrite("/tmp/debug_closed.png", closed)

        # Flood fill sur l'image INVERSÉE
        # On travaille sur une copie où murs=blanc, vide=noir
        flood_input = cv2.bitwise_not(closed)  # murs=noir, vide=blanc
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood_input, mask, (req.click_x, req.click_y), 128)

        room_mask = np.uint8(flood_input == 128) * 255
        cv2.imwrite("/tmp/debug_flood.png", room_mask)

        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"CONTOURS TROUVÉS: {len(contours)}")

        if not contours:
            raise HTTPException(status_code=404, detail="Aucune pièce détectée")

        def dist_to_click(c):
            M = cv2.moments(c)
            if M["m00"] == 0:
                return float('inf')
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (cx - req.click_x)**2 + (cy - req.click_y)**2

        contour = min(contours, key=dist_to_click)

        print(f"SURFACE CONTOUR: {cv2.contourArea(contour)}")

        if cv2.contourArea(contour) < 500:
            raise HTTPException(status_code=404, detail="Zone trop petite")

        epsilon = 0.008 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(p[0][0]), int(p[0][1])] for p in polygon]

        print(f"POINTS POLYGONE: {len(points)}")

        return {"polygon": points, "point_count": len(points)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
