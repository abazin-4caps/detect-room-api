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

        # Détecter les murs (pixels sombres)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Fermer les portes
        kernel_close = np.ones((25, 25), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_dilate = np.ones((5, 5), np.uint8)
        closed = cv2.dilate(closed, kernel_dilate, iterations=3)

        # Flood fill pour isoler la pièce
        flood = cv2.bitwise_not(closed)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (req.click_x, req.click_y), 128)
        room_mask = np.uint8(flood == 128) * 255

        cv2.imwrite("/tmp/debug_crop.png", img)
        cv2.imwrite("/tmp/debug_binary.png", binary)
        cv2.imwrite("/tmp/debug_flood.png", room_mask)

        if cv2.countNonZero(room_mask) < 500:
            raise HTTPException(status_code=404, detail="Aucune pièce détectée")

        # Trouver la bounding box de la zone flood
        coords = cv2.findNonZero(room_mask)
        x, y, bw, bh = cv2.boundingRect(coords)

        # Chercher les vrais bords des murs dans chaque direction
        # en partant du centre et en cherchant les murs les plus proches
        cx, cy = req.click_x, req.click_y

        def find_wall(mask, start_x, start_y, dx, dy):
            """Cherche le premier pixel NON dans la pièce en partant du centre"""
            px, py = start_x, start_y
            while 0 <= px < w and 0 <= py < h:
                if mask[py, px] == 0:
                    return px - dx, py - dy
                px += dx
                py += dy
            return px - dx, py - dy

        # Trouver les 4 murs (haut, bas, gauche, droite)
        _, top_y = find_wall(room_mask, cx, cy, 0, -1)
        _, bot_y = find_wall(room_mask, cx, cy, 0, 1)
        left_x, _ = find_wall(room_mask, cx, cy, -1, 0)
        right_x, _ = find_wall(room_mask, cx, cy, 1, 0)

        print(f"MURS: top={top_y}, bot={bot_y}, left={left_x}, right={right_x}")

        # Affiner en cherchant les coins réels depuis les bords
        # Haut-gauche
        tl_x, _ = find_wall(room_mask, left_x, top_y + 5, -1, 0)
        tr_x, _ = find_wall(room_mask, right_x, top_y + 5, 1, 0)
        bl_x, _ = find_wall(room_mask, left_x, bot_y - 5, -1, 0)
        br_x, _ = find_wall(room_mask, right_x, bot_y - 5, 1, 0)

        _, tl_y = find_wall(room_mask, tl_x + 5, top_y, 0, -1)
        _, tr_y = find_wall(room_mask, tr_x - 5, top_y, 0, -1)
        _, bl_y = find_wall(room_mask, bl_x + 5, bot_y, 0, 1)
        _, br_y = find_wall(room_mask, br_x - 5, bot_y, 0, 1)

        # Construire le polygone orthogonal
        points = [
            [tl_x, tl_y],
            [tr_x, tr_y],
            [br_x, br_y],
            [bl_x, bl_y],
        ]

        print(f"POINTS: {points}")

        return {"polygon": points, "point_count": len(points)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
