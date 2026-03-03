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
    crop_radius: float = 300

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

        pdf = fitz.open(stream=response.content, filetype="pdf")
        page = pdf[req.page_number - 1]
        page_rect = page.rect

        # Zone de crop autour du clic
        r = req.crop_radius
        clip = fitz.Rect(
            max(0, req.click_x - r),
            max(0, req.click_y - r),
            min(page_rect.width, req.click_x + r),
            min(page_rect.height, req.click_y + r),
        )
        crop_x = clip.x0
        crop_y = clip.y0

        zoom = 3
        img_w = int(clip.width * zoom)
        img_h = int(clip.height * zoom)

        print(f"CROP: {img_w}x{img_h}px")

        # Créer une image noire
        walls_img = np.zeros((img_h, img_w), dtype=np.uint8)

        # Extraire les paths vectoriels de la page
        paths = page.get_drawings()
        print(f"PATHS TROUVÉS: {len(paths)}")

        # Analyser les épaisseurs pour trouver le seuil des murs
        widths = [p.get("width", 0) for p in paths if p.get("width", 0) > 0]
        if widths:
            widths_sorted = sorted(set(widths))
            print(f"ÉPAISSEURS: {widths_sorted[:20]}")
            # Prendre les traits dont l'épaisseur est dans le top 30%
            threshold_width = np.percentile(widths, 70)
            print(f"SEUIL ÉPAISSEUR: {threshold_width}")
        else:
            threshold_width = 0.5

        # Dessiner uniquement les murs (traits épais) sur l'image
        wall_count = 0
        for path in paths:
            width = path.get("width", 0)
            if width < threshold_width:
                continue

            # Vérifier que le path est dans la zone de crop
            rect = path.get("rect")
            if rect is None:
                continue
            if not clip.intersects(rect):
                continue

            # Dessiner les segments du path
            for item in path.get("items", []):
                if item[0] == "l":  # ligne
                    p1, p2 = item[1], item[2]
                    x1 = int((p1.x - crop_x) * zoom)
                    y1 = int((p1.y - crop_y) * zoom)
                    x2 = int((p2.x - crop_x) * zoom)
                    y2 = int((p2.y - crop_y) * zoom)
                    thickness = max(3, int(width * zoom))
                    cv2.line(walls_img, (x1, y1), (x2, y2), 255, thickness)
                    wall_count += 1
                elif item[0] == "re":  # rectangle
                    rect2 = item[1]
                    x1 = int((rect2.x0 - crop_x) * zoom)
                    y1 = int((rect2.y0 - crop_y) * zoom)
                    x2 = int((rect2.x1 - crop_x) * zoom)
                    y2 = int((rect2.y1 - crop_y) * zoom)
                    thickness = max(3, int(width * zoom))
                    cv2.rectangle(walls_img, (x1, y1), (x2, y2), 255, thickness)
                    wall_count += 1
                elif item[0] == "c":  # courbe de bezier
                    p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                    x1 = int((p1.x - crop_x) * zoom)
                    y1 = int((p1.y - crop_y) * zoom)
                    x2 = int((p4.x - crop_x) * zoom)
                    y2 = int((p4.y - crop_y) * zoom)
                    thickness = max(3, int(width * zoom))
                    cv2.line(walls_img, (x1, y1), (x2, y2), 255, thickness)
                    wall_count += 1

        print(f"SEGMENTS DESSINÉS: {wall_count}")

        # Fermer les portes
        kernel_close = np.ones((20, 20), np.uint8)
        closed = cv2.morphologyEx(walls_img, cv2.MORPH_CLOSE, kernel_close)

        # Debug
        cv2.imwrite("/tmp/debug_walls.png", walls_img)
        cv2.imwrite("/tmp/debug_closed.png", closed)

        # Flood fill
        flood = cv2.bitwise_not(closed)
        click_px = int((req.click_x - crop_x) * zoom)
        click_py = int((req.click_y - crop_y) * zoom)

        mask = np.zeros((img_h + 2, img_w + 2), np.uint8)
        cv2.floodFill(flood, mask, (click_px, click_py), 128)
        room_mask = np.uint8(flood == 128) * 255

        cv2.imwrite("/tmp/debug_flood.png", room_mask)

        if cv2.countNonZero(room_mask) < 500:
            raise HTTPException(status_code=404, detail="Aucune pièce détectée")

        # Simplification contour
        kernel_erode = np.ones((6, 6), np.uint8)
        room_eroded = cv2.erode(room_mask, kernel_erode, iterations=1)

        contours, _ = cv2.findContours(room_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise HTTPException(status_code=404, detail="Aucun contour trouvé")

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Reconvertir en coordonnées PDF
        points = [
            [round(p[0][0] / zoom + crop_x, 2), round(p[0][1] / zoom + crop_y, 2)]
            for p in polygon
        ]

        print(f"POINTS: {len(points)}")
        return {"polygon": points, "point_count": len(points)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
