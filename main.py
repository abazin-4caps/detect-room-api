from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
import os
import requests
import fitz
from collections import Counter

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

@app.post("/explore-vectors")
def explore_vectors(req: DetectRoomRequest):
    try:
        response = requests.get(req.pdf_url, timeout=30)
        pdf = fitz.open(stream=response.content, filetype="pdf")
        page = pdf[req.page_number - 1]
        paths = page.get_drawings()

        widths = Counter()
        segments = {}

        for p in paths:
            w = round(p.get('width') or 0, 2)
            widths[w] += 1
            col = str(p.get('color'))
            fill = str(p.get('fill'))
            for item in p.get('items', []):
                if item[0] == 'l':
                    x0, y0 = item[1].x, item[1].y
                    x1, y1 = item[2].x, item[2].y
                    length = round(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1)
                    key = f"w={w}|col={col}|fill={fill}"
                    segments.setdefault(key, []).append(length)

        summary = {}
        for key, lengths in segments.items():
            ls = sorted(lengths, reverse=True)
            summary[key] = {
                "nb_segments": len(ls),
                "longueur_max": ls[0],
                "longueur_moy": round(sum(ls) / len(ls), 1),
                "nb_gt100": len([l for l in ls if l > 100]),
                "nb_gt50":  len([l for l in ls if l > 50]),
                "nb_gt20":  len([l for l in ls if l > 20]),
                "top10": ls[:10],
            }

        return {
            "nb_paths_total": len(paths),
            "page": {"width": page.rect.width, "height": page.rect.height},
            "widths_distribution": dict(widths.most_common(20)),
            "segments_par_categorie": summary,
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug-vectors")
def debug_vectors(req: DetectRoomRequest):
    try:
        response = requests.get(req.pdf_url, timeout=30)
        pdf = fitz.open(stream=response.content, filetype="pdf")
        page = pdf[req.page_number - 1]

        min_width = 0.43
        min_length = 30

        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        wall_img = np.ones((pix.height, pix.width, 3), dtype=np.uint8) * 255

        paths = page.get_drawings()
        n_drawn = 0

        for p in paths:
            w = p.get('width') or 0
            col = p.get('color')
            if w < min_width:
                continue
            if col is None:
                continue

            for item in p.get('items', []):
                if item[0] != 'l':
                    continue
                x0, y0 = item[1].x, item[1].y
                x1, y1 = item[2].x, item[2].y
                length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if length < min_length:
                    continue

                pt1 = (int(x0 * zoom), int(y0 * zoom))
                pt2 = (int(x1 * zoom), int(y1 * zoom))

                if w >= 0.85:
                    color = (0, 0, 200)
                    thickness = 6
                elif w >= 0.43:
                    color = (0, 150, 255)
                    thickness = 4
                else:
                    color = (0, 200, 0)
                    thickness = 2

                cv2.line(wall_img, pt1, pt2, color, thickness)
                n_drawn += 1

        cv2.imwrite("/tmp/debug_vectors.png", wall_img)

        overlay = img_bgr.copy()
        mask = np.any(wall_img < 200, axis=2)
        overlay[mask] = wall_img[mask]
        cv2.imwrite("/tmp/debug_vectors_overlay.png", overlay)

        return {"segments_dessines": n_drawn, "zoom": zoom}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-room")
def detect_room(req: DetectRoomRequest):
    try:
        response = requests.get(req.pdf_url, timeout=30)
        pdf = fitz.open(stream=response.content, filetype="pdf")
        page = pdf[req.page_number - 1]
        page_rect = page.rect

        r = req.crop_radius
        clip = fitz.Rect(
            max(0, req.click_x - r), max(0, req.click_y - r),
            min(page_rect.width, req.click_x + r), min(page_rect.height, req.click_y + r),
        )
        crop_x, crop_y = clip.x0, clip.y0

        zoom = 4
        mat = fitz.Matrix(zoom, zoom).pretranslate(-clip.x0, -clip.y0)
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]

        cv2.imwrite("/tmp/debug_render.png", img_bgr)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10
        )
        cv2.imwrite("/tmp/debug_adaptive.png", adaptive)

        kernel_thin = np.ones((2, 2), np.uint8)
        cleaned = cv2.erode(adaptive, kernel_thin, iterations=1)

        kernel_thick = np.ones((4, 4), np.uint8)
        cleaned = cv2.dilate(cleaned, kernel_thick, iterations=1)

        cv2.imwrite("/tmp/debug_cleaned.png", cleaned)

        kernel_close = np.ones((25, 25), np.uint8)
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        cv2.imwrite("/tmp/debug_closed.png", closed)

        flood = cv2.bitwise_not(closed)
        click_px = int((req.click_x - crop_x) * zoom)
        click_py = int((req.click_y - crop_y) * zoom)

        click_px = max(1, min(click_px, img_w - 2))
        click_py = max(1, min(click_py, img_h - 2))

        mask = np.zeros((img_h + 2, img_w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(flood, mask, (click_px, click_py), 128,
                      loDiff=10, upDiff=10, flags=flags)

        room_mask = mask[1:-1, 1:-1]
        room_mask = np.uint8(room_mask == 255) * 255
        cv2.imwrite("/tmp/debug_flood.png", room_mask)

        if cv2.countNonZero(room_mask) < 500:
            raise HTTPException(status_code=404, detail="Aucune pièce détectée au point de clic")

        kernel_erode = np.ones((8, 8), np.uint8)
        room_eroded = cv2.erode(room_mask, kernel_erode, iterations=1)
        contours, _ = cv2.findContours(room_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise HTTPException(status_code=404, detail="Contour introuvable après érosion")

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.015 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        points = [
            [round(p[0][0] / zoom + crop_x, 2), round(p[0][1] / zoom + crop_y, 2)]
            for p in polygon
        ]

        debug_poly = img_bgr.copy()
        pts = np.array([[int(p[0][0]), int(p[0][1])] for p in polygon], dtype=np.int32)
        cv2.polylines(debug_poly, [pts], True, (0, 255, 0), 3)
        cv2.circle(debug_poly, (click_px, click_py), 10, (0, 0, 255), -1)
        cv2.imwrite("/tmp/debug_polygon.png", debug_poly)

        return {"polygon": points, "point_count": len(points)}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
