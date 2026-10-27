"""
Slice and extract clean alpha-channel 256x256 PNG sprites for EVOCURE Godot 4.7 Frontend.
Uses corner-seeded flood-fill + soft glow feathering so dark interior features (eyes, Benzene core)
remain 100% opaque while the outer #000000 backdrop becomes smoothly transparent.
"""

from __future__ import annotations

import pathlib
from collections import deque
import numpy as np
from PIL import Image

CREATURES_SRC = pathlib.Path(
    r"C:\Users\andre\.gemini\antigravity\brain\c5ad5edb-1edd-4faf-b8b7-8989bd27342c\chemical_creatures_spritesheet_1793108016539.jpg"
)
CELLS_SRC = pathlib.Path(
    r"C:\Users\andre\.gemini\antigravity\brain\c5ad5edb-1edd-4faf-b8b7-8989bd27342c\biological_cells_spritesheet_1793108040651.jpg"
)
OUT_DIR = pathlib.Path(r"e:\MA\EVOCURE\godot_frontend\assets\sprites")

CREATURE_NAMES = [
    "creature_water",
    "creature_methane",
    "creature_benzene",
    "creature_glucose",
    "creature_phosphate",
    "creature_lipid",
    "creature_dioxygen",
    "creature_amino",
]

CELL_NAMES = [
    "cell_healthy",
    "cell_tumor_pdac",
    "cell_tumor_resistant",
    "cell_immune",
    "cell_stroma_caf",
    "cell_tumor_nsclc",
    "cell_bacteria_amr",
    "cell_hepatocyte_masld",
]


def extract_cell_rgba(tile_rgb: np.ndarray, out_size: int = 256) -> Image.Image:
    h, w, _ = tile_rgb.shape
    lum = np.max(tile_rgb.astype(np.float32), axis=2)

    # Flood-fill from border pixels where max channel < 38 (outer black background + faint glow)
    visited = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if lum[y, x] < 42.0:
                visited[y, x] = True
                bg_mask[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not visited[y, x] and lum[y, x] < 42.0:
                visited[y, x] = True
                bg_mask[y, x] = True
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                if lum[ny, nx] < 36.0:
                    bg_mask[ny, nx] = True
                    q.append((ny, nx))

    alpha = np.full((h, w), 255.0, dtype=np.float32)
    # In outer background region, keep subtle bioluminescent halo proportional to brightness
    bg_lum = lum[bg_mask]
    halo_alpha = np.clip((bg_lum - 8.0) * (255.0 / 30.0), 0.0, 220.0)
    alpha[bg_mask] = halo_alpha

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = tile_rgb
    rgba[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    # Crop tight bounding box around non-zero alpha and center in square out_size x out_size
    ys, xs = np.where(rgba[:, :, 3] > 18)
    pil_tile = Image.fromarray(rgba, mode="RGBA")
    if len(xs) > 10 and len(ys) > 10:
        pad = 10
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
        pil_tile = pil_tile.crop((x0, y0, x1, y1))

    cw, ch = pil_tile.size
    scale = min((out_size - 20) / max(1, cw), (out_size - 20) / max(1, ch))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = pil_tile.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    canvas.paste(resized, ((out_size - nw) // 2, (out_size - nh) // 2), resized)
    return canvas


def slice_grid(sheet_path: pathlib.Path, names: list[str], cols: int = 4, rows: int = 2) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(sheet_path).convert("RGB")
    arr = np.asarray(img)
    H, W, _ = arr.shape
    cw = W // cols
    ch = H // rows

    idx = 0
    for r in range(rows):
        for c in range(cols):
            tile = arr[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
            sprite = extract_cell_rgba(tile, out_size=256)
            out_file = OUT_DIR / f"{names[idx]}.png"
            sprite.save(out_file, "PNG")
            print(f"Saved {out_file.name} ({sprite.size})")
            idx += 1


if __name__ == "__main__":
    slice_grid(CREATURES_SRC, CREATURE_NAMES)
    slice_grid(CELLS_SRC, CELL_NAMES)
