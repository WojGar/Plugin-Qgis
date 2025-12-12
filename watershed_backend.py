# watershed_backend.py
"""
Backend do wtyczki QGIS:
- bierze wycinek LAS/LAZ,
- filtruje wysoką roślinność (klasa 5),
- buduje CHM,
- wygładza,
- wykrywa lokalne maksima (LM),
- segmentuje koronami (watershed),
- przypisuje ID drzewa do punktów,
- zapisuje LAS z tree_id.
"""

import sys
from pathlib import Path

import laspy
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

CELL = 0.5
GAUSS_SIGMA = 2.5
MIN_DIST = 10
MIN_HEIGHT = 2.5


def load_las(src: Path):
    print("==> Wczytywanie LAS:", src)
    las = laspy.read(src)
    return las


def filter_high_vegetation(las):
    mask = las.classification == 5
    x = las.x[mask]
    y = las.y[mask]
    z = las.z[mask]
    print(f"Punktów klasy 5: {len(x)}")
    return x, y, z, mask


def build_chm(x, y, z, cell_size: float):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    nx = int(np.ceil((xmax - xmin) / cell_size)) + 1
    ny = int(np.ceil((ymax - ymin) / cell_size)) + 1
    print(f"CHM grid: nx={nx}, ny={ny}, cell={cell_size} m")

    chm = np.full((ny, nx), np.nan, dtype=float)
    ix = ((x - xmin) / cell_size).astype(int)
    iy = ((y - ymin) / cell_size).astype(int)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    for xi, yi, zi in zip(ix[valid], iy[valid], z[valid]):
        if np.isnan(chm[yi, xi]) or zi > chm[yi, xi]:
            chm[yi, xi] = zi

    chm = np.nan_to_num(chm, nan=0.0)
    return chm, xmin, ymin, nx, ny


def smooth_chm(chm, sigma: float):
    return gaussian_filter(chm, sigma=sigma)


def detect_treetops(chm_smooth, min_dist: int, min_height: float,
                    xmin: float, ymin: float, cell_size: float):
    coords = peak_local_max(
        chm_smooth,
        min_distance=min_dist,
        threshold_abs=min_height
    )
    print(f"Znaleziono potencjalnych drzew (LM): {len(coords)}")

    if len(coords) == 0:
        return coords, np.empty((0, 2), dtype=float)

    pred_x = xmin + coords[:, 1] * cell_size
    pred_y = ymin + coords[:, 0] * cell_size
    pred_xy = np.column_stack((pred_x, pred_y))
    return coords, pred_xy


def segment_crowns(chm_smooth, coords):
    markers = np.zeros_like(chm_smooth, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    labels = watershed(-chm_smooth, markers=markers, mask=(chm_smooth > 0))
    n_trees = labels.max()
    print("Liczba segmentów (drzew):", n_trees)
    return labels


def assign_tree_ids(las, labels, xmin, ymin, cell_size):
    tree_ids = np.zeros_like(las.x, dtype=np.uint32)

    ix_all = ((las.x - xmin) / cell_size).astype(int)
    iy_all = ((las.y - ymin) / cell_size).astype(int)

    nx = labels.shape[1]
    ny = labels.shape[0]
    valid_all = (ix_all >= 0) & (ix_all < nx) & (iy_all >= 0) & (iy_all < ny)

    mask_veg = las.classification == 5

    for idx in np.where(valid_all & mask_veg)[0]:
        ti = labels[iy_all[idx], ix_all[idx]]
        tree_ids[idx] = ti

    print("Punktów z przypisanym ID drzewa:", int((tree_ids > 0).sum()))
    return tree_ids

def process_wycinek(src: Path) -> dict:
    print("\n===================================")
    print(" PRZETWARZAM WYCINEK:", src.name)
    print("===================================\n")

    las = load_las(src)

    x, y, z, _ = filter_high_vegetation(las)

    if len(x) == 0:
        print("[WARN] Brak punktów klasy 5 – nic nie robię.")
        tree_ids = np.zeros_like(las.x, dtype=np.uint32)
        empty_xy = np.empty((0, 2), dtype=float)
        empty_ids = np.array([], dtype=np.uint32)

        return {
            "las": las,
            "tree_ids": tree_ids,
            "n_points": len(las.x),
            "n_trees": 0,
            "treetops_xy": empty_xy,
            "crown_points_xy": empty_xy,
            "crown_tree_ids": empty_ids,
        }

    chm, xmin, ymin, nx, ny = build_chm(x, y, z, CELL)
    chm_smooth = smooth_chm(chm, GAUSS_SIGMA)

    coords, pred_xy = detect_treetops(
        chm_smooth,
        min_dist=MIN_DIST,
        min_height=MIN_HEIGHT,
        xmin=xmin,
        ymin=ymin,
        cell_size=CELL
    )

    if len(coords) == 0:
        print("[WARN] Brak wykrytych czubków drzew – tree_id będą zerowe.")
        tree_ids = np.zeros_like(las.x, dtype=np.uint32)
        empty_xy = np.empty((0, 2), dtype=float)
        empty_ids = np.array([], dtype=np.uint32)

        return {
            "las": las,
            "tree_ids": tree_ids,
            "n_points": len(las.x),
            "n_trees": 0,
            "treetops_xy": pred_xy,
            "crown_points_xy": empty_xy,
            "crown_tree_ids": empty_ids,
        }

    labels = segment_crowns(chm_smooth, coords)
    tree_ids = assign_tree_ids(las, labels, xmin, ymin, CELL)

    n_trees = int(labels.max())
    print(f"[OK] Zakończono. Liczba drzew (segmentów): {n_trees}")

    mask = tree_ids > 0
    crown_xy = np.column_stack((las.x[mask], las.y[mask]))
    crown_ids = tree_ids[mask]

    return {
        "las": las,
        "tree_ids": tree_ids,
        "n_points": len(las.x),
        "n_trees": n_trees,
        "treetops_xy": pred_xy,
        "crown_points_xy": crown_xy,
        "crown_tree_ids": crown_ids,
    }



def _cli():
    if len(sys.argv) < 3:
        print("Użycie:")
        print("  python -m watershed_backend input.las output_trees.las")
        sys.exit(1)

    src = Path(sys.argv[1])

    process_wycinek(src)


if __name__ == "__main__":
    _cli()
