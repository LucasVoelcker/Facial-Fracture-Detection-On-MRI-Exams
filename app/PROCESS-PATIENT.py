"""
PIPELINE: YOLOv7 (detecção) -> CROP+padding BASE -> ResNet (classificação por janela)

- Etapa 1: replica inference-yolo-patient.py: salva crops BASE em OUTPUT_CROPS_DIR.
- Etapa 2: replica inference-resnet-patient-v2.py: lê os crops gerados (OUTPUT_CROPS_DIR),
          agrupa por imagem original, classifica e gera TXT final.

Requisitos:
pip install opencv-python pillow albumentations torch torchvision
"""

from __future__ import annotations

# =========================================================
# SUPRESSAO DE WARNINGS / LOGS (ANTES DOS IMPORTS PESADOS)
# =========================================================
import os
import warnings
import logging

warnings.filterwarnings("ignore")  # suprime warnings do Python (DeprecationWarning, UserWarning, etc.)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # inofensivo aqui, mas ajuda se algo puxar TF indiretamente
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from PIL import Image
from torchvision import models, transforms

# PyTorch: suprime alguns warnings internos (quando disponíveis)
try:
    torch.set_warn_always(False)
except Exception:
    pass

# =========================================================
# OVERRIDE DE INPUT_DIR VIA ARGUMENTO (GUI / CLI)
# =========================================================
# if len(sys.argv) > 1:
#     INPUT_DIR = Path(sys.argv[1])

# =========================================================
# 1) CONFIG: EDITE AQUI (YOLO)
# =========================================================

# --- YOLOv7 ---
YOLO_WEIGHTS      = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\treinamentos\2026-10-01-yolov7\best.pt")
YOLO_IMG_SIZE     = 640
YOLO_CONF_THRES   = 0.5
YOLO_IOU_THRES    = 0.45
YOLO_DEVICE       = "cpu"       # "cuda" ou "cpu"
YOLO_CLASSES      = None        # ex: [0,1,2,3] ou None (todas)
YOLO_AGNOSTIC_NMS = False

CROP_MARGIN_PX = 20

# --- Pad/Resize do crop (base) ---
CROP_OUT_SIZE = 224  # gera crop final 224x224 com padding 0 (estilo "__base")

# --- Pastas ---
DEFAULT_INPUT_DIR = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\inputs\test-patient\2025-Paciente-30")

if len(sys.argv) > 1:
    INPUT_DIR = Path(sys.argv[1])
else:
    INPUT_DIR = DEFAULT_INPUT_DIR

OUTPUT_DIR       = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\outputs\test-patient\2025-Paciente-30")
OUTPUT_CROPS_DIR = OUTPUT_DIR / "crops"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================================================
# 2) CONFIG: EDITE AQUI (RESNET)
# =========================================================

CKPT_BY_REGION = {
    "MAND":  Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\treinamentos\best-dataset-resnet-2026-12-01(MAND)_v4.pth"),
    "MED":   Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\treinamentos\best-dataset-resnet-2026-14-01(MED)_v1.pth"),
    "NARIZ": Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\treinamentos\best-dataset-resnet-2026-11-01-v5.pth"),
    "SUP":   Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\treinamentos\best-dataset-resnet-2026-14-01(SUP)_v1.pth"),
}

DEVICE    = "cpu"  # "cuda" ou "cpu"
PREFER_YOLO_CLASS_FOR_REGION = True

# IMPORTANTE: aqui a gente força o ResNet a ler os crops gerados na etapa YOLO:
CROPS_DIR = OUTPUT_CROPS_DIR

RESULTS_TXT = OUTPUT_DIR / "patient-result-todos.txt"

ONLY_YOLO_CLASS      = None      # NomeDaClasse ou None
FRACTURE_CLASS_NAME  = "com_fratura"
FRACTURE_PROB_THRES  = 0.5

WINDOW_SIZE   = 10
MIN_FRACTURES = 5

# =========================================================
# YOLOv7 repo (CAMINHO ABSOLUTO)
# =========================================================
YOLOV7_REPO_DIR = Path(r"C:\Users\lucas\OneDrive\Documentos\yolov7-dir\yolov7").resolve()
if not YOLOV7_REPO_DIR.exists():
    raise RuntimeError(f"YOLOv7 repo não encontrado em: {YOLOV7_REPO_DIR}")

sys.path.insert(0, str(YOLOV7_REPO_DIR))

# =========================================================
# Import YOLOv7 internals
# =========================================================
from models.experimental import attempt_load  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging  # noqa: E402
from utils.torch_utils import select_device, time_synchronized  # noqa: E402
from utils.datasets import letterbox  # noqa: E402

# =========================================================
# Utilitários (YOLO)
# =========================================================

def list_images_sorted(input_dir: Path) -> List[Path]:
    paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort(key=lambda x: x.name.lower())
    return paths

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CROPS_DIR.mkdir(parents=True, exist_ok=True)

def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Falha ao ler imagem: {path}")
    return img

def crop_with_margin(im_bgr: np.ndarray, xyxy: Tuple[int,int,int,int], margin: int) -> np.ndarray | None:
    x1, y1, x2, y2 = xyxy
    H, W = im_bgr.shape[:2]
    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)
    cx2 = min(W, x2 + margin)
    cy2 = min(H, y2 + margin)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return im_bgr[cy1:cy2, cx1:cx2].copy()

def save_unique(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        ok = cv2.imwrite(str(path), img)
        if not ok:
            raise RuntimeError(f"Falha ao salvar: {path}")
        return path

    k = 2
    while True:
        new_path = path.with_name(f"{path.stem}-{k}{path.suffix}")
        if not new_path.exists():
            ok = cv2.imwrite(str(new_path), img)
            if not ok:
                raise RuntimeError(f"Falha ao salvar: {new_path}")
            return new_path
        k += 1

def make_base_like_dataset(crop_bgr: np.ndarray, out_size: int = 224) -> np.ndarray:
    """
    Reproduz o 'base' do dataset:
    - grayscale (1 canal)
    - LongestMaxSize(out_size)
    - PadIfNeeded até out_size x out_size com padding 0
    Retorna uint8 (H,W).
    """
    if crop_bgr.ndim == 3:
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop_bgr

    tfm = A.Compose([
        A.LongestMaxSize(max_size=out_size, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(
            min_height=out_size, min_width=out_size,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
        ),
    ])
    return tfm(image=crop_gray)["image"]

@torch.no_grad()
def run_yolo_and_crops(
    model,
    device,
    half: bool,
    img_path: Path,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    classes=None,
    agnostic_nms: bool=False,
    margin_px: int=20,
) -> List[Tuple[Path, str, float]]:
    """
    Retorna lista de (crop_path_salvo_base, yolo_cls_name, yolo_conf)
    """
    im0 = read_image_bgr(img_path)
    im0_orig = im0.copy()

    img = letterbox(im0, imgsz, stride=int(model.stride.max()), auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)

    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.half() if half else img_t.float()
    img_t /= 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img_t, augment=False)[0]
    t2 = time_synchronized()

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

    det = pred[0]
    crops_info: List[Tuple[Path, str, float]] = []

    if det is None or len(det) == 0:
        print(f"[YOLO] {img_path.name}: sem detecção. ({1e3*(t2-t1):.1f}ms infer, {1e3*(t3-t2):.1f}ms nms)")
        return crops_info

    det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], im0.shape).round()
    names = model.module.names if hasattr(model, "module") else model.names

    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = map(int, xyxy)

        crop_raw = crop_with_margin(im0_orig, (x1, y1, x2, y2), margin_px)
        if crop_raw is None:
            continue

        crop_base = make_base_like_dataset(crop_raw, out_size=CROP_OUT_SIZE)

        cls_i = int(cls)
        cls_name = names[cls_i] if isinstance(names, (list, tuple)) else str(cls_i)
        conf_f = float(conf)
        conf_str = f"{conf_f:.2f}".replace(".", "_")

        out_name = f"{img_path.stem}-{cls_name}-{conf_str}.png"
        out_path = OUTPUT_CROPS_DIR / out_name
        saved_path = save_unique(out_path, crop_base)

        crops_info.append((saved_path, cls_name, conf_f))

    print(f"[YOLO] {img_path.name}: {len(crops_info)} crops BASE salvos. ({1e3*(t2-t1):.1f}ms infer, {1e3*(t3-t2):.1f}ms nms)")
    return crops_info

# =========================================================
# Utilitários (RESNET)
# =========================================================

CROP_RE = re.compile(
    r"^(?P<orig>.+?)-(?P<yclass>[^-]+)-(?P<conf>[\d_]+)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE
)

def list_imgs_sorted(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda x: x.name.lower())
    return paths

def patient_has_fracture_by_window(image_labels: List[str], window_size: int, min_fractures: int) -> Tuple[bool, int]:
    if len(image_labels) < window_size:
        return (False, -1)
    for i in range(0, len(image_labels) - window_size + 1):
        window = image_labels[i:i + window_size]
        cnt = sum(1 for x in window if x == FRACTURE_CLASS_NAME)
        if cnt >= min_fractures:
            return (True, i)
    return (False, -1)

def load_ckpt_and_classes(ckpt_path: Path, device: str) -> Tuple[dict, List[str]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get("classes", None)
    if classes is None:
        raise KeyError("Checkpoint não possui a chave 'classes'.")
    if not isinstance(classes, (list, tuple)) or not all(isinstance(c, str) for c in classes):
        raise TypeError(f"ckpt['classes'] inválido: {type(classes)} -> {classes}")
    return ckpt, list(classes)

def build_model(num_classes: int, ckpt: dict, device: str) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = ckpt.get("model_state", None)
    if state is None:
        state = ckpt.get("state_dict", ckpt)

    cleaned = {}
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(k, str) and k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v
    else:
        cleaned = state

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model

PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # garante 3 canais p/ ResNet
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # padrão ImageNet (ok p/ baseline)
                         std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_path(model: torch.nn.Module, img_path: Path, device: str, classes: List[str], fracture_idx: int) -> Tuple[str, float]:
    img = Image.open(img_path).convert("RGB")
    x = PREPROCESS(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    pred_name = classes[pred_idx]
    fracture_prob = float(probs[fracture_idx].item())
    return pred_name, fracture_prob

def detect_region_from_text(text: str) -> str | None:
    up = text.upper()
    for key in ("MAND", "MED", "NARIZ", "SUP"):
        if key in up:
            return key
    return None

def load_all_resnets(ckpt_by_region: Dict[str, Path], device: str):
    bundle = {}
    for region, ckpt_path in ckpt_by_region.items():
        ckpt, classes = load_ckpt_and_classes(ckpt_path, device)

        if FRACTURE_CLASS_NAME not in classes:
            raise ValueError(f"[{region}] classe '{FRACTURE_CLASS_NAME}' não existe em ckpt['classes']: {classes}")

        fracture_idx = classes.index(FRACTURE_CLASS_NAME)
        model = build_model(num_classes=len(classes), ckpt=ckpt, device=device)

        bundle[region] = {
            "model": model,
            "classes": classes,
            "fracture_idx": fracture_idx,
            "ckpt_path": ckpt_path,
            "epoch": ckpt.get("epoch", "N/A"),
        }
    return bundle

def choose_region_for_crop(crop_path: Path, yclass: str) -> str:
    if PREFER_YOLO_CLASS_FOR_REGION:
        r = detect_region_from_text(yclass)
        if r is not None:
            return r

    r = detect_region_from_text(crop_path.name)
    if r is not None:
        return r

    raise ValueError(f"Não consegui detectar região para o crop: {crop_path.name} (yclass={yclass})")

# =========================================================
# MAIN (PIPELINE)
# =========================================================

def main():
    ensure_dirs()

    # =========================
    # (1) YOLO -> gerar crops
    # =========================
    imgs = list_images_sorted(INPUT_DIR)
    if not imgs:
        print(f"Nenhuma imagem encontrada em: {INPUT_DIR}")
        return

    # silencia logs do YOLOv7 (não são warnings, são prints/loggers)
    set_logging()
    logging.getLogger("yolov7").setLevel(logging.ERROR)
    logging.getLogger("utils").setLevel(logging.ERROR)

    yolo_device = select_device(YOLO_DEVICE)
    half = yolo_device.type != "cpu"

    model_yolo = attempt_load(str(YOLO_WEIGHTS), map_location=yolo_device)
    stride = int(model_yolo.stride.max())
    imgsz = check_img_size(YOLO_IMG_SIZE, s=stride)

    if half:
        model_yolo.half()

    if yolo_device.type != "cpu":
        model_yolo(torch.zeros(1, 3, imgsz, imgsz).to(yolo_device).type_as(next(model_yolo.parameters())))

    total_crops = 0
    t0 = time.time()

    for idx, img_path in enumerate(imgs, start=1):
        print(f"\n[YOLO {idx}/{len(imgs)}] Processando: {img_path.name}")
        crops = run_yolo_and_crops(
            model=model_yolo,
            device=yolo_device,
            half=half,
            img_path=img_path,
            imgsz=imgsz,
            conf_thres=YOLO_CONF_THRES,
            iou_thres=YOLO_IOU_THRES,
            classes=YOLO_CLASSES,
            agnostic_nms=YOLO_AGNOSTIC_NMS,
            margin_px=CROP_MARGIN_PX,
        )
        total_crops += len(crops)

    print("\n" + "=" * 80)
    print(f"[YOLO] Finalizado. Total de crops salvos: {total_crops}")
    print(f"[YOLO] Crops (BASE) em: {OUTPUT_CROPS_DIR}")
    print("=" * 80)

    # =========================
    # (2) RESNET -> ler OUTPUT_CROPS_DIR e classificar
    # =========================
    if not CROPS_DIR.exists():
        raise RuntimeError(f"CROPS_DIR não existe: {CROPS_DIR}")

    crops = list_imgs_sorted(CROPS_DIR)
    if not crops:
        print(f"Nenhuma imagem encontrada em: {CROPS_DIR}")
        return

    run_device = DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"

    models_bundle = load_all_resnets(CKPT_BY_REGION, run_device)

    print("\n[RESNET] Modelos carregados:")
    for region, info in models_bundle.items():
        print(f"  - {region}: ckpt={info['ckpt_path'].name} | epoch={info['epoch']} | classes={info['classes']}")

    groups: Dict[str, List[Tuple[Path, str, float]]] = {}
    skipped = 0

    for p in crops:
        m = CROP_RE.match(p.name)
        if not m:
            skipped += 1
            continue

        orig = m.group("orig")
        yclass = m.group("yclass")
        conf_str = m.group("conf").replace("_", ".")
        try:
            yconf = float(conf_str)
        except ValueError:
            yconf = -1.0

        if ONLY_YOLO_CLASS is not None and yclass != ONLY_YOLO_CLASS:
            continue

        groups.setdefault(orig, []).append((p, yclass, yconf))

    if not groups:
        print("[RESNET] Nenhum crop passou pelo filtro/parse. Confira ONLY_YOLO_CLASS e o padrão de nome.")
        return

    orig_names = sorted(groups.keys(), key=lambda s: s.lower())

    results_lines: List[str] = []
    image_level_labels: List[str] = []
    image_best_fracture_prob: Dict[str, float] = {}

    image_fracture_segments: Dict[str, set] = {}

    t1 = time.time()

    results_lines.append("Checkpoints por região:")
    for region, info in models_bundle.items():
        results_lines.append(f"  - {region}: {info['ckpt_path']} | epoch={info['epoch']} | classes={info['classes']}")

    results_lines.append(f"CROPS_DIR: {CROPS_DIR}")
    results_lines.append(f"ONLY_YOLO_CLASS: {ONLY_YOLO_CLASS}")
    results_lines.append(f"FRACTURE_CLASS_NAME: {FRACTURE_CLASS_NAME}")

    results_lines.append(f"FRACTURE_PROB_THRES: {FRACTURE_PROB_THRES}")
    results_lines.append(f"WINDOW: size={WINDOW_SIZE}, min_fractures={MIN_FRACTURES}")
    results_lines.append("-" * 80)

    for i, orig in enumerate(orig_names, start=1):
        crops_list = groups[orig]
        any_fracture = False
        best_fracture_prob = 0.0

        fracture_segments_this_image = set()

        results_lines.append(f"\n[{i}/{len(orig_names)}] {orig} (crops={len(crops_list)})")

        for crop_path, yclass, yconf in crops_list:
            region = choose_region_for_crop(crop_path, yclass)
            info = models_bundle[region]

            pred_name, fracture_prob = predict_path(
                model=info["model"],
                img_path=crop_path,
                device=run_device,
                classes=info["classes"],
                fracture_idx=info["fracture_idx"],
            )

            # =========================================================
            # >>> NOVO: PRINT NO TERMINAL PARA CADA IMAGEM (CROP) DA RESNET
            # =========================================================
            print(
                f"[RESNET] crop={crop_path.name} | region={region} | "
                f"resnet={pred_name} | fracture_prob={fracture_prob:.4f} | "
                f"yolo={yclass} | yolo_conf={yconf:.4f}"
            )

            best_fracture_prob = max(best_fracture_prob, fracture_prob)

            if pred_name == FRACTURE_CLASS_NAME and fracture_prob >= FRACTURE_PROB_THRES:
                any_fracture = True
                fracture_segments_this_image.add(region)

            results_lines.append(
                f"  - {crop_path.name}\tregion={region}\tckpt={info['ckpt_path'].name}\t"
                f"resnet={pred_name}\tfracture_prob={fracture_prob:.4f}\t"
                f"yolo={yclass}\tyolo_conf={yconf:.4f}"
            )

        image_label = FRACTURE_CLASS_NAME if any_fracture else "sem_fratura"
        image_level_labels.append(image_label)
        image_best_fracture_prob[orig] = best_fracture_prob
        results_lines.append(f"  => IMAGE_LABEL={image_label}\tbest_fracture_prob={best_fracture_prob:.4f}")

        image_fracture_segments[orig] = fracture_segments_this_image
        if fracture_segments_this_image:
            results_lines.append(f"  => FRAC_SEGMENTS={sorted(fracture_segments_this_image)}")

    has_fracture_patient, win_start = patient_has_fracture_by_window(
        image_level_labels,
        window_size=WINDOW_SIZE,
        min_fractures=MIN_FRACTURES
    )

    if has_fracture_patient:
        diag_msg = f"DIAGNÓSTICO DO PACIENTE: COM FRATURA (existe janela de {WINDOW_SIZE} com >={MIN_FRACTURES} fraturas)"
    else:
        diag_msg = f"DIAGNÓSTICO DO PACIENTE: SEM FRATURA (não existe janela de {WINDOW_SIZE} com >={MIN_FRACTURES} fraturas)"

    results_lines.append("\n" + "=" * 80)
    results_lines.append(diag_msg)
    results_lines.append("=" * 80)

    # Segmentos com fratura (com base na janela que disparou "COM FRATURA")
    if has_fracture_patient and win_start >= 0:
        win_origs = orig_names[win_start:win_start + WINDOW_SIZE]
        segments_in_window = set()
        for o in win_origs:
            segments_in_window.update(image_fracture_segments.get(o, set()))

        pretty = {"MAND": "mandíbula", "MED": "terço médio", "NARIZ": "nariz", "SUP": "terço superior"}
        seg_list = [pretty.get(s, s) for s in sorted(segments_in_window)]

        results_lines.append("\nSEGMENTO(S) COM FRATURA DETECTADA NA JANELA:")
        if seg_list:
            results_lines.append("  - " + ", ".join(seg_list))
        else:
            results_lines.append("  (não foi possível determinar)")
    else:
        results_lines.append("\nSEGMENTO(S) COM FRATURA DETECTADA NA JANELA:")
        results_lines.append("  (nenhum — pasta classificada como SEM FRATURA)")

    RESULTS_TXT.write_text("\n".join(results_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("[RESNET]", diag_msg)
    print(f"[RESNET] Resultados salvos em: {RESULTS_TXT}")
    print(f"[RESNET] Tempo ResNet: {time.time() - t1:.2f}s")
    print(f"[PIPELINE] Tempo total: {time.time() - t0:.2f}s")
    if skipped:
        # isso NÃO é warning, é um print controlado seu
        print(f"[RESNET] Aviso: {skipped} crops ignorados (nome não bateu com o padrão esperado).")
    print("=" * 80)

if __name__ == "__main__":
    main()
