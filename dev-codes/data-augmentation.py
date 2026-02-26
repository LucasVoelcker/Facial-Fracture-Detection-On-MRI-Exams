import os
from pathlib import Path
import cv2
import numpy as np

# pip install albumentations opencv-python
import albumentations as A

# =========================
# CONFIG
# =========================
INPUT_DIR = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\datasets\dataset-resnet-2026-11-01-v5(NARIZ)\test\com_fratura")   # <-- MUDE AQUI
OUTPUT_DIR = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\datasets\dataset-resnet-2026-11-01-v5(NARIZ)\test\com_fratura\saida_aug")            # <-- MUDE AQUI

AUGS_PER_IMAGE = 0          # quantas versões criar por imagem
OUT_SIZE = 224              # saída final (224 para ResNet "padrão")
SEED = 123                  # reprodutibilidade (opcional)

# Extensões aceitas
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_augmentations(out_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=out_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(
                min_height=out_size, min_width=out_size,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
            ),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-7, 7),
                #shear={"x": (-2, 2), "y": (-2, 2)},
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                p=0.9
            ),
            A.HorizontalFlip(p=0.75),
            A.RandomGamma(gamma_limit=(90, 110), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.08,
                p=0.5
            ),
        ],
        p=1.0
    )


def imread_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Falha ao ler imagem: {path}")
    return img


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    np.random.seed(SEED)

    ensure_dir(OUTPUT_DIR)

    aug = build_augmentations(OUT_SIZE)

    files = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        print(f"Nenhuma imagem encontrada em: {INPUT_DIR}")
        return

    print(f"Encontradas {len(files)} imagens em {INPUT_DIR}")
    print(f"Gerando {AUGS_PER_IMAGE} augmentations por imagem em {OUTPUT_DIR}...")

    total_written = 0

    for img_path in files:
        img = imread_grayscale(img_path)  # (H,W) uint8

        stem = img_path.stem
        # salva também a "original padronizada" (opcional)
        # (se você não quiser, comente esse bloco)
        base = A.Compose([
            A.LongestMaxSize(max_size=OUT_SIZE, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=OUT_SIZE, min_width=OUT_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        ])(image=img)["image"]

        out_base = OUTPUT_DIR / f"{stem}__base.png"
        cv2.imwrite(str(out_base), base)
        total_written += 1

        for k in range(AUGS_PER_IMAGE):
            augmented = aug(image=img)["image"]

            out_path = OUTPUT_DIR / f"{stem}__aug{k:02d}.png"
            ok = cv2.imwrite(str(out_path), augmented)
            if not ok:
                print(f"[WARN] Falha ao salvar: {out_path}")
            else:
                total_written += 1

    print(f"Concluído! Arquivos gerados: {total_written}")


if __name__ == "__main__":
    main()
