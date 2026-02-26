import json
from pathlib import Path

# ====== CONFIG ======
SRC_JSON_ROOT = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\datasets\dataset-yolov7-2026-10-01\annotations\train")     # pasta onde estão os .json (pode ter subpastas)
DST_LABELS = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\datasets\dataset-yolov7-2026-10-01\labels\train")  # pasta destino dos .txt (YOLO)
RECURSIVE = True

# Se quiser fixar IDs manualmente, preencha aqui.
# Ex: {"roi": 0, "selo_dobrado": 0, "selo_normal": 1}
CLASS_MAP: dict[str, int] = {}  # vazio => auto-cria com base nos labels encontrados


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rect_points_to_yolo(points, img_w: int, img_h: int):
    """
    points: [[x1,y1],[x2,y2]] (LabelMe rectangle)
    retorna (xc, yc, w, h) normalizados [0..1]
    """
    (x1, y1), (x2, y2) = points
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

    # opcional: clamp dentro da imagem (evita valores fora)
    x_min = clamp(x_min, 0, img_w)
    x_max = clamp(x_max, 0, img_w)
    y_min = clamp(y_min, 0, img_h)
    y_max = clamp(y_max, 0, img_h)

    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w <= 0 or box_h <= 0:
        return None

    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0

    # normaliza
    return (xc / img_w, yc / img_h, box_w / img_w, box_h / img_h)


def main():
    safe_mkdir(DST_LABELS)

    json_files = SRC_JSON_ROOT.rglob("*.json") if RECURSIVE else SRC_JSON_ROOT.glob("*.json")
    json_files = list(json_files)

    # 1) Se CLASS_MAP vazio, vamos descobrir todos os labels e criar mapeamento estável
    discovered_labels: set[str] = set()

    if not CLASS_MAP:
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for sh in data.get("shapes", []) or []:
                lbl = sh.get("label")
                if isinstance(lbl, str) and lbl.strip():
                    discovered_labels.add(lbl.strip())

        # ordena para manter IDs consistentes
        labels_sorted = sorted(discovered_labels)
        for i, lbl in enumerate(labels_sorted):
            CLASS_MAP[lbl] = i

        # salva classes.txt
        (DST_LABELS / "classes.txt").write_text("\n".join(labels_sorted), encoding="utf-8")
    else:
        # salva classes.txt conforme IDs
        inv = sorted(CLASS_MAP.items(), key=lambda kv: kv[1])
        (DST_LABELS / "classes.txt").write_text("\n".join([k for k, _ in inv]), encoding="utf-8")

    converted = 0
    empty = 0
    skipped_shapes = 0
    skipped_json = 0
    warnings = []

    # 2) Converter cada JSON
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            skipped_json += 1
            warnings.append(f"[JSON INVÁLIDO] {jf} | {e}")
            continue

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not isinstance(img_w, int) or not isinstance(img_h, int) or img_w <= 0 or img_h <= 0:
            skipped_json += 1
            warnings.append(f"[SEM DIMENSÕES] {jf} (imageWidth/imageHeight inválidos)")
            continue

        shapes = data.get("shapes", []) or []

        lines = []
        for sh in shapes:
            label = (sh.get("label") or "").strip()
            if not label:
                skipped_shapes += 1
                continue

            if label not in CLASS_MAP:
                # label não mapeado => pula (ou você pode decidir lançar erro)
                skipped_shapes += 1
                continue

            points = sh.get("points")
            if not isinstance(points, list) or len(points) != 2:
                # este script está focado em bbox (rectangle); se quiser polygon->bbox eu adapto
                skipped_shapes += 1
                continue

            # opcional: exigir shape_type == rectangle (se quiser ficar mais estrito)
            # if (sh.get("shape_type") or "").lower() != "rectangle":
            #     skipped_shapes += 1
            #     continue

            yolo = rect_points_to_yolo(points, img_w, img_h)
            if yolo is None:
                skipped_shapes += 1
                continue

            cls_id = CLASS_MAP[label]
            xc, yc, bw, bh = yolo
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # nome do .txt: usa o stem do JSON (recomendado)
        out_txt = DST_LABELS / f"{jf.stem}.txt"
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if lines:
            converted += 1
        else:
            empty += 1

    print("Concluído.")
    print(f"JSONs encontrados: {len(json_files)}")
    print(f"TXT com anotações: {converted}")
    print(f"TXT vazios (sem bbox): {empty}")
    print(f"Shapes pulados: {skipped_shapes}")
    print(f"JSONs pulados: {skipped_json}")
    print(f"classes.txt salvo em: {DST_LABELS / 'classes.txt'}")

    if warnings:
        print("\n=== Avisos ===")
        for w in warnings[:50]:
            print(w)
        if len(warnings) > 50:
            print(f"... ({len(warnings)-50} avisos a mais)")


if __name__ == "__main__":
    main()
