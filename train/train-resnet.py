from pathlib import Path
import shutil
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import shutil


# CONFIGURANDO AMBIENTE (script rodou no Google Colab, dentro do Google Drive)

DRIVE_DATASET_DIR = Path("/content/drive/MyDrive/DoutoradoMauricio/datasets/dataset-resnet-2026-12-01(MAND)")  # <- ajuste
LOCAL_DATASET_DIR = Path("/content/resnet_dataset")

# Limpa e copia (se for grande, pode demorar um pouco, mas evita travar no meio)
if LOCAL_DATASET_DIR.exists():
    shutil.rmtree(LOCAL_DATASET_DIR)

shutil.copytree(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR)

print("OK, dataset local em:", LOCAL_DATASET_DIR)
print("train exists?", (LOCAL_DATASET_DIR/"train").exists())
print("val exists?", (LOCAL_DATASET_DIR/"val").exists())


# PREPARANDO TREINO PARA RESNET-18 UTILIZANDO CLASS WEIGHTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Transforms SEM AUG (baseline)
train_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # garante 3 canais p/ ResNet
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # padrão ImageNet (ok p/ baseline)
                         std=[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dir = str(LOCAL_DATASET_DIR / "train")
val_dir   = str(LOCAL_DATASET_DIR / "val")

train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds   = datasets.ImageFolder(val_dir, transform=val_tfms)

print("Classes:", train_ds.classes)
print("class_to_idx:", train_ds.class_to_idx)
print("Train size:", len(train_ds), "Val size:", len(val_ds))

# Agora calculamos class weights diretamente do dataset e treinamos com CrossEntropy
# class weights serve para compensar o desbalanceamento das classes (muito mais sem fratura do que com fratura)



# Conta quantas imagens por classe no TRAIN
targets = [y for _, y in train_ds.samples]
counts = Counter(targets)
num_classes = len(train_ds.classes)
class_counts = np.array([counts[i] for i in range(num_classes)], dtype=np.float32)

print("Counts (train):", {train_ds.classes[i]: int(class_counts[i]) for i in range(num_classes)})

# Peso inversamente proporcional à frequência
class_weights = (class_counts.sum() / (num_classes * class_counts))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)


# RODANDO O TREINAMENTO

# Modelo
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# learning rate:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device=="cuda"))

def run_epoch(model, loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds, all_true = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_true.append(y.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_true, all_preds

EPOCHS = 30
best_val_f1 = -1.0
#best_path_local = Path("/content/best_resnet18.pth")

# salvando num lugar onde eu nao perca caso caia a internet
BEST_PATH = Path("/content/drive/MyDrive/DoutoradoMauricio/models/best-dataset-resnet-2026-12-01(MAND)_v4.pth")
BEST_PATH.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    t0 = time.time()

    train_loss, _, _ = run_epoch(model, train_loader, train=True)
    val_loss, y_true, y_pred = run_epoch(model, val_loader, train=False)

    # relatório focando na classe "com_fratura"
    report = classification_report(
        y_true, y_pred,
        target_names=train_ds.classes,
        output_dict=True,
        zero_division=0
    )
    f1_com = report["com_fratura"]["f1-score"] if "com_fratura" in report else report[train_ds.classes[0]]["f1-score"]
    rec_com = report["com_fratura"]["recall"] if "com_fratura" in report else report[train_ds.classes[0]]["recall"]
    prec_com = report["com_fratura"]["precision"] if "com_fratura" in report else report[train_ds.classes[0]]["precision"]

    dt = time.time() - t0
    print(f"\nEpoch {epoch}/{EPOCHS} | {dt:.1f}s")
    print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    print(f"  com_fratura: precision={prec_com:.3f} recall={rec_com:.3f} f1={f1_com:.3f}")

    # salva o melhor (por F1 de com_fratura)
    if f1_com > best_val_f1:
        best_val_f1 = f1_com
        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "classes": train_ds.classes,
            "epoch": epoch,
            "best_val_f1_com_fratura": best_val_f1
        }, BEST_PATH)
        print("  >>> salvou melhor modelo em:", BEST_PATH)

