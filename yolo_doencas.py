import os
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, classification_report
)
from ultralytics import YOLO

# ============================================================
# 1. Organizar dataset no formato YOLOv8 classify
# ============================================================

pastaDeDados = 'cells'
datasetDir = 'dataset_yolo_doencas'

doencaClasses = ['Negativa', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'carcinoma']

# Coletar todos os caminhos e rótulos
file_paths = []
labels = []

for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            class_name = os.path.basename(root)
            if class_name in doencaClasses:
                file_paths.append(os.path.join(root, file))
                labels.append(class_name)

print(f'Total de imagens encontradas: {len(file_paths)}')
print(f'Distribuição: {pd.Series(labels).value_counts().to_dict()}')

# Dividir em treino/validação (80/20) com stratify
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f'Treino: {len(train_paths)} | Validação: {len(val_paths)}')

# Criar estrutura de diretórios e copiar arquivos
for split, paths, lbls in [('train', train_paths, train_labels), ('val', val_paths, val_labels)]:
    for path, label in zip(paths, lbls):
        dest_dir = os.path.join(datasetDir, split, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(path, dest_dir)

print(f'Dataset organizado em {datasetDir}/')

# ============================================================
# 2. Treinar modelo YOLOv8 classify
# ============================================================

model = YOLO('yolov8n-cls.pt')

results = model.train(
    data=datasetDir,
    epochs=10,
    device="mps",
    imgsz=50,
    batch=32,
    project='runs_yolo_doencas',
    name='treino',
    exist_ok=True,
    verbose=True,
    val=True
)

# ============================================================
# 3. Carregar histórico de treinamento
# ============================================================

results_csv = os.path.join('runs_yolo_doencas', 'treino', 'results.csv')
df_results = pd.read_csv(results_csv)
df_results.columns = df_results.columns.str.strip()

train_loss = df_results['train/loss'].values
val_loss = df_results['val/loss'].values
top1_acc = df_results.get('metrics/accuracy_top1', pd.Series([0]*len(df_results))).values

# ============================================================
# 4. Avaliar no conjunto de validação
# ============================================================

best_model_path = os.path.join('runs_yolo_doencas', 'treino', 'weights', 'best.pt')
model_best = YOLO(best_model_path)

val_dir = os.path.join(datasetDir, 'val')
class_names_sorted = sorted(os.listdir(val_dir))
class_names_sorted = [c for c in class_names_sorted if not c.startswith('.')]

y_true = []
y_pred = []

for class_idx, class_name in enumerate(class_names_sorted):
    class_dir = os.path.join(val_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        result = model_best.predict(img_path, imgsz=50, verbose=False)
        pred_class = result[0].probs.top1
        y_true.append(class_idx)
        y_pred.append(pred_class)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

num_classes = len(class_names_sorted)

# Métricas gerais
precision = precision_score(y_true, y_pred, average='weighted')
recall_val = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = np.mean(y_true == y_pred)

cm = confusion_matrix(y_true, y_pred)

# Especificidade por classe
def specificity_per_class(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specs = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
    return specs

specs = specificity_per_class(y_true, y_pred, num_classes)

# Relatório detalhado
report = classification_report(y_true, y_pred, target_names=class_names_sorted)

print("\n" + "=" * 60)
print("RESULTADOS YOLO - CLASSIFICAÇÃO POR DOENÇA")
print("=" * 60)
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão (weighted): {precision:.4f}")
print(f"Revocação (weighted): {recall_val:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")
print(f"\nEspecificidade por classe:")
for i, name in enumerate(class_names_sorted):
    print(f"  {name}: {specs[i]:.4f}")
print(f"\nMatriz de Confusão:\n{cm}")
print(f"\nRelatório Detalhado:\n{report}")

# ============================================================
# 5. Gráficos
# ============================================================

epochs_range = range(1, len(train_loss) + 1)

# --- Gráfico 1: Acurácia e Perda ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, top1_acc, label='Acurácia (Top-1)')
plt.title('YOLOv8 Multiclasse - Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Treino Perda')
plt.plot(epochs_range, val_loss, label='Validação Perda')
plt.title('YOLOv8 Multiclasse - Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.savefig('yolo_multiclasse_acuracia_perda.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 2: Matriz de Confusão ---
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names_sorted).plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão - YOLOv8 Multiclasse')
plt.tight_layout()
plt.savefig('yolo_multiclasse_matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 3: Precisão por Classe ---
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)

plt.figure(figsize=(10, 5))
plt.bar(class_names_sorted, precision_per_class, color='blue', alpha=0.7)
plt.title('Precisão por Classe - YOLOv8 Multiclasse')
plt.xlabel('Classe (Doença)')
plt.ylabel('Precisão')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('yolo_multiclasse_precisao_classe.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 4: Revocação por Classe ---
plt.figure(figsize=(10, 5))
plt.bar(class_names_sorted, recall_per_class, color='green', alpha=0.7)
plt.title('Revocação por Classe - YOLOv8 Multiclasse')
plt.xlabel('Classe (Doença)')
plt.ylabel('Revocação')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('yolo_multiclasse_revocacao_classe.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 5: Convergência com marcadores ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, top1_acc, 'go', label='Acurácia (Top-1)')
plt.title('Treino e Validação Acurácia - YOLOv8 Multiclasse')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, 'go', label='Treino Perda')
plt.plot(epochs_range, val_loss, 'b', label='Validação Perda')
plt.title('Treino e Validação Perda - YOLOv8 Multiclasse')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.savefig('yolo_multiclasse_convergencia.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 6: F1-Score e Especificidade por época (valor final) ---
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
f1_per_class = f1_score(y_true, y_pred, average=None)
plt.bar(class_names_sorted, f1_per_class, color='darkorange', alpha=0.7)
plt.title('F1-Score por Classe - YOLOv8 Multiclasse')
plt.xlabel('Classe (Doença)')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')

plt.subplot(1, 2, 2)
plt.bar(class_names_sorted, specs, color='purple', alpha=0.7)
plt.title('Especificidade por Classe - YOLOv8 Multiclasse')
plt.xlabel('Classe (Doença)')
plt.ylabel('Especificidade')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')

plt.tight_layout()
plt.savefig('yolo_multiclasse_f1_especificidade.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 7: Distribuição das classes ---
plt.figure(figsize=(10, 5))
class_counts = pd.Series(labels).value_counts()
plt.bar(class_counts.index, class_counts.values,
        color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#e67e22'], alpha=0.8)
plt.title('Distribuição das Classes no Dataset')
plt.xlabel('Classe (Doença)')
plt.ylabel('Número de Imagens')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('yolo_multiclasse_distribuicao_classes.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 8: Especificidade e F1 por classe (lado a lado) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(class_names_sorted, specs, color='purple', alpha=0.7)
axes[0].set_title('Especificidade por Classe')
axes[0].set_xlabel('Classe (Doença)')
axes[0].set_ylabel('Especificidade')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y')

axes[1].bar(class_names_sorted, f1_per_class, color='orange', alpha=0.7)
axes[1].set_title('F1-Score por Classe')
axes[1].set_xlabel('Classe (Doença)')
axes[1].set_ylabel('F1-Score')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y')

plt.tight_layout()
plt.savefig('yolo_multiclasse_especificidade_f1_classe.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGráficos salvos com prefixo 'yolo_multiclasse_'")
