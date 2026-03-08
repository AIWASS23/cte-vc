import os
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)
from ultralytics import YOLO

# ============================================================
# 1. Organizar dataset (reutiliza se já existir)
# ============================================================

pastaDeDados = 'cells'
datasetDir = 'dataset_yolo_doencas'

doencaClasses = ['Negativa', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'carcinoma']

if not os.path.exists(os.path.join(datasetDir, 'train')):
    file_paths = []
    labels = []
    for root, dirs, files in os.walk(pastaDeDados):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                class_name = os.path.basename(root)
                if class_name in doencaClasses:
                    file_paths.append(os.path.join(root, file))
                    labels.append(class_name)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    for split, paths, lbls in [('train', train_paths, train_labels), ('val', val_paths, val_labels)]:
        for path, label in zip(paths, lbls):
            dest_dir = os.path.join(datasetDir, split, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(path, dest_dir)

    print(f'Dataset organizado em {datasetDir}/')
else:
    print(f'Dataset já existe em {datasetDir}/, reutilizando.')

# ============================================================
# 2. Definir experimentos de variação de hiperparâmetros
# ============================================================

experimentos = [
    # Experimento base
    {'nome': 'base',       'model_pt': 'yolov8n-cls.pt', 'epochs': 10, 'batch': 32, 'imgsz': 50, 'lr0': 0.01},
    # Variação de épocas
    {'nome': 'epochs_5',   'model_pt': 'yolov8n-cls.pt', 'epochs': 10,  'batch': 32, 'imgsz': 50, 'lr0': 0.01},
    {'nome': 'epochs_20',  'model_pt': 'yolov8n-cls.pt', 'epochs': 15, 'batch': 32, 'imgsz': 50, 'lr0': 0.01},
    # Variação de batch size
    {'nome': 'batch_16',   'model_pt': 'yolov8n-cls.pt', 'epochs': 10, 'batch': 16, 'imgsz': 50, 'lr0': 0.01},
    {'nome': 'batch_64',   'model_pt': 'yolov8n-cls.pt', 'epochs': 10, 'batch': 64, 'imgsz': 50, 'lr0': 0.01},
    # Variação de learning rate
    {'nome': 'lr_0001',    'model_pt': 'yolov8n-cls.pt', 'epochs': 10, 'batch': 32, 'imgsz': 50, 'lr0': 0.001},
    {'nome': 'lr_005',     'model_pt': 'yolov8n-cls.pt', 'epochs': 10, 'batch': 32, 'imgsz': 50, 'lr0': 0.05},
    # Variação de modelo (nano vs small)
    {'nome': 'modelo_s',   'model_pt': 'yolov8s-cls.pt', 'epochs': 10, 'batch': 32, 'imgsz': 50, 'lr0': 0.01},
]

# ============================================================
# 3. Função de avaliação multiclasse
# ============================================================

def avaliar_modelo(best_path, val_dir, imgsz):
    model_eval = YOLO(best_path)
    class_names = sorted([c for c in os.listdir(val_dir) if not c.startswith('.')])
    num_classes = len(class_names)
    y_true, y_pred = [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(val_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            result = model_eval.predict(img_path, imgsz=imgsz, verbose=False)
            y_true.append(class_idx)
            y_pred.append(result[0].probs.top1)
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Especificidade média
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specs = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec_media = np.mean(specs)

    return {'acuracia': acc, 'precisao': prec, 'revocacao': rec, 'f1': f1, 'especificidade_media': spec_media}

# ============================================================
# 4. Executar experimentos
# ============================================================

val_dir = os.path.join(datasetDir, 'val')
resultados = []

for exp in experimentos:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {exp['nome']}")
    print(f"  Modelo: {exp['model_pt']} | Épocas: {exp['epochs']} | Batch: {exp['batch']} | LR: {exp['lr0']}")
    print(f"{'='*60}")

    model = YOLO(exp['model_pt'])
    model.train(
        data=datasetDir,
        epochs=exp['epochs'],
        device='mps',
        imgsz=exp['imgsz'],
        batch=exp['batch'],
        lr0=exp['lr0'],
        project='runs_hiper_doencas',
        name=exp['nome'],
        exist_ok=True,
        verbose=True,
        val=True
    )

    best_path = os.path.join('runs_hiper_doencas', exp['nome'], 'weights', 'best.pt')
    metricas = avaliar_modelo(best_path, val_dir, exp['imgsz'])
    metricas['experimento'] = exp['nome']
    metricas['modelo'] = exp['model_pt']
    metricas['epochs'] = exp['epochs']
    metricas['batch'] = exp['batch']
    metricas['lr0'] = exp['lr0']
    resultados.append(metricas)

    print(f"  Acurácia: {metricas['acuracia']:.4f} | F1: {metricas['f1']:.4f} | Espec: {metricas['especificidade_media']:.4f}")

# ============================================================
# 5. Tabela comparativa
# ============================================================

df = pd.DataFrame(resultados)
print("\n" + "=" * 80)
print("TABELA COMPARATIVA DE HIPERPARÂMETROS - MULTICLASSE")
print("=" * 80)
print(df[['experimento', 'modelo', 'epochs', 'batch', 'lr0', 'acuracia', 'precisao', 'revocacao', 'f1', 'especificidade_media']].to_string(index=False))

df.to_csv('resultados_hiperparametros_doencas.csv', index=False)
print("\nResultados salvos em resultados_hiperparametros_doencas.csv")

# ============================================================
# 6. Gráficos comparativos
# ============================================================

nomes = df['experimento'].values
acuracias = df['acuracia'].values
f1_scores = df['f1'].values
specs = df['especificidade_media'].values

x = np.arange(len(nomes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width, acuracias, width, label='Acurácia', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, f1_scores, width, label='F1-Score', color='#e67e22', alpha=0.8)
bars3 = ax.bar(x + width, specs, width, label='Especif. Média', color='#9b59b6', alpha=0.8)

ax.set_xlabel('Experimento')
ax.set_ylabel('Valor da Métrica')
ax.set_title('Comparação de Hiperparâmetros - Classificação Multiclasse')
ax.set_xticks(x)
ax.set_xticklabels(nomes, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('yolo_multiclasse_comparacao_hiperparametros.png', dpi=150, bbox_inches='tight')
plt.show()

# Gráfico de linhas por variação
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Variação de épocas
ep_exp = df[df['experimento'].isin(['epochs_5', 'base', 'epochs_20'])]
axes[0].plot([5, 10, 20], ep_exp['acuracia'].values, 'o-', label='Acurácia', color='#3498db')
axes[0].plot([5, 10, 20], ep_exp['f1'].values, 's-', label='F1-Score', color='#e67e22')
axes[0].set_xlabel('Número de Épocas')
axes[0].set_ylabel('Valor')
axes[0].set_title('Impacto do Número de Épocas')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.6, 1.0)

# Variação de batch
bt_exp = df[df['experimento'].isin(['batch_16', 'base', 'batch_64'])]
axes[1].plot([16, 32, 64], bt_exp['acuracia'].values, 'o-', label='Acurácia', color='#3498db')
axes[1].plot([16, 32, 64], bt_exp['f1'].values, 's-', label='F1-Score', color='#e67e22')
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('Valor')
axes[1].set_title('Impacto do Batch Size')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.6, 1.0)

# Variação de learning rate
lr_exp = df[df['experimento'].isin(['lr_0001', 'base', 'lr_005'])]
axes[2].plot([0.001, 0.01, 0.05], lr_exp['acuracia'].values, 'o-', label='Acurácia', color='#3498db')
axes[2].plot([0.001, 0.01, 0.05], lr_exp['f1'].values, 's-', label='F1-Score', color='#e67e22')
axes[2].set_xlabel('Taxa de Aprendizado')
axes[2].set_ylabel('Valor')
axes[2].set_title('Impacto da Taxa de Aprendizado')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xscale('log')
axes[2].set_ylim(0.6, 1.0)

plt.tight_layout()
plt.savefig('yolo_multiclasse_variacao_hiperparametros.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGráficos salvos!")
