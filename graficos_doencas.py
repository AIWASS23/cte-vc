import hashlib
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras import Input

# ============================================================
# 1. Preparação dos dados - Classificação por doença específica
# ============================================================

pastaDeDados = 'cells'
doencaClasses = ['Negativa', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'carcinoma']

filenames = []
classes = []

for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            class_name = os.path.basename(root)
            if class_name in doencaClasses:
                filenames.append(file)
                classes.append(class_name)

if not filenames:
    print("Nenhum arquivo de imagem foi encontrado.")
else:
    data = {'filename': filenames, 'class': classes}
    df = pd.DataFrame(data)
    output_csv = 'cell_images_doencas.csv'
    df.to_csv(output_csv, index=False)
    print(f'Tabela salva como {output_csv}')

csv_file = 'cell_images_doencas.csv'
df_csv = pd.read_csv(csv_file)

# ============================================================
# 2. Detecção de duplicatas
# ============================================================

def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

file_hashes = {}
duplicate_files = []

for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            file_path = os.path.join(root, file)
            file_hash = calculate_file_hash(file_path)
            if file_hash in file_hashes:
                duplicate_files.append((file_path, file_hashes[file_hash]))
            else:
                file_hashes[file_hash] = file_path

if duplicate_files:
    print(f"Arquivos duplicados encontrados: {len(duplicate_files)}")
else:
    print("Nenhum arquivo duplicado encontrado.")

# ============================================================
# 3. Carregamento e treinamento
# ============================================================

shape = 50
activation = 'relu6'

def load_data(main_folder, labels_file, target_size=(shape, shape)):
    labels_df = pd.read_csv(labels_file)
    images = []
    labels = []
    label_mapping = {label: idx for idx, label in enumerate(labels_df['class'].unique())}

    for root, dirs, files in os.walk(main_folder):
        for filename in files:
            if filename.endswith(".tif"):
                image_path = os.path.join(root, filename)
                img = image.load_img(image_path, target_size=target_size)
                img_array = image.img_to_array(img)
                images.append(img_array)
                label = labels_df[labels_df['filename'] == filename]['class'].values[0]
                label_idx = label_mapping[label]
                labels.append(label_idx)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_mapping

images, labels, label_mapping = load_data(pastaDeDados, csv_file)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_images = train_images / 255.0
test_images = test_images / 255.0

num_classes = len(label_mapping)

train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels_cat = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# Modelo multiclasse
model = Sequential()
model.add(Input(shape=(shape, shape, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation=activation))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(64, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(train_images, train_labels_cat, epochs=10, batch_size=32,
                    validation_data=(test_images, test_labels_cat))

# ============================================================
# 4. Avaliação
# ============================================================

results_test = model.evaluate(test_images, test_labels_cat)
results_train = model.evaluate(train_images, train_labels_cat)

predictions = model.predict(test_images)
predictions_classes = np.argmax(predictions, axis=1)
test_labels_classes = np.argmax(test_labels_cat, axis=1)

precision = precision_score(test_labels_classes, predictions_classes, average='weighted')
recall = recall_score(test_labels_classes, predictions_classes, average='weighted')

cm = confusion_matrix(test_labels_classes, predictions_classes)

inverse_label_mapping = {v: k for k, v in label_mapping.items()}
class_names = [inverse_label_mapping[i] for i in range(num_classes)]

print("Resultados do modelo multiclasse:")
print(f"Acurácia teste com {activation}: {results_test[1]}")
print(f"Acurácia treino com {activation}: {results_train[1]}")
print(f"Perda teste com {activation}: {results_test[0]}")
print(f"Perda treino com {activation}: {results_train[0]}")
print(f"Precisão: {precision}")
print(f"Revocação: {recall}")

# ============================================================
# 5. Gráficos
# ============================================================

# --- Gráfico 1: Acurácia e Perda ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino Acurácia')
plt.plot(history.history['val_accuracy'], label='Validação Acurácia')
plt.title('Modelo Multiclasse - Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino Perda')
plt.plot(history.history['val_loss'], label='Validação Perda')
plt.title('Modelo Multiclasse - Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.savefig('multiclasse_acuracia_perda.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 2: Matriz de Confusão ---
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão - Classificação por Doença')
plt.tight_layout()
plt.savefig('multiclasse_matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 3: Precisão por Classe ---
precision_per_class = precision_score(test_labels_classes, predictions_classes, average=None)
recall_per_class = recall_score(test_labels_classes, predictions_classes, average=None)

plt.figure(figsize=(10, 5))
plt.bar(class_names, precision_per_class, color='blue', alpha=0.7)
plt.title('Precisão por Classe - Multiclasse')
plt.xlabel('Classe (Doença)')
plt.ylabel('Precisão')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('multiclasse_precisao_classe.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 4: Revocação por Classe ---
plt.figure(figsize=(10, 5))
plt.bar(class_names, recall_per_class, color='green', alpha=0.7)
plt.title('Revocação por Classe - Multiclasse')
plt.xlabel('Classe (Doença)')
plt.ylabel('Revocação')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('multiclasse_revocacao_classe.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 5: Convergência com marcadores ---
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], 'go', label='Treino Acurácia')
plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validação Acurácia')
plt.title('Treino e Validação Acurácia - Multiclasse')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], 'go', label='Treino Perda')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validação Perda')
plt.title('Treino e Validação Perda - Multiclasse')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.savefig('multiclasse_convergencia.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 6: F1-Score e Especificidade por época ---

def specificity_multiclass(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specs = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
    return np.mean(specs)

f1_scores = []
specificities = []

for epoch in range(1, 11):
    history_epoch = model.fit(train_images, train_labels_cat, epochs=epoch, initial_epoch=epoch - 1,
                              batch_size=32, validation_data=(test_images, test_labels_cat), verbose=0)

    preds = model.predict(test_images)
    preds_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(test_labels_cat, axis=1)

    f1 = f1_score(true_classes, preds_classes, average='weighted')
    spec = specificity_multiclass(true_classes, preds_classes, num_classes)

    f1_scores.append(f1)
    specificities.append(spec)

print(f'F1-Score por época: {f1_scores}')
print(f'Especificidade média por época: {specificities}')

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), f1_scores, marker='o')
plt.title('F1-Score (Weighted) - Multiclasse')
plt.xlabel('Época')
plt.ylabel('F1-Score')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), specificities, marker='o')
plt.title('Especificidade Média - Multiclasse')
plt.xlabel('Época')
plt.ylabel('Especificidade')
plt.grid(True)

plt.tight_layout()
plt.savefig('multiclasse_f1_especificidade.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 7: Distribuição das classes (para referência) ---
plt.figure(figsize=(10, 5))
class_counts = df_csv['class'].value_counts()
plt.bar(class_counts.index, class_counts.values, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#e67e22'], alpha=0.8)
plt.title('Distribuição das Classes no Dataset')
plt.xlabel('Classe (Doença)')
plt.ylabel('Número de Imagens')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('multiclasse_distribuicao_classes.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 8: Especificidade por classe (barra) ---
def specificity_per_class(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specs = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
    return specs

specs_per_class = specificity_per_class(test_labels_classes, predictions_classes, num_classes)
f1_per_class = f1_score(test_labels_classes, predictions_classes, average=None)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(class_names, specs_per_class, color='purple', alpha=0.7)
axes[0].set_title('Especificidade por Classe')
axes[0].set_xlabel('Classe (Doença)')
axes[0].set_ylabel('Especificidade')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y')

axes[1].bar(class_names, f1_per_class, color='orange', alpha=0.7)
axes[1].set_title('F1-Score por Classe')
axes[1].set_xlabel('Classe (Doença)')
axes[1].set_ylabel('F1-Score')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y')

plt.tight_layout()
plt.savefig('multiclasse_especificidade_f1_classe.png', dpi=150, bbox_inches='tight')
plt.show()
