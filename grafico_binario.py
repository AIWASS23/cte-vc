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

pastaDeDados = 'cells'

normalClasses = ['Negativa']
anormalClasses = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'carcinoma']

filenames = []
classes = []

for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            class_name = os.path.basename(root)
            
            if class_name in normalClasses:
                filenames.append(file)
                classes.append('Normal')
            elif class_name in anormalClasses:
                filenames.append(file)
                classes.append('Anormal')

if not filenames:
    print("Nenhum arquivo de imagem foi encontrado.")
else:
    data = {
        'filename': filenames,
        'class': classes
    }

    df = pd.DataFrame(data)

    output_csv = 'cell_images_classification.csv'
    df.to_csv(output_csv, index = False)

    print(f'Tabela salva como {output_csv}')

csv_file = 'cell_images_classification.csv'
df_csv = pd.read_csv(csv_file)

image_files = []

for root, dirs, files in os.walk(pastaDeDados):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_files.append(file)


for filename in image_files:
    if filename not in df_csv['filename'].values:
        print(f"A imagem {filename} não tem uma classe associada no CSV.")

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
    print("Arquivos duplicados encontrados:")
    for duplicate in duplicate_files:
        print(f"Duplicado: {duplicate[0]} é duplicata de {duplicate[1]}")
else:
    print("Nenhum arquivo duplicado encontrado.")

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

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.2, random_state = 42)

train_images = train_images / 255.0
test_images = test_images / 255.0

num_classes = len(label_mapping)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

model1 = Sequential()
model1.add(Input(shape=(shape, shape, 3)))
model1.add(Conv2D(16, kernel_size=(3, 3), activation = activation))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation =activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(32, activation = activation))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes, activation='softmax'))

model1.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'])

history1 = model1.fit(train_images, train_labels, epochs = 10, batch_size=32, validation_data = (test_images, test_labels))

results_model1_test = model1.evaluate(test_images, test_labels)
results_model1_train = model1.evaluate(train_images, train_labels)

predictions = model1.predict(test_images)

predictions_classes = np.argmax(predictions, axis=1)
test_labels_classes = np.argmax(test_labels, axis=1)

precision = precision_score(test_labels_classes, predictions_classes, average = 'weighted')
recall = recall_score(test_labels_classes, predictions_classes, average = 'weighted')

cm = confusion_matrix(test_labels_classes, predictions_classes)

print("Resultados do modelo:")
print(f"Acurácia teste com {activation}: {results_model1_test[1]}")
print(f"Acurácia treino com {activation}: {results_model1_train[1]}")
print(f"Perda teste com {activation}: {results_model1_test[0]}")
print(f"Perda treino com {activation}: {results_model1_train[0]}")
print(f"Precisão: {precision}")
print(f"Revocação: {recall}")
print(f'Matriz de confusão: {cm}')

# Plotar acurácia
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Treino Acurácia')
plt.plot(history1.history['val_accuracy'], label='Validação Acurácia')
plt.title('Modelo de Acurácia')
plt.xlabel('Epocas')
plt.ylabel('Acurácia')
plt.legend()

# Plotar perda
plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'], label='Treino Perda')
plt.plot(history1.history['val_loss'], label='Validação Perda')
plt.title('Modelo de Perda')
plt.xlabel('Epocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.savefig('binaria_acuracia_perda.png', dpi=150, bbox_inches='tight')
plt.show()

# Calcular a matriz de confusão
cm = confusion_matrix(test_labels_classes, predictions_classes)
print(cm)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=label_mapping.keys()).plot(cmap=plt.cm.Blues)
plt.title('Matrix de Confusão')
plt.savefig('binaria_matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.show()

# Calcular precisão e recall por classe
precision_per_class = precision_score(test_labels_classes, predictions_classes, average=None)
recall_per_class = recall_score(test_labels_classes, predictions_classes, average=None)

# Preparar nomes das classes
class_names = list(label_mapping.keys())

# Plotar precisão por classe
plt.figure(figsize=(10, 5))
plt.bar(class_names, precision_per_class, color='blue', alpha=0.7)
plt.title('Precisão por Classe')
plt.xlabel('Classe')
plt.ylabel('Precisão')
plt.xticks(rotation = 45)
plt.ylim(0, 1)
plt.grid(axis = 'y')
plt.tight_layout()
plt.savefig('binaria_precisao_classe.png', dpi=150, bbox_inches='tight')
plt.show()

# Plotar recall por classe
plt.figure(figsize=(10, 5))
plt.bar(class_names, recall_per_class, color='green', alpha=0.7)
plt.title('Revocação por Classe')
plt.xlabel('Classe')
plt.ylabel('Revocação')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('binaria_revocacao_classe.png', dpi=150, bbox_inches='tight')
plt.show()

# Plotar convergência da acurácia e da perda
epochs = range(1, len(history1.history['accuracy']) + 1)

plt.figure(figsize=(12, 6))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(epochs, history1.history['accuracy'], 'go', label='Treino Acurácia')
plt.plot(epochs, history1.history['val_accuracy'], 'b', label='Validação Acurácia')
plt.title('Treino e Validação Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(epochs, history1.history['loss'], 'go', label='Treino Perda')
plt.plot(epochs, history1.history['val_loss'], 'b', label='Validação Perda')
plt.title('Treino e Validação Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.savefig('binaria_convergencia.png', dpi=150, bbox_inches='tight')
plt.show()

# Função para calcular a especificidade
def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# Inicializar listas para armazenar F1-Score e Especificidade por época
f1_scores = []
specificities = []

# Avaliar o modelo em cada época
for epoch in range(1, 11):
    # Treinar o modelo até a época corrente
    history1 = model1.fit(train_images, train_labels, epochs=epoch, initial_epoch=epoch-1, batch_size=32, validation_data=(test_images, test_labels), verbose=0)
    
    # Previsões nos dados de teste
    predictions = model1.predict(test_images)
    predictions_classes = np.argmax(predictions, axis=1)
    test_labels_classes = np.argmax(test_labels, axis=1)
    
    # Calcular F1-Score e Especificidade
    f1 = f1_score(test_labels_classes, predictions_classes, average='weighted')
    spec = specificity(test_labels_classes, predictions_classes)
    
    f1_scores.append(f1)
    specificities.append(spec)

print(f'F1-Score: {f1_scores}')
print(f'Especificidade: {specificities}')

# Plotar F1-Score por época
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), f1_scores, marker='o')
plt.title('F1-Score')
plt.xlabel('Época')
plt.ylabel('F1-Score')
plt.grid(True)

# Plotar Especificidade por época
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), specificities, marker='o')
plt.title('Especificidade')
plt.xlabel('Época')
plt.ylabel('Especificidade')
plt.grid(True)

plt.tight_layout()
plt.savefig('binaria_f1_especificidade.png', dpi=150, bbox_inches='tight')
plt.show()