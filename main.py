import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Configurações iniciais
IMG_SIZE = (100, 100)
BATCH_SIZE = 64
EPOCHS = 10
CLASSES = ['apple_red_1', 'carrot_1', 'pear_1']

# Caminhos para os diretórios do dataset
TRAIN_DIR = "fruits-360-original-size/Training"
VALIDATION_DIR = "fruits-360-original-size/Validation"
TEST_DIR = "fruits-360-original-size/Test"

# Pré-processamento das imagens
data_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
data_gen_validation = ImageDataGenerator(rescale=1./255)
data_gen_test = ImageDataGenerator(rescale=1./255)

# Carregar os dados de treino, validação e teste
train_data = data_gen_train.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES
)

validation_data = data_gen_validation.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES
)

test_data = data_gen_test.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES
)

# Carregar o modelo VGG16 pré-treinado sem as camadas superiores (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Congelar as camadas do modelo base para preservar os pesos pré-treinados
for layer in base_model.layers:
    layer.trainable = False

# Adicionar novas camadas de classificação ao modelo
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data
)


# Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(test_data)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

# Exibir predições para um subconjunto de imagens de teste
sample_test_images, sample_test_labels = next(test_data)
predictions = model.predict(sample_test_images)

# Salvar o modelo no formato HDF5
model.save("models/fruit_classifier_model.h5")
print("Modelo salvo como HDF5 em 'fruit_classifier_model.h5'")


# Plotar as imagens com rótulos preditos e reais
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_test_images[i])
    predicted_label = CLASSES[np.argmax(predictions[i])]
    true_label = CLASSES[np.argmax(sample_test_labels[i])]
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()
