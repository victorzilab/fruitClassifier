import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Carregar o modelo salvo
model = load_model("fruit_classifier_model.h5")

# Exibir o resumo da arquitetura do modelo
print("Resumo do Modelo:")
model.summary()

# Definir as classes
CLASSES = ['apple_red_1', 'carrot_1', 'pear_1']


# Função para pré-processar a imagem
def preprocess_image(image_path, target_size=(100, 100)):
    img = load_img(image_path, target_size=target_size)  # Carregar a imagem com o tamanho alvo
    img_array = img_to_array(img)  # Converter a imagem em um array
    img_array = img_array / 255.0  # Normalizar os valores dos pixels
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar a dimensão do lote
    return img_array


# Função para testar todas as imagens da pasta
def test_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"\nTestando imagem: {filename}")

            # Pré-processar a imagem
            processed_image = preprocess_image(image_path)

            # Fazer uma previsão
            predictions = model.predict(processed_image)
            predicted_class = CLASSES[np.argmax(predictions)]
            confidence = np.max(predictions)  # Índice de confiabilidade

            # Exibir o resultado
            print(f"Classe Prevista: {predicted_class}")
            print(f"Índice de Confiabilidade: {confidence * 100:.2f}%")
            print(f"Probabilidades de Previsão: {predictions}")

            # Visualizar a imagem e a previsão
            img = load_img(image_path)
            plt.imshow(img)
            plt.title(f"Previsto: {predicted_class}\nConfiança: {confidence * 100:.2f}%")
            plt.axis('off')
            plt.show()


# Definir o caminho para a pasta de imagens
images_folder = "images"  # Substitua pelo caminho correto da sua pasta

# Testar todas as imagens na pasta
test_images_in_folder(images_folder)
