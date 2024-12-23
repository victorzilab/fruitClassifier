# Fruit Classifier

Este repositório contém um classificador de frutas utilizando uma Rede Neural Convolucional (CNN) desenvolvida em TensorFlow/Keras, com base no dataset **Fruits-360** fornecido pela Kaggle. O projeto foi desenvolvido como parte de um teste de conhecimentos para a fase final do processo seletivo de estágio. O dataset utilizado foi uma versão reduzida, contendo uma seleção representativa de itens de cada classe. A arquitetura **VGG16** foi aplicada para **transfer learning** no modelo.

O dataset completo pode ser obtido em: https://www.kaggle.com/datasets/moltean/fruits

## Estrutura do Repositório
```
fruit-classifier/
├── README.md               # Documentação do projeto
├── requirements.txt        # Dependências do projeto
├── main.py                 # Código principal para treinar e testar o modelo
├── testmodel.py            # Código para testar o modelo treinado
├── fruits-360-original-size/
│   ├── Test/               # Dados de teste
│   ├── Training/           # Dados de treinamento
│   └── Validation/         # Dados de validação
├── models/                 # Onde o modelo treinado é salvo
│   └── fruit_classifier_model.h5  # Modelo treinado
├── images/                 # Insira aqui as imagens para teste do modelo gerado

```

### Explicação das pastas:
- **fruits-360-original-size/**: Contém os conjuntos de dados de treinamento, teste e validação para o classificador de frutas. Este dataset foi obtido do Kaggle.
- **models/**: Contém o modelo treinado e salvo no formato `.h5`.
- **images/**: Pasta onde você pode colocar suas próprias imagens para testar o modelo treinado.

## Instalação

### Dependências
Certifique-se de ter o **Python 3.12** (ou superior) instalado. Para gerenciar as dependências, o projeto usa um arquivo `requirements.txt` que lista todos os pacotes necessários.

### Preparando o Ambiente
1. **Clone o repositório**:
   ```bash
   git clone https://github.com/victorzilab/fruit-classifier.git
   cd fruit-classifier
   ```

2. **Atualize o pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para testar o funcionamento do código, escolhemos 3 elementos do dataset: **maçã, cenoura e pêra**, para serem detectadas pelo classificador.

### Treinamento e Teste do Modelo

Execute o script principal para treinar e testar o modelo:

```bash
python main.py
```

### Teste do Modelo

Para testar o modelo treinado com imagens diferentes do dataset, coloque as imagens desejadas na pasta **images** e execute o script de teste:

```bash
python testmodel.py
```

Este script irá carregar o modelo treinado, fazer previsões sobre imagens específicas e exibir os resultados de classificação, como o índice de confiabilidade de cada previsão.
