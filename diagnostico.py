import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Carregando o dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Realizando padding dos dados
max_length = max([len(item) for item in data_dict['data']])
padded_data = [np.pad(item, (0, max_length - len(item)), mode='constant') for item in data_dict['data']]

# Convertendo para arrays numpy
data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])

# Verificando o balanceamento das classes
class_counts = Counter(labels)
print("Distribuição de classes:", class_counts)

# Reduzir o dataset para visualização (usando PCA para reduzir para 2D)
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# Visualizando as classes em 2D
plt.figure(figsize=(10, 6))
for label in np.unique(labels):
    indices = np.where(labels == label)
    plt.scatter(data_reduced[indices, 0], data_reduced[indices, 1], label=f'Classe {label}', alpha=0.6)
plt.title("Visualização das Classes após PCA")
plt.legend()
plt.show()

# Treinando com apenas 3 classes para diagnóstico
# Aqui selecionamos as classes 'A', 'C', e 'M' apenas para simplificar o problema
selected_classes = [0, 2, 12]
indices = np.isin(labels, selected_classes)
data_small = data[indices]
labels_small = labels[indices]

# Verificando o balanceamento das classes reduzidas
class_counts_small = Counter(labels_small)
print("Distribuição de classes reduzida (A, C, M):", class_counts_small)

# Dividindo os dados
x_train, x_test, y_train, y_test = train_test_split(data_small, labels_small, test_size=0.3, random_state=42)

# Treinando um modelo simples de Random Forest com as classes reduzidas
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Fazendo predições no conjunto de teste
y_pred = rf_model.predict(x_test)

# Verificando o desempenho
accuracy = np.mean(y_pred == y_test)
print(f'Acurácia no dataset reduzido: {accuracy * 100:.2f}%')

# Exibindo matriz de confusão para as classes 'A', 'C' e 'M'
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão (A, C, M):\n", conf_matrix)
