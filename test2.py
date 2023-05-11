import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import numpy as np


# Génère des données aléatoires pour x et y
x = [1,2,3,4,5,6,7,8,9,10]
y = [2,4,5,4,6,7,8,9,11,12]

# Convertit les données en tableaux numpy pour utilisation avec scikit-learn

x = np.array([7, 9, 9, 10, 13, 17, 19, 20, 21, 25]).reshape((-1, 1))
y = np.array([5, 4, 6, 4, 1, 2, 0, 1, 1, 0])


# Crée un modèle de régression linéaire
model = LinearRegression()

# Entraîne le modèle sur les données
model.fit(x, y)

# Effectue des prédictions sur les données d'entraînement
y_pred = model.predict(x)

# Trace les données d'entraînement et la droite de régression
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

# affichage de l'equation de la droite
equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
print(equation)