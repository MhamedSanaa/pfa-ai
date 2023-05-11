import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(0)
x = np.array([7, 9, 9, 10, 13, 17, 19, 20, 21, 25])
y = np.array([5, 4, 6, 4, 1, 2, 0, 1, 1, 0])

# Initialisation des paramètres de la régression linéaire
a = 0
a0 = a
b = 0
b0 = b

# Initialisation des paramètres
learning_rate = 0.001
n_iterations = 25000

# Boucle pour n_iterations

for i in range(25000):
    y_pred = a * x + b

    # gradients (dérivées partielles) #error= 1/(2*n)* np.sum((y_pred - y) ** 2)

    grada = np.mean(
        (y_pred - y) * x
    )  # pour être précis, ceci est d error sur d a :  d(error)/da
    gradb = np.mean(
        y_pred - y
    )  # pour être précis, ceci est d error sur d a: a d(error)/db

    # Mise à jour des paramètres  (pousser les poids contrairement à la force de leur montée)
    a = a - learning_rate * grada
    b = b - learning_rate * gradb
    print("iterationnumber:",i,"(a,b)", a, b)
    y_pred = a * x + b
    erreur = 1/2*np.mean((y_pred - y)**2)
    print('erreur: ',erreur)
    # col = (np.random.random(), np.random.random(), np.random.random())
    # plt.plot(x, a * x + b, color=col)

# Affichage de la droite de régression
plt.scatter(x, y)
plt.plot(x, a0 * x + b0, color="red")

col = (np.random.random(), np.random.random(), np.random.random())
print("(a,b)", a, b)
plt.plot(x, a * x + b, color=col)
plt.show()
