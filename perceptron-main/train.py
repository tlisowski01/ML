import numpy as np
from sklearn.linear_model import Perceptron
import pickle

X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 1, 1, 0])

model = Perceptron()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
