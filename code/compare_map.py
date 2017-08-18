import numpy as np

eps = 0.1
A = np.array([[1, 1],
              [1, 1 + eps]])

eta = 0.01
gamma = np.array([[eta ** 2, 0],
                  [0, eta ** 2]])
gamma_inv = np.linalg.inv(gamma)

Ai = np.linalg.inv(A)
AiMAP = np.linalg.inv(A.T.dot(gamma_inv).dot(A) + np.eye(2)).dot(A.T).dot(gamma_inv)

print(Ai)
print(AiMAP)

print(np.linalg.norm(Ai - AiMAP, ord=2) / np.linalg.norm(Ai, ord=2))