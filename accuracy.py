import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = np.array([
    [2, 1, 60, 20, 10, 1],
    [1, 2, 58, 18, 12, 0],
    [3, 2, 62, 22, 11, 1],

])

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

new_game_features = np.array([2, 1, 65, 21, 9])
predicted_result = model.predict([new_game_features])
print(f"Predicted result (1 for a win, 0 for a loss): {int(predicted_result[0])}")
