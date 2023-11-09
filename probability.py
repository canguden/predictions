import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = np.array([
    [100, 95, 40, 20, 10, 1],
    [95, 100, 38, 18, 12, 0],
    [110, 105, 42, 22, 11, 1],
    # Add more data entries here
])

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

new_game_features = np.array([105, 100, 45, 21, 9])
predicted_result = model.predict([new_game_features])
print(f"Predicted result (1 for a win, 0 for a loss): {int(round(predicted_result[0])}")
