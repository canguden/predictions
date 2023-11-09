import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

opponent_scores = np.array([80, 75, 90, 70, 85, 95, 78, 92, 88, 79])
home_away = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0])
team_scores = np.array([85, 90, 88, 75, 92, 97, 80, 94, 90, 78])

X = np.column_stack((opponent_scores, home_away))
y = team_scores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

new_data = np.array([[85, 1]])
predicted_score = model.predict(new_data)
print(f"Predicted score: {predicted_score[0]}")
print(f"Predicted score: {predicted_score[0]}")
