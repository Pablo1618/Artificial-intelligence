import numpy as np
import matplotlib.pyplot as plt
from data import get_data, inspect_data, split_data
data = get_data()
#inspect_data(data)
train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error
# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2
# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]

# macierz obserwacji z jedynkami w kolumnie po lewej stronie
#observation_matrix = np.concatenate([np.ones((len(x_train), 1)), x_train.reshape(-1, 1)], axis=1)
observation_matrix = np.column_stack((np.ones(len(x_train)),x_train))
# np.ones((len(x_train), 1) -> utworzenie macierzy o 1 kolumnie wypelniona cyframi 1
# x_train.reshape(-1, 1) -> utworzenie macierzy o 1 kolumnie z danych
#print(observation_matrix)
# wzór 1.13 - wyznaczenie optymalnej thety, rozwiazanie jawne
transposition_matrix = observation_matrix.T
theta_best = np.linalg.inv(transposition_matrix.dot(observation_matrix)).dot(transposition_matrix).dot(y_train)
print("\n Optymalna theta: " + str(theta_best))

a = theta_best[1]
b = theta_best[0]

# TODO: calculate error

# Funkcja kosztu - wzór 1.3
# Obliczenie błędu średniokwadratowego (MSE)

def calculate_mse(a, b):
    y_predicted = []
    for i in range(len(x_test)):
        y_predicted_i = a * x_test[i] + b
        y_predicted.append(y_predicted_i)

    return np.mean((np.array(y_predicted) - y_test) ** 2)

mse = calculate_mse(a, b)

print(f"Wspolczynnik a: {a:.10f}")
print(f"Wyraz wolny b: {b:.10f}")
print(f"Błąd średnio-kwadratowy MSE: {mse:.10f}")

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standarization

# Standaryzacja danych - wzór 1.15
standarized_x_train = (x_train - np.mean(x_train)) / np.std(x_train)
standarized_y_train = (y_train - np.mean(y_train)) / np.std(y_train)

# przy standaryzacji danych testowych do obliczenia sredniej i odchylenia standardowego
# danych testowych uzywamy zbioru treningowego, zeby nie dopuscic do wycieku danych testowych
standarized_x_test = (x_test - np.mean(x_train)) / np.std(x_train)
standarized_y_test = (y_test - np.mean(y_train)) / np.std(y_train)

x_train = standarized_x_train
y_train = standarized_y_train
x_test = standarized_x_test
y_test = standarized_y_test

# TODO: calculate theta using Batch Gradient Descent

learning_rate = 0.001
theta = np.array([0.0, 0.0])

# ponowne wyliczenie macierzy obserwacji (po standaryzacji danych)
observation_matrix = np.concatenate([np.ones((len(x_train), 1)), x_train.reshape(-1, 1)], axis=1)

for iteration in range(15000):

    #y_predicted = observation_matrix.dot(theta)
    y_predicted = theta[1] * x_train + theta[0]
    
    # gradient ze wzoru 1.7
    gradient = (-2 / len(y_train)) * observation_matrix.T.dot(y_train - y_predicted)
    
    # aktualizujemy thete zgodnie z metoda gradientu prostego, wzór - 1.14
    theta -= learning_rate * gradient

a = theta[1]  # wsp. a
b = theta[0]  # wyraz wolny b

print(f"a: {a:.4f}")
print(f"b: {b:.40f}")

# TODO: calculate error

mse = calculate_mse(a,b)
print(f"Błąd średnio-kwadratowy MSE: {mse:.10f}")

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()