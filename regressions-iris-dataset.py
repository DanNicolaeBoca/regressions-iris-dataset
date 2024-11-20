from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np


def main():
    def predictWithInput():
        # Predict an arbitrary value using the linear model
        inputed_value = input(
            "Insert value for the petal length to predict petal width with both linear and polynomial regression (centimeters) : "
        )
        arbitrary_value = np.array([[float(inputed_value)]])
        predicted_width_linear = linear_regressor.predict(arbitrary_value)
        print(
            f"Predicted Petal Width (Linear Regression) for Petal Length of {float(inputed_value)} cm: {predicted_width_linear[0][0]:.2f} cm"
        )

        # Predict an arbitrary value using the polynomial model
        predicted_width_poly = poly_regressor.predict(
            poly_features.transform(arbitrary_value)
        )
        print(
            f"Predicted Petal Width (Polynomial Regression) for Petal Length of {float(inputed_value)} cm: {predicted_width_poly[0][0]:.2f} cm"
        )

    # Load the Iris dataset
    iris = load_iris()

    # Here are the columns names indexed from 0 to 4 in order : sepal length, sepal width, petal length, petal width, class
    x = iris.data[:, 2].reshape(-1, 1)  # Petal Length
    y = iris.data[:, 3].reshape(-1, 1)  # Petal Width

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # Simple Linear Regression
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train, y_train)

    # Predict using the linear regression model
    y_pred_linear = linear_regressor.predict(x_test)

    while True:
        poly_degree = input(
            "Choose degree of polynomial to be used in the respective regression (natural number) : "
        )
        if poly_degree.isdigit():
            break
        else:
            print("Please choose a natural number.")
            continue

    poly_features = PolynomialFeatures(degree=int(poly_degree))
    x_poly_train = poly_features.fit_transform(x_train)

    # Fit the polynomial regression model
    poly_regressor = LinearRegression()
    poly_regressor.fit(x_poly_train, y_train)

    # Predict using the polynomial model
    x_poly_test = poly_features.transform(x_test)
    y_pred_poly = poly_regressor.predict(x_poly_test)

    # Generate a smooth curve for polynomial regression
    x_range = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)
    x_poly_range = poly_features.transform(x_range)

    while True:
        answer = input(
            "Print Linear and Polynomial regressions of the Irirs dataset? [Y/n] : "
        )
        if answer == "" or answer.lower() == "y" or answer.lower() == "yes":
            predictWithInput()

            # Plot the training data and linear regression line
            plt.figure(figsize=(10, 6))
            plt.scatter(x_train, y_train, color="black", label="Training data")
            plt.plot(
                x_train,
                linear_regressor.predict(x_train),
                color="red",
                label="Regression line",
            )
            plt.title("Simple Linear Regression (Petal Length vs Petal Width)")
            plt.xlabel("Petal Length (cm)")
            plt.ylabel("Petal Width (cm)")
            plt.legend()
            plt.show()

            # Plot the training data and polynomial regression curve
            plt.figure(figsize=(10, 6))
            plt.scatter(x_train, y_train, color="black", label="Training data")

            plt.plot(
                x_range,
                poly_regressor.predict(x_poly_range),
                color="red",
                label="Polynomial regression curve ",
            )

            plt.title("Polynomial Regression (Petal Length vs Petal Width)")
            plt.xlabel("Petal Length (cm)")
            plt.ylabel("Petal Width (cm)")
            plt.legend()
            plt.show()

            break

        elif answer.lower() == "n" or answer.lower() == "no":
            predictWithInput()
            break

        else:
            print(
                "Please give a valid answer (y,yes,n,no) or leave empty and press enter."
            )
            continue


if __name__ == "__main__":
    main()

    while True:
        answer = input("\nDo you want to restart the program? [Y/n] : ")
        if answer == "" or answer.lower() == "y" or answer.lower() == "yes":
            main()

        elif answer.lower() == "n" or answer.lower() == "no":
            break

        else:
            print(
                "\nPlease give a valid answer (y,yes,n,no) or leave empty and press enter."
            )
            continue
