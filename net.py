import random as rand


class Net:
    def __init__(self, data):  # Initializes net
        self.data = data
        self.weights = []

    def preprocess_data(self, data):  # Seperates training data into inputs and outputs
        self.X = []
        self.Y = []
        for r in range(len(data)):
            for c in range(len(data[r]) - 1):
                self.X.append(data[r][c])
                self.Y.append(data[r][c + 1])

        return self.X, self.Y

    def init_weights(self):  # Initializes random weights (0-1)
        self.weights = [rand.random() for i in range(len(self.X))]
        return self.weights

    def SquaredError(self, y_hat, Y):  # Calculated distance from label and squares it
        self.loss = (y_hat - Y) ** 2
        self.dw = 2 * (y_hat - Y) * self.X[self.n]
        return self.loss

    def forward(self, X, W):  # Multiplies input by weight
        self.y_hat = X * W
        return self.y_hat

    def gradient_descent(
        self, W, learning_rate
    ):  # Updates weights using gradient descent
        W = W - (learning_rate * self.dw)
        return W

    def train(self):  # Trains model using forward prop, loss, and gradient descent
        self.learning_rate = 0.1
        self.epochs = 100
        self.n = 0
        self.input = self.X[self.n]

        for epoch in range(0, self.epochs):
            self.out = self.forward(self.input, self.weights[self.n])
            self.loss = self.SquaredError(self.out, self.Y[self.n])
            self.weights[self.n] = self.gradient_descent(
                self.weights[self.n], self.learning_rate
            )
            print(f"\nEpoch: {epoch+1}/{self.epochs} Loss: {self.loss}")
            print(
                f"{self.input} x {self.Y[self.n]} = {self.input*2} --> {self.input} x {self.weights[self.n]} = {self.out}"
            )
        print(
            "------------------------------------------------------------------------"
        )

    def eval(
        self, data
    ):  # Uses trained model to input random inputs and check the accuracy of the model
        self.main()
        self.correct = 0

        for input in data:
            print(f"\n{input} x {2} = {self.forward(input, self.weights[self.n])}")
            if self.loss < 0.1:
                self.correct += 1
        print(f"Eval Accuracy: {(self.correct / len(data)) * 100}%")

    def main(self):  # Cleans data, initializes the weights and then trains the model
        self.preprocess_data(self.data)
        self.init_weights()
        self.train()


train_data = [
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10],
    [6, 12],
]  # X = input, Y = output[[X, Y]]
test_data = [rand.randint(0, 50) for i in range(12)]  # Random sample of inputs

myNet = Net(train_data)  # Initialize Neural Network

myNet.main()  # Run main
myNet.eval(test_data)  # Evaluate model after training
