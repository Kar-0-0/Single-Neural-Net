import random as rand


class Net:
    def __init__(self, data, layers=2):
        self.data = data
        self.layers = layers
        self.weights = []

    def preprocess_data(self, data):
        self.X = []
        self.Y = []
        for r in range(len(data)):
            for c in range(len(data[r]) - 1):
                self.X.append(data[r][c])
                self.Y.append(data[r][c + 1])

        return self.X, self.Y

    def init_weights(self):
        self.weights = [rand.random() for i in range(len(self.X))]
        return self.weights

    def MSE(self, y_hat, Y):
        self.loss = (y_hat - Y) ** 2
        self.dw = 2 * (y_hat - Y) * self.X[self.n]
        return self.loss

    def forward(self, X, W):
        self.y_hat = X * W
        return self.y_hat

    def backward(self, W, learning_rate):
        W = W - (learning_rate * self.dw)
        return W

    def train(self):
        self.learning_rate = 0.1
        self.epochs = 100
        self.n = 0
        self.input = self.X[self.n]

        for epoch in range(0, self.epochs):
            self.out = self.forward(self.input, self.weights[self.n])
            self.loss = self.MSE(self.out, self.Y[self.n])
            self.weights[self.n] = self.backward(
                self.weights[self.n], self.learning_rate
            )
            print(f"\nEpoch: {epoch+1}/{self.epochs} Loss: {self.loss}")
            print(f"1 x 2 = 2 --> {self.input} x {self.weights[self.n]} = {self.out}")
        print(
            "------------------------------------------------------------------------"
        )

    def eval(self, data):
        self.main()
        self.correct = 0

        for input in data:
            print(f"\n{input} x {2} = {self.forward(input, self.weights[self.n])}")
            if self.loss < 0.1:
                self.correct += 1
        print(f"Eval Accuracy: {(self.correct / len(data)) * 100}%")

    def main(self):
        self.preprocess_data(self.data)
        self.init_weights()
        self.train()


train_data = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12]]
test_data = [rand.randint(0, 50) for i in range(12)]

myNet = Net(train_data)

myNet.main()
myNet.eval(test_data)
