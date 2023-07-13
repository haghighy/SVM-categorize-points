import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, alpha=0.001, landa=0.01, iterationCount=1000):
        self.lr = alpha #learning rate
        self.landa = landa
        self.iterationCount = iterationCount
        self.w = None
        self.b = None

    def SVMfit(self, X, y):
        SamplesCount, featuresCount = X.shape
        y2 = []
        for l in y:
            if l <= 0:
                y2.append(-1)
            else:
                y2.append(1)
        self.w = np.zeros(featuresCount)
        self.b = 0
        for i in range(self.iterationCount):
            for idx in range(len(X)): 
                if y2[idx] * (np.dot(X[idx], self.w) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.landa * self.w)
                else:
                    self.w -= self.lr * (2 * self.landa * self.w - np.dot(X[idx], y2[idx]))
                    self.b -= self.lr * y2[idx]
    def ResultPredict(self, X):
        a = np.dot(X, self.w) - self.b
        return np.sign(a)

def drawSVM():
    fg = plt.figure()
    nemoodar = fg.add_subplot(1, 1, 1)
    for i in range(len(y)):
        if y[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], marker="o", color = "green")
        else:
            plt.scatter(X[i, 0], X[i, 1], marker=".", color = "red")
    Xpoint_1 = np.amin(X[:, 0])
    Xpoint_2 = np.amax(X[:, 0])
    Ypoint_min = np.amin(X[:, 1])
    Ypoint_max = np.amax(X[:, 1])
    def ClassValueGet(x, w, b, offset):
        return (-w[0]*x+b+offset)/w[1]
    Ypoint_1 = ClassValueGet(Xpoint_1, svm.w, svm.b, 0)
    Ypoint_2 = ClassValueGet(Xpoint_2, svm.w, svm.b, 0)
    Ypoint_1_m = ClassValueGet(Xpoint_1, svm.w, svm.b, -1)
    Ypoint_2_m = ClassValueGet(Xpoint_2, svm.w, svm.b, -1)
    Ypoint_1_p = ClassValueGet(Xpoint_1, svm.w, svm.b, 1)
    Ypoint_2_p = ClassValueGet(Xpoint_2, svm.w, svm.b, 1)

    nemoodar.plot([Xpoint_1, Xpoint_2], [Ypoint_1, Ypoint_2], "k")
    nemoodar.plot([Xpoint_1, Xpoint_2], [Ypoint_1_m, Ypoint_2_m], "b--")
    nemoodar.plot([Xpoint_1, Xpoint_2], [Ypoint_1_p, Ypoint_2_p], "b--")
    nemoodar.set_ylim([Ypoint_min-1, Ypoint_max+1])

X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
# X, y = datasets.make_blobs(n_samples=70, n_features=2, centers=2, cluster_std=0.65, random_state=50)
# X, y = datasets.make_blobs(n_samples=30, n_features=2, centers=2, cluster_std=0.20, random_state=30)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
svm = SVM()
svm.SVMfit(X_train, y_train)
res_predictions = svm.ResultPredict(X_test)

deghat = np.sum(y_test == res_predictions) / len(y_test)
print("-deghat class bandi: ", deghat)

drawSVM()
plt.show()

