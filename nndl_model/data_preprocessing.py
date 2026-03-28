import numpy as np

def load_data():
    X_train = np.loadtxt('C:/Users/karuj/OneDrive/Desktop/project/data/UCI HAR Dataset/train/X_train.txt')
    y_train = np.loadtxt('C:/Users/karuj/OneDrive/Desktop/project/data/UCI HAR Dataset/train/y_train.txt')

    X_test = np.loadtxt('C:/Users/karuj/OneDrive/Desktop/project/data/UCI HAR Dataset/test/X_test.txt')
    y_test = np.loadtxt('C:/Users/karuj/OneDrive/Desktop/project/data/UCI HAR Dataset/test/y_test.txt')

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)