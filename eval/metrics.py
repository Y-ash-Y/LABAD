# Metrics module

def accuracy(y_true, y_pred):
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
