import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.title("LSTM Training Accuracy")
    plt.savefig("outputs/accuracy_curve.png")
    plt.show()
