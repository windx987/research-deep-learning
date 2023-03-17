import matplotlib.pyplot as plt

def loss_curves(epochs_count, trains_loss_values, tests_loss_values):
    # Plot the loss curves
    plt.plot(epochs_count, trains_loss_values, label="Train loss")
    plt.plot(epochs_count, tests_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(prop={"size": 14})
    plt.show()

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14});
    plt.show()