import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def draw_confusion_matrix(y_true, y_pred, save_path=None):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues', annot_kws={"size": 8})
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    title = f'Accuracy Score: {np.sum(y_true == y_pred) / len(y_true):.3f}, based on {len(y_true)} test samples'
    plt.title(title, size=15)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()



def draw_plot_accuracy(accuracy_train, accuracy_validation, saved_models=None, save_path=None):
    plt.figure(figsize=(10, 10))
    if saved_models is not None:
        for i, was_saved in enumerate(saved_models):
            if was_saved:
                # Mark the corresponding point. Adjust the marker style as needed.
                plt.scatter(i, accuracy_train[i], marker='o', color='green')
                plt.scatter(i, accuracy_validation[i], marker='o', color='red')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy (o marks indicates moments of saving model)')
    plt.plot(accuracy_train, label='accuracy_train')
    plt.plot(accuracy_validation, label='accuracy_validation')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def draw_plot_loss(loss_train, loss_validation, saved_models=None, save_path=None):
    plt.figure(figsize=(10, 10))
    if saved_models is not None:
        for i, was_saved in enumerate(saved_models):
            if was_saved:
                # Mark the corresponding point. Adjust the marker style as needed.
                plt.scatter(i, loss_train[i], marker='o', color='green')
                plt.scatter(i, loss_validation[i], marker='o', color='red')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss (o marks indicates moments of saving model)')
    plt.plot(loss_train, label='loss_train')
    plt.plot(loss_validation, label='loss_validation')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()