import torch, itertools
import numpy as np
import matplotlib.pyplot as plt
from main import get_data, CNN
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm, classes, figsize, save_name, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, name='test'):
    plt.figure(figsize=figsize)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name + title, fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=17)
    plt.yticks(tick_marks, classes, fontsize=17)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=17)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=9)
    plt.xlabel('Predicted label', fontsize=9)
    plt.savefig(save_name, dpi=150)
    return cm

if __name__ == '__main__':
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_test, y_train, y_test, y_class, y_class_num = get_data()
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_dataset = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

    model = torch.load('model.pt').to(DEVICE)
    model.eval()

    y_class = [np.array(list(map(lambda x:'{:.0f}'.format(x), i))) for i in y_class]

    y_true1, y_pred1 = [], []
    y_true2, y_pred2 = [], []
    y_true3, y_pred3 = [], []
    with torch.no_grad():
        for x, y in test_dataset:
            y_true1.extend(list(y.cpu().detach().numpy()[:, 0].reshape((-1))))
            y_true2.extend(list(y.cpu().detach().numpy()[:, 1].reshape((-1))))
            y_true3.extend(list(y.cpu().detach().numpy()[:, 2].reshape((-1))))

            pred = model(x.to(DEVICE).float())
            y_pred1.extend(list(np.argmax(pred[0].cpu().detach().numpy(), axis=-1)))
            y_pred2.extend(list(np.argmax(pred[1].cpu().detach().numpy(), axis=-1)))
            y_pred3.extend(list(np.argmax(pred[2].cpu().detach().numpy(), axis=-1)))

    print(classification_report(y_true1, y_pred1, target_names=y_class[0]))
    cm = confusion_matrix(y_true1, y_pred1)
    plot_confusion_matrix(cm, y_class[0], (5, 5), 'cm1.png')

    print(classification_report(y_true2, y_pred2, target_names=y_class[1]))
    cm = confusion_matrix(y_true2, y_pred2)
    plot_confusion_matrix(cm, y_class[1], (50, 50), 'cm2.png')

    print(classification_report(y_true3, y_pred3, target_names=y_class[2]))
    cm = confusion_matrix(y_true3, y_pred3)
    plot_confusion_matrix(cm, y_class[2], (6, 6), 'cm3.png')
