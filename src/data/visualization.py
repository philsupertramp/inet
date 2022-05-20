import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from scripts.constants import CLASS_MAP, LABEL_MAP
from src.models.data_structures import BoundingBox


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False, colormap=plt.cm.cool_r,
                          title=None):
    """

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    potentially good color maps from matplotlib:

    Color maps to visualize positive cases
    ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
     'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot_r', 'autumn_r', 'binary', 'bone_r', 'cividis_r', 'cool_r', ]

     Color maps to visualize negative cases
    ['Wistia', 'brg_r', 'bwr_r']

    :param y_true: array of ground truth values
    :param y_pred: predictions done by a model
    :param classes: verbatim class names
    :param normalize: use normalized confusion matrix
    :param colormap: the color map to use
    :param title: the title for the resulting plot
    :return: a matplotlib.pyplot.axis object containing the generated plot
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    classes = np.array(classes)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = np.nan_to_num(cm.astype('float')) / np.nan_to_num(cm.sum(axis=1))[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig, ax = plt.subplots()
    fig.set_dpi(250)
    im = ax.imshow(cm, interpolation='nearest', cmap=colormap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    ax.set_title(title, fontsize=20)
    ax.set_ylabel('True label', fontsize=16)
    ax.set_xlabel('Predicted label', fontsize=16)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()


def plot_img_bbs(image, prediction, true_label):
    height, width, _ = image.shape

    def scale_bbs(y, x, h, w):
        return x * width / 100., y * height / 100., w * width / 100., h * height / 100.

    plt.imshow(image)
    # bbs = scale_label(bbs)
    scaled_bbs = BoundingBox(*scale_bbs(*prediction))
    scaled_bbs2 = BoundingBox(*scale_bbs(*true_label))
    scaled_bbs.draw(plt.gca(), 'red')
    scaled_bbs2.draw(plt.gca(), 'green')
    plt.axis('off')


def plot_img_labels(image, predicted_label, true_label):
    plt.imshow(image)
    plt.xlabel(predicted_label)
    plt.ylabel(true_label)


def plot_histories(hists, keys, titles):
    def plot_val(h, n, k):
        plt.plot(np.arange(0, len(h.history[k])), h.history[k], '--', label=f'{n} train_{k}')
        plt.plot(np.arange(0, len(h.history[f'val_{k}'])), h.history[f'val_{k}'], label=f'{n} val_{k}')

    for i, key in enumerate(keys):
        fig = plt.figure(figsize=(7.5, 7.5))
        plt.title(titles[i], fontsize=25)
        plt.xlabel('Epoch #', fontsize=18)
        plt.ylabel(key.capitalize(), fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        for name, hist in hists.items():
            plot_val(hist, name, key)

        if len(hists.keys()) > 2:
            fig.legend(
                loc='center left',  # Position of legend,
                borderaxespad=0.1,  # Small spacing around legend box
                bbox_to_anchor=(0.95, 0.5, 0, 0),
                fontsize=12
            )
        else:
            plt.legend(loc='best', fontsize=12)
        plt.show()


def plot_bbox_samples(predictions, validation_labels, validation_features, img_width=224, img_height=224):
    index = 0
    fig = plt.figure(figsize=(15, 15))
    for i, pred in enumerate(predictions):
        if index == 25:
            break
        index += 1
        plt.subplot(5, 5, index)

        scaled_img = validation_features[i] / 255.

        plot_img_bbs(
            scaled_img.reshape(img_height, img_width, 3),
            pred,
            list(validation_labels[i])
        )

    fig.legend(
        ['Prediction', 'Ground truth'],
        loc='upper center',  # Position of legend
        borderaxespad=0.1,  # Small spacing around legend box
        title='Bounding Boxes',  # Title for the legend,
        fontsize=25
    )
    plt.show()


def plot_prediction(image, bb, label, color='y'):
    height, width, _ = image.shape

    def scale_bbs(y, x, h, w):
        return x * width / 100., y * height / 100., w * width / 100., h * height / 100.

    plt.imshow(image)
    scaled_bbs = BoundingBox(*scale_bbs(*bb))
    scaled_bbs.draw(plt.gca(), color)
    plt.gca().text(
        scaled_bbs.x_min + 0.02 * width,
        scaled_bbs.y_min - height*0.05,
        label,
        backgroundcolor=color,
        fontsize=10,
        bbox={'color': color}
    )
    plt.axis('off')


def plot_prediction_samples(predicted_bbs, predicted_labels, validation_features, img_width=224, img_height=224,
                            include_score=True):
    index = 0
    fig = plt.figure(figsize=(15, 15))

    for i, pred in enumerate(predicted_bbs):
        if index == 25:
            break
        index += 1
        plt.subplot(5, 5, index)

        scaled_img = validation_features[i] / 255.

        if isinstance(predicted_labels[i], np.int64):
            label_str = f'{LABEL_MAP.get(predicted_labels[i])}'
        else:
            label_str = f'{LABEL_MAP.get(np.argmax(predicted_labels[i]))}'
        if include_score:
            label_str += f' ({np.max(predicted_labels[i]) * 100.:.2f}%)'

        plot_prediction(
            scaled_img.reshape(img_height, img_width, 3),
            pred,
            label_str
        )

    fig.legend(
        ['Prediction'],
        loc='upper center',  # Position of legend
        borderaxespad=0.1,  # Small spacing around legend box
        title='Bounding Boxes',  # Title for the legend,
        #fontsize=25
    )
