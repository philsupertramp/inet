import matplotlib.pyplot as plt
import numpy as np

from src.data.load_dataset import directory_to_two_in_one_dataset
from src.losses.giou_loss import GIoULoss
from src.models.data_structures import BoundingBox
from src.models.tflite_methods import evaluate_interpreted_model
from tests.helper import Timer, build_tf_model_from_file


def filter_classes(classes_in):
    classes_out = []
    for i in range(classes_in.shape[0]):
        classes_out.append(classes_in.argmax())
    return classes_out


def process_best_prediction(prediction):
    processed_predictions = []
    for pred in prediction:
        preds = pred[0]
        preds = preds[preds[..., 5] > 0.25]
        max_conf = preds[..., 5].argmax()
        filtered_predictions = preds[max_conf]
        boxes = filtered_predictions[:4]
        scores = filtered_predictions[5]
        class_label = filtered_predictions[5:]
        b1_x1, b1_x2 = boxes[0] - boxes[2] / 2, boxes[0] + boxes[2] / 2
        b1_y1, b1_y2 = boxes[1] - boxes[3] / 2, boxes[1] + boxes[3] / 2
        yxhw = np.array([b1_y1, b1_x1, (b1_y2 - b1_y1), (b1_x2 - b1_x1)]) * 100.
        processed_predictions.append([yxhw, class_label, scores])
    return processed_predictions


def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]# * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]# * image_height

    #bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2

    return bboxes


def scale_bb(bb, h, w):
    bb_vals = np.array(bb) / 100.
    bb_vals[::2] *= w
    bb_vals[1::2] *= h
    return bb_vals


if __name__ == '__main__':
    mytimer = Timer()

    tfmodel = build_tf_model_from_file('weights/yolo-best-fp16.tflite')

    test_set, train_set, validation_set = directory_to_two_in_one_dataset('data/iNat/data', img_width=640,
                                                                          img_height=640)
    test_images, test_labels = tuple(zip(*test_set.unbatch()))
    test_images = np.array(test_images)
    o_test_images = test_images.copy()
    test_labels = np.array(test_labels)
    with mytimer:
        predictions = evaluate_interpreted_model(tfmodel, test_images)

    print(f'Inference time: {mytimer.results[0].microseconds / 100_000} seconds')

    index = 0
    gloss_fn = GIoULoss()
    gious = []
    for i, batch in enumerate(predictions):
        for sample in batch:
            img = o_test_images[index]
            sample = sample[sample[:, 4] >= 1.]  # [-1:, :]
            bbs = yolo2voc(img.shape[0], img.shape[1], sample[:, :4])
            conf = sample[:, 4:5]
            cls = sample[:, 5:]
            # Do whatever you want

            #cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[0], 2)
            fig = plt.figure()
            gc = plt.gca()
            gc.imshow(img/255.)
            current_gious = []
            for bb in bbs:
                x, y, x2, y2 = bb
                bb_obj = BoundingBox(x*100., y*100., (x2 - x)*100., (y2 - y)*100.)
                scaled_bb = BoundingBox(*scale_bb(bb_obj.values, img.shape[0], img.shape[1]))
                scaled_bb.draw(gc)
                current_gious.append(bb_obj.GIoU(BoundingBox(*test_labels[i][1])))

            if current_gious:
                gious.append(sum(current_gious)/len(current_gious))

            index += 1
            plt.show()
    print('AVG GIoU:', sum(gious)/len(gious))
