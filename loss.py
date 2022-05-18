from tensorflow import math as m
import numpy as np


def bb_intersection_over_union(prediction, target):
    result = np.ones((7, 7, 30))
    for i in range(0, 7):
        for j in range(0, 7):
            intersection_width_A = max([0, (target[i, j, 3] + prediction[i, j, 3]) / 2 - abs(
                target[i, j, 1] - prediction[i, j, 1])])
            intersection_height_A = max([0, (target[i, j, 4] + prediction[i, j, 4]) / 2 - abs(
                target[i, j, 2] - prediction[i, j, 2])])

            area_of_the_intersection_A = intersection_width_A * intersection_height_A
            result[i, j, 0] = area_of_the_intersection_A / (prediction[i, j, 3] * prediction[i, j, 4])  # IOU_A

            intersection_width_B = max([0, (target[i, j, 8] + prediction[i, j, 8]) / 2 - abs(
                target[i, j, 6] - prediction[i, j, 6])])
            intersection_height_B = max([0, (target[i, j, 9] + prediction[i, j, 9]) / 2 - abs(
                target[i, j, 7] - prediction[i, j, 7])])

            area_of_the_intersection_B = intersection_width_B * intersection_height_B
            result[i, j, 1] = area_of_the_intersection_B / (prediction[i, j, 8] * prediction[i, j, 9])  # IOU_B
    return prediction * result


def loss(predictions, target):
    S = 7
    B = 2
    C = 20
    lambda_noobj = 0.5
    lambda_coord = 5
    predictions = bb_intersection_over_union(predictions, target)
    bounding_box_location = 0
    bounding_box_size = 0
    confidence_no_object = 0
    confidence_object = 0
    class_prob = 0
    for i in range(7):
        for j in range(7):
            for b in range(2):
                if target[i, j, b * 5] == 1:
                    confidence_object += m.square(predictions[i, j, 10:30] - target[i, j, 10:30])
                    bounding_box_location += (m.square(predictions[i, j, 1 + b * 5] - target[i, j, 1 + b * 5])
                                              + m.square(predictions[i, j, 2 + b * 5] - target[i, j, 2 + b * 5]))
                    bounding_box_size += (
                            m.square(m.sqrt(predictions[i, j, 3 + b * 5]) - m.sqrt(target[i, j, 3 + b * 5]))
                            + m.square(m.sqrt(predictions[i, j, 4 + b * 5]) - m.sqrt(target[i, j, 4 + b * 5])))
                else:
                    confidence_no_object += m.square(predictions[i, j, 10:30] - target[i, j, 10:30])
            if target[i, j, 0] == 1 or target[i, j, 5] == 1:
                class_prob += m.square(predictions[i, j, 10:30] - target[i, j, 10:30])

    loss = (
            + bounding_box_location * lambda_coord
            + bounding_box_size * lambda_coord
            + confidence_no_object * lambda_noobj
            + confidence_object
            + class_prob
    )

    return loss