import numpy as np

def decode_predictions(scores, geometry, min_confidence):
    detections = []
    confidences = []

    for i in range(0, scores.shape[2]):
        for j in range(0, scores.shape[3]):
            if scores[0, 0, i, j] < min_confidence:
                continue

            (offsetX, offsetY) = (j * 4.0, i * 4.0)
            angle = geometry[0, 4, i, j]
            cosA = np.cos(angle)
            sinA = np.sin(angle)
            h = geometry[0, 0, i, j] + geometry[0, 2, i, j]
            w = geometry[0, 1, i, j] + geometry[0, 3, i, j]

            endX = int(offsetX + (cosA * geometry[0, 1, i, j]) + (sinA * geometry[0, 2, i, j]))
            endY = int(offsetY - (sinA * geometry[0, 1, i, j]) + (cosA * geometry[0, 2, i, j]))
            startX = int(endX - w)
            startY = int(endY - h)

            detections.append((startX, startY, endX, endY))
            confidences.append(float(scores[0, 0, i, j]))

    return detections
