from model import Integration
import cv2
from utils.utils import draw_detections
import matplotlib.pyplot as plt

model = Integration()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        # Preprocess the image
        shape = (480,640,3)
        frame = cv2.resize(frame, (shape[1] , shape[0]))
        frame = frame/255


        bounding_box, scores, cls_idx, _ = model(frame)

        combined_img = draw_detections(frame, bounding_box, scores, cls_idx)
        # Display the resulting frame

        cv2.imshow('Object', combined_img)
        cv2.imshow('Depth', _)

        if cv2.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()


def object_onnx_run(img, model):
    
    arr = np.expand_dims(img, 0)
    arr = np.array(arr, dtype = np.float32)
    arr = np.array(np.transpose(arr, (0, 3, 1, 2)), dtype=np.float32)

    bounding_box, scores, cls = model(arr)
    
    return bounding_box, scores, cls