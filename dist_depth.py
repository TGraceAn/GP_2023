from nicolai import depth_midas
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    n, ema = 0, np.zeros((480,640))
    K = 10
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv.flip(frame, 1)
        frame = cv.copyMakeBorder(
            frame,
            0, 40, 0, 0,
            cv.BORDER_CONSTANT, None, value=[255,255,255]
        )
        frame = cv.resize(frame, (640,480))
        depth = depth_midas(frame)

        # EMA
        n += 1
        if n <= K:
            ema += depth
            if n == K: ema /= K
            continue

        C = 2/(K+1)
        ema = depth * C + ema * (1-C)

        # normalization
        display = cv.normalize(
            ema, None, 0, 255,
            norm_type=cv.NORM_MINMAX,
            dtype=cv.CV_8U
        )
        display = cv.applyColorMap(display, cv.COLORMAP_MAGMA)

        cv.imshow('image', frame)
        cv.moveWindow('image', 0, 0)

        cv.imshow('depth', display)
        cv.moveWindow('depth', 640, 0)

        if cv.waitKey(1) == ord('q'): break

    cap.release()
    cv.destroyAllWindows()