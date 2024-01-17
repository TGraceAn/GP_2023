import utils
import cv2
import numpy as np
import onnx
import onnxruntime as rt

from transforms import Resize, NormalizeImage, PrepareForNet


def run(img_path, model_path):
    model = rt.InferenceSession(model_path, providers = ['AzureExecutionProvider', 'CPUExecutionProvider'])
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    net_w, net_h = 256, 256
    resize_image = Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
    
    def compose2(f1, f2):
        return lambda x: f2(f1(x))

    transform = compose2(resize_image, PrepareForNet())

    img = utils.read_image(img_path)
    img_input = transform({"image": img})["image"]

    import time
    s = time.time()
    output = model.run([output_name], {input_name: img_input.reshape(1, 3, net_h, net_w).astype(np.float32)})[0]
    print(f'Time: {time.time() - s}')
    prediction = np.array(output).reshape(net_h, net_w)
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    print(np.max(prediction))
    return prediction


if __name__ == "__main__":
    img_path = 'input/test.png'
    model_path = 'model-small.onnx'
    run(img_path, model_path)
