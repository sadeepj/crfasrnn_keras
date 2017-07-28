from crfrnn_model import get_crfrnn_model_def
import util


def main():
    input_file = "image.jpg"
    output_file = "labels.png"

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = "crfrnn_keras_model.h5"

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    img_data, img_h, img_w = util.get_preprocessed_image(input_file)
    probs = model.predict(img_data, verbose=False)[0, :, :, :]
    segmentation = util.get_label_image(probs, img_h, img_w)
    segmentation.save(output_file)


if __name__ == "__main__":
    main()
