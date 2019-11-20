"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from crfrnn_model import get_crfrnn_model_def
import util
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='full path to the .h5 model (download from https://goo.gl/ciEYZi)',
                        required=True)
    parser.add_argument('--image', help='full path to the image', required=True)
    parser.add_argument('--output', help='full path to the output label image', default=None)
    args = parser.parse_args()

    saved_model_path = args.model
    input_file = args.image
    output_file = args.output or input_file + '_labels.png'

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    img_data, img_h, img_w, original_size = util.get_preprocessed_image(input_file)
    probs = model.predict(img_data, verbose=False)[0]
    segmentation = util.get_label_image(probs, img_h, img_w, original_size)
    segmentation.save(output_file)


if __name__ == '__main__':
    main()
