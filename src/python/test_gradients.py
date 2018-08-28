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

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module


class HighDimGradTest(tf.test.TestCase):

    def test_high_dim_filter_grad(self):
        x_shape = [5, 10, 10]

        # Test inputs: unaries and RGB values
        unary_np = np.random.randn(*x_shape).astype(np.float32)
        rgb_np = np.random.randint(low=0, high=256, size=x_shape).astype(np.float32)

        with self.test_session():
            unary_tf = constant_op.constant(unary_np)
            rgb_tf = constant_op.constant(rgb_np)
            y_tf = custom_module.high_dim_filter(unary_tf, rgb_tf,
                                                 bilateral=True,
                                                 theta_alpha=1000.,
                                                 theta_beta=1000.,
                                                 theta_gamma=1000.)

            out = gradient_checker.compute_gradient([unary_tf, rgb_tf], [x_shape, x_shape],
                                                    y_tf, x_shape)

            # We only need to compare gradients w.r.t. unaries
            computed = out[0][0].flatten()
            estimated = out[0][1].flatten()

            mask = (computed != 0)
            computed = computed[mask]
            estimated = estimated[mask]
            difference = computed - estimated

            measure1 = np.mean(difference) / np.mean(computed)
            measure2 = np.max(difference) / np.max(computed)

            print('Gradient check: measure1 = {:.6f}, measure2 = {:.6f}'.format(measure1, measure2))
            self.assertLess(measure1, 1e-3, 'Errors found in the gradient computation.')
            self.assertLess(measure2, 2e-2, 'Errors found in the gradient computation.')
            print('Gradient check: success!')


if __name__ == '__main__':
    tf.test.main()
