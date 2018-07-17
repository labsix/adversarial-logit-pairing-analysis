import robustml
import tensorflow as tf
import numpy as np
import model_lib
from datasets import imagenet_input

class ALP(robustml.model.Model):
    '''
    ALP for ImageNet 64x64
    '''
    def __init__(self, sess, checkpoint_path, quantize=False):
        self._sess = sess
        self._input = tf.placeholder(tf.float32, (None, 64, 64, 3))
        self._logits = _model(sess, self._input, checkpoint_path)
        self._logits = self._logits[:, 1:] # ignore background class
        self._predictions = tf.argmax(self._logits, 1)
        self._dataset = robustml.dataset.ImageNet((64, 64, 3))
        self._threat_model = robustml.threat_model.Linf(epsilon=16.0/255.0, targeted=True)
        self._quantize = quantize

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        if self._quantize:
            x = (x*255).astype(np.uint8).astype(np.float32)/255.0
        return self._sess.run(self._predictions, {self._input: [x]})[0]

    # exposing some internals to make it less annoying for attackers to do a
    # white-box attack

    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions

def _model(sess, input_, checkpoint_path):
    model_fn_two_args = model_lib.get_model('resnet_v2_50', 1001)
    model_fn = lambda x: model_fn_two_args(x, is_training=False)
    preprocessed = imagenet_input._normalize(input_)
    logits = model_fn(preprocessed)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_path)
    return logits
