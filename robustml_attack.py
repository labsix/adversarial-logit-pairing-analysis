import robustml
import tensorflow as tf
import numpy as np
import sys

class NullAttack(robustml.attack.Attack):
    def run(self, x, y, target):
        return x

class PGDAttack(robustml.attack.Attack):
    def __init__(self, sess, model, epsilon, max_steps=100, step_size=0.01, quantize=False, debug=False):
        self._sess = sess
        self._model = model
        self._epsilon = epsilon
        self._max_steps = max_steps
        self._step_size = step_size
        self._quantize = quantize
        self._debug = debug

        self._label = tf.placeholder(tf.int32, ())
        one_hot = tf.expand_dims(tf.one_hot(self._label, 1000), axis=0)
        self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=one_hot)
        self._grad, = tf.gradients(self._loss, model.input)

    def run(self, x, y, target):
        mult = -1
        untargeted = not target
        if target is None:
            target = y
            mult = 1
        lower = np.clip(x - self._epsilon, 0, 1)
        upper = np.clip(x + self._epsilon, 0, 1)
        adv = x + np.random.uniform(low=-self._epsilon, high=self._epsilon, size=x.shape)
        adv = np.clip(adv, lower, upper)
        for i in range(self._max_steps):
            if self._quantize:
                adv_eval = (adv*255).astype(np.uint8).astype(np.float32)/255.0
            else:
                adv_eval = adv
            p, l, g = self._sess.run(
                [self._model.predictions, self._loss, self._grad],
                {self._model.input: [adv_eval], self._label: target}
            )
            if self._debug:
                print(
                    'attack: step %d/%d, loss = %g (true %d, predicted %d, target %d)' % (
                        i+1,
                        self._max_steps,
                        l,
                        y,
                        p,
                        target
                    ),
                    file=sys.stderr
                )
            if untargeted and p != y or not untargeted and p == target:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break
            adv += mult * self._step_size * np.sign(g[0])
            adv = np.clip(adv, lower, upper)
        return adv
