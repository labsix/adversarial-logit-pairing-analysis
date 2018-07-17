from robustml_model import *
from robustml_attack import *
import tensorflow as tf
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, required=True,
            help='directory containing `val.txt` and `val/` folder')
    parser.add_argument('--checkpoint-path', type=str, default='imagenet64_alp025_2018_06_26.ckpt',
            help='path to imagenet64 checkpoint')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--attack', type=str, default='pgd', help='none | pgd')
    parser.add_argument('--attack-iterations', type=int, default=1000)
    parser.add_argument('--attack-step-size', type=float, default=0.005)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()

    sess = tf.Session()

    model = ALP(sess, args.checkpoint_path, quantize=args.quantize)

    if args.attack == 'none':
        attack = NullAttack()
    elif args.attack == 'pgd':
        attack = PGDAttack(
            sess,
            model,
            model.threat_model.epsilon,
            debug=args.debug,
            max_steps=args.attack_iterations,
            step_size=args.attack_step_size,
            quantize=args.quantize
        )
    else:
        raise ValueError('invalid attack: %s' % args.attack)

    provider = robustml.provider.ImageNet(args.imagenet_path, shape=(64, 64, 3))

    success_rate = robustml.evaluate.evaluate(
        model,
        attack,
        provider,
        start=args.start,
        end=args.end,
        deterministic=True,
        debug=True
    )

    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end - args.start))

if __name__ == '__main__':
    main()
