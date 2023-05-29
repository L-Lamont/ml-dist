import argparse

import tensorflow as tf
from tensorflow.keras import layers
import horovod.tensorflow.keras as hvd

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=16, metavar='N',
                        help='number of epochs to train (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--steps-per-epoch', type=int, default=500,
                        help='steps per epoch for training (default: 500)')
    # Unsupported argument kept for compatability
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # Unsupported argument kept for compatability
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # Unsupported argument kept for compatability
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-dir', default='./data',
                        help='Directory dataset is in')
    parser.add_argument('--output-dir', default='./output/default',
                        help='Specify where to save the current model and log')
    args = parser.parse_args()

    if len(tf.config.list_physical_devices('GPU')) < 1 or args.no_cuda:
        print("CPU tensorflow unsupported")
        return 1

    # Initialize horovod
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    args.lr = args.lr * hvd.size()

    tf.random.set_seed(args.seed)

    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='{}/mnist-{}.npz'.format(args.data_dir, hvd.rank()))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
                 tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)

    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=opt,
        metrics=['accuracy']
    )

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
    ]

    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    model.fit(
        dataset,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        callbacks=callbacks
    )

    if args.save_model and hvd.rank() == 0:
        tf.keras.Sequential.save("{}/mnist_model".format(args.output_dir))


if __name__ == '__main__':
    main()
