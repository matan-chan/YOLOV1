import tensorflow as tf
from model import build_model
from loss import loss
from keras.optimizers import Adam
from utils import get_batch


def train_step(checkpoint, index, batch_size=64):
    images, labels = get_batch(batch_size)
    with tf.GradientTape() as model_tape:
        prediction = checkpoint.model(images, training=True)
        model_loss = loss(prediction, labels)

    gradients_of_model = model_tape.gradient(model_loss, checkpoint.model.trainable_variables)
    checkpoint.optimizer.apply_gradients(
        zip(gradients_of_model, checkpoint.model.trainable_variables))

    print(f'epoch: {index} model loss: {model_loss}')


def train():
    checkpoint_dir = 'models/'
    save_every = 100
    model = build_model()
    optimizer = Adam(learning_rate=1e-4)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
    )

    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    epoch_start = int(manager.latest_checkpoint.split(sep='ckpt-')[-1]) * save_every if manager.latest_checkpoint else 0
    print('starting at:', epoch_start)
    checkpoint.restore(manager.latest_checkpoint)

    for epoch in range(epoch_start + 1, 200_000 + 1):
        train_step(checkpoint, epoch)
        if epoch % save_every == 0:
            manager.save()


train()
