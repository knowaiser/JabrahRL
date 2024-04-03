import tensorflow as tf
import numpy as np

# Example log directory - ensure this matches your TensorBoard logdir path
log_dir = 'C:/Users/g201901650/Desktop/ddpg/JabrahRL/self_driving_agent/logs/ddpg'
writer = tf.summary.create_file_writer(log_dir)

with writer.as_default():
    for step in range(100):
        # Example scalar value - replace with your actual training metric
        tf.summary.scalar('my_metric', np.sin(step / 10), step=step)
        writer.flush()