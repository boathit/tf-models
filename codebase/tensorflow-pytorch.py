import tensorflow as tf
import torch

################################################################################
tf.expand_dims, tf.newaxis
torch.unsqueeze

x = tf.constant([1, 2, 3])
tf.expand_dims(x, 1)
x[:, tf.newaxis]

a = torch.tensor([1, 2, 3])
torch.unsqueeze(a, 1)
################################################################################
tf.squeeze
torch.squeeze

x = tf.random.uniform([1, 3])
tf.squeeze(x, axis=0)

a = torch.rand([1, 3])
torch.squeeze(a, dim=0)
################################################################################
tf.concat
torch.cat

x = tf.constant([[1, 2, 3], [2, 3, 1]])
y = tf.constant([[0, 0, 1], [1, 0, 0]])
tf.concat([x, y], axis=1)

a = torch.tensor([[1, 2, 3], [2, 3, 1]])
b = torch.tensor([[0, 0, 1], [1, 0, 0]])
torch.cat([a, b], dim=1)
################################################################################
tf.stack
torch.stack

x = tf.constant([[1, 2, 3]])
y = tf.constant([[0, 0, 1]])
tf.stack([x, y], axis=2)

a = torch.tensor([[1, 2, 3]])
b = torch.tensor([[0, 0, 1]])
torch.stack([a, b], dim=2)
################################################################################
tf.random.uniform([2, 3])

torch.rand([2, 3])
################################################################################
tf.random.uniform([2, 3], minval=2, maxval=10, dtype=tf.int32)
torch.randint(low=2, high=10, size=[2, 3])
################################################################################
x = tf.random.uniform([2, 10])
tf.split(x, 2, axis=1)

a = torch.rand([2, 10])
torch.split(a, 5, dim=1)
################################################################################
