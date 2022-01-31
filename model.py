import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Flatten, Dense, Activation, Input, SpatialDropout3D, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import regularizers
import config

def GreenBlock(inp, out_features, groups=8):
  residual = inp
  inp = GroupNormalization(groups, axis=-1)(inp)
  inp = Activation('relu')(inp)
  inp = Conv3D(out_features, 3, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(inp)
  inp = GroupNormalization(groups, axis=-1)(inp)
  inp = Activation('relu')(inp)
  inp = Conv3D(out_features, 3, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(inp)
  return residual + inp

def sampling(args):
  z_mean, z_var = args
  batch = K.shape(z_mean)[0]
  dim = K.int_shape(z_mean)[1]
  # by default, random_normal has mean = 0 and std = 1.0
  epsilon = K.random_normal(shape=(batch, dim))
  return z_mean + K.exp(0.5 * z_var) * epsilon


upper_inputs = Input(shape=config.INPUT_SHAPE)
x = Conv3D(32, 3, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(upper_inputs)
x = SpatialDropout3D(0.2)(x)
res1 = GreenBlock(x, 32, 8)
x = Conv3D(64, 3, 2, padding='same', kernel_regularizer=regularizers.l2(l2=1e-5))(res1)
x = GreenBlock(x, 64, 8)
res2 = GreenBlock(x, 64, 8)
x = Conv3D(128, 3, 2, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(res2)
x = GreenBlock(x, 128, 8)
res3 = GreenBlock(x, 128, 8)
x = Conv3D(256, 3, 2, 'same')(res3)
x = GreenBlock(x, 256, 8)
x = GreenBlock(x, 256, 8)
x = GreenBlock(x, 256, 8)
res4 = GreenBlock(x, 256, 8)
x = Conv3DTranspose(128, 1, 2, 'same')(res4)
x = res3 + x
x = GreenBlock(x, 128, 8)
x = Conv3DTranspose(64, 1, 2, 'same')(x)
x = res2 + x
x = GreenBlock(x, 64, 8)
x = Conv3DTranspose(32, 1, 2, 'same')(x)
x = res1 + x
x = Conv3D(config.N_CLASSES, 1, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(x)
out_upper = Activation('sigmoid', name='segmentation_output')(x)

upper = Model(upper_inputs, [out_upper, res4], name='upper_model')


lower_inputs = Input(shape=res4.shape[1:])

x = GroupNormalization(8, -1)(lower_inputs)
x = Activation('relu')(x)
x = Conv3D(16, 3, 2, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(x)
x = Dense(256, activation='relu')(x)
x = Flatten()(x)
z_mean = Dense(128, activation='relu', name='z_mean')(x)
z_var = Dense(128, activation='relu', name='z_var')(x)
x = Lambda(sampling)([z_mean, z_var])
x = Dense(config.DIM[0]//16 * config.DIM[1]//16 * config.DIM[2]//4 * config.NUM_CHANNELS//4, activation='relu')(x)
x = Reshape((config.DIM[0]//16, config.DIM[1]//16, config.DIM[2]//4, config.NUM_CHANNELS//4))(x)
x = Conv3D(256, 1, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(x)
x = Conv3DTranspose(256, 1, 2, 'same', name='VAE_output')(x)
x = Conv3D(128, 1, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(x)
x = Conv3DTranspose(128, 1, 2, 'same')(x)
x = GreenBlock(x, 128, 8)
x = Conv3D(128, 1, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(x)
x = Conv3DTranspose(64, 1, 2, 'same')(x)
x = GreenBlock(x, 64, 8)
x = Conv3D(64, 1, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5))(x)
x = Conv3DTranspose(32, 1, 2, 'same')(x)
out_lower = Conv3D(config.NUM_CHANNELS, 1, 1, 'same', kernel_regularizer=regularizers.l2(l2=1e-5), name='vae_output')(x)

lower = Model(lower_inputs, [z_mean, z_var, out_lower], name='lower_model')

class MyModel(tf.keras.Model):
  def __init__(self, upper, lower, **kwargs):
    super(MyModel, self).__init__()
    self.upper = upper
    self.lower = lower
    self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    self.dice_loss_tracker = tf.keras.metrics.Mean(name='dice_loss')
  
  @property
  def metrics(self):
    return [self.total_loss_tracker, self.reconstruction_loss_tracker,
            self.kl_loss_tracker, self.dice_loss_tracker]

  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
      out_upper, res4 = self.upper(x)
      z_mean, z_var, out_lower = self.lower(res4)

      l2_loss = K.mean(K.square(x - out_lower), axis=(1,2,3,4))
      # kl_loss = (K.sum((K.exp(z_var) + K.square(z_mean) - 1 - z_var), axis=-1)) / (160*192*4)

      kl_loss = (K.sum((K.square(z_mean) + K.square(z_var) - K.log(K.square(z_var) + 1e-16) - 1), axis=(-1))) / (config.DIM[0]*config.DIM[1]*config.DIM[2]*config.NUM_CHANNELS)
      intersection = 2 * K.sum((y * out_upper), axis=(1,2,3,4))
      dice_loss = 1 - intersection / (K.sum(K.square(y), axis=(1,2,3,4)) + K.sum(K.square(out_upper), axis=(1,2,3,4)) + 1e-16)

      total_loss = dice_loss + kl_loss + l2_loss

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(l2_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    self.dice_loss_tracker.update_state(dice_loss)

    return {'total_loss': self.total_loss_tracker.result(), 'dice_loss': self.dice_loss_tracker.result()}

model = MyModel(upper, lower)
model.compile(optimizer=Adam(learning_rate=1e-4))