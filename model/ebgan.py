from __future__ import absolute_import
import numpy as np
import tensorflow as tf

_BATCH_SIZE_ = 64
_HIDDEN_LAYER_DIM_ = 1000

# convolution/pool stride
_CONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_DECONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_REGULAR_FACTOR_ = 1.0e-4

def _construct_full_connection_layer(input, output_dim, stddev = 0.02, name = 'fc_layer'):
	with tf.variable_scope(name):
		init_weight = tf.random_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		weight = tf.get_variable(
			name = name + '_weight',
			shape = [input.get_shape()[1], output_dim],
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + '_bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0),
			regularizer = None)
		fc = tf.matmul(input, weight)
		fc = tf.nn.bias_add(fc, bias)
		return fc

# define EBGAN network
# define generator network
class Generative:
	def __init__(self, name = 'generator'):
		self.name = name

	def inference(self, z, output_image_size = 28, output_image_channel = 1, reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			print 'generator input z:', z.get_shape()

			hidden0 = tf.nn.relu(_construct_full_connection_layer(z, _HIDDEN_LAYER_DIM_, name = 'de_hidden0'))
			hidden1 = tf.nn.relu(_construct_full_connection_layer(hidden0, _HIDDEN_LAYER_DIM_, name = 'de_hidden1'))
			output_layer = tf.nn.tanh(_construct_full_connection_layer(hidden1, output_image_size * output_image_size * output_image_channel, name = 'de_hidden2'))

			output_layer_reshape = tf.reshape(output_layer, [-1, output_image_size, output_image_size, output_image_channel])

			print "decoder output :", output_layer_reshape.get_shape()

			return output_layer_reshape

# define disctriminative network
# it an auto-encoder netwrok
class Discriminative:
	def __init__(self, name = 'discriminator'):
		self.name = name
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.embedding = None
		self.decode_image = None

	def inference(self, input_images, embedding_size = 2, reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			input_images_shape = input_images.get_shape().as_list()
			image_size = input_images_shape[1]
			image_channel = input_images_shape[3]
			self.embedding = self.encoder.inference(input_images, embedding_size, reuse)
			self.decode_image = self.decoder.inference(self.embedding, image_size, image_channel, reuse)
			
			return self.decode_image, self.embedding

class Encoder:
	def __init__(self, name = 'encoder'):
		self.name = name

	def inference(self, images, output_dim, reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			print 'encoder input', images.get_shape()
			if len(images.get_shape()) > 2:
				dim = 1
				for d in images.get_shape().as_list()[1:]:
					dim *= d
				images_reshape = tf.reshape(images, [-1, dim])
			else:
				images_reshape = hidden1

			hidden0 = tf.nn.relu(_construct_full_connection_layer(images_reshape, _HIDDEN_LAYER_DIM_, name = 'en_hidden0'))
			hidden1 = tf.nn.relu(_construct_full_connection_layer(hidden0, _HIDDEN_LAYER_DIM_, name = 'en_hidden1'))
			output_layer = _construct_full_connection_layer(hidden1, output_dim, name = 'en_hidden2')
			
			print "encoder output:", output_layer.get_shape()

			return output_layer

class Decoder:
	def __init__(self, name = 'decoder'):
		self.name = name

	def inference(self, input_embedding, output_image_size, output_image_channel, reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			print 'decoder input embedding:', input_embedding.get_shape()
			
			hidden0 = tf.nn.relu(_construct_full_connection_layer(input_embedding, _HIDDEN_LAYER_DIM_, name = 'de_hidden0'))
			hidden1 = tf.nn.relu(_construct_full_connection_layer(hidden0, _HIDDEN_LAYER_DIM_, name = 'de_hidden1'))
			output_layer = tf.nn.tanh(_construct_full_connection_layer(hidden1, output_image_size * output_image_size * output_image_channel, name = 'de_hidden2'))

			output_layer_reshape = tf.reshape(output_layer, [-1, output_image_size, output_image_size, output_image_channel])

			print "decoder output :", output_layer_reshape.get_shape()

			return output_layer_reshape

class EBGAN:
	def __init__(self, generator_name = 'generator', discriminator_name = 'discriminator'):
		self.generator_name = generator_name
		self.discriminator_name = discriminator_name

	def inference(self, images, z):
		self.generator = Generative(self.generator_name)
		self.discriminator = Discriminative(self.discriminator_name)
		
		# generative
		image_shape = images.get_shape().as_list()
		self.image_size = image_shape[1]
		self.image_channel = image_shape[3]
		self.generate_images = self.generator.inference(z, self.image_size, self.image_channel)

		# discriminative
		self.decode_images, self.image_embeddings = self.discriminator.inference(images, embedding_size = 2)
		self.decode_generate_images, self.generate_images_embeddings = self.discriminator.inference(self.generate_images, embedding_size = 2, reuse = True)

		return self.decode_images, self.image_embeddings, self.generate_images, self.decode_generate_images, self.generate_images_embeddings

	def test_generate_images(self, z, row=8, col=8):
		images = tf.cast(tf.mul(tf.add(self.generator.inference(z, self.image_size, self.image_channel, reuse = True), 1.0), 127.5), tf.uint8)
		images = [image for image in tf.split(0, _BATCH_SIZE_, images)]
		rows = []
		for i in range(row):
			rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
		image = tf.concat(1, rows)
		return tf.image.encode_png(tf.squeeze(image, [0]))
