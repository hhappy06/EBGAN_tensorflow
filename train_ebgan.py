import os, sys
import tensorflow as tf
import numpy as np
import time
from model.ebgan import EBGAN
from input.line_parser.line_parser import ImageParser
from input.data_reader import read_data
from loss.ebgan_loss import EBGANLoss
from train_op import discriminator_train_opt, generator_train_opt
from util import util

_BATCH_SIZE_ = 64
_Z_DIM_ = 2
_EPOCH_ = 10
_TRAINING_SET_SIZE_ = 60000
_DATA_DIR_ = './data/mnist/train_images'
_CSVFILE_ = ['./data/mnist/train_images/file_list']

_OUTPUT_INFO_FREQUENCE_ = 100
_OUTPUT_IMAGE_FREQUENCE_ = 100

line_parser = ImageParser()
ebgan_loss = EBGANLoss()

def train():
	with tf.Graph().as_default():
		images, image_labels = read_data(_CSVFILE_, line_parser = line_parser, data_dir = _DATA_DIR_, batch_size = _BATCH_SIZE_)
		z = tf.placeholder(tf.float32, [None, _Z_DIM_], name = 'z')

		ebgan = EBGAN()
		decode_images, image_embeddings, generate_images, decode_generate_images, generate_images_embeddings = ebgan.inference(images, z)
		discriminator_loss, generator_loss = ebgan_loss.loss(images, decode_images, generate_images, decode_generate_images, generate_images_embeddings)

		# opt
		trainable_vars = tf.trainable_variables()
		dis_vars = [var for var in trainable_vars if 'discriminator' in var.name]
		gen_vars = [var for var in trainable_vars if 'generator' in var.name]

		dis_opt = discriminator_train_opt(discriminator_loss, dis_vars)
		gen_opt = generator_train_opt(generator_loss, gen_vars)

		# generate_images for showing
		test_generate_images = ebgan.test_generate_images(z, 4, 4)
		
		# summary
		sum_z = tf.summary.histogram('z', z)
		sum_adv_loss = tf.summary.scalar('discriminator_loss', discriminator_loss)
		sum_gen_loss = tf.summary.scalar('generator_loss', generator_loss)

		sum_adv = tf.summary.merge([sum_z, sum_adv_loss])
		sum_gen = tf.summary.merge([sum_gen_loss])

		# initialize variable
		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.global_variables())

		session = tf.Session()
		file_writer = tf.summary.FileWriter('./logs', session.graph)
		session.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		print 'EBGAN training starts...'
		sys.stdout.flush()
		counter = 0
		max_steps = int(_TRAINING_SET_SIZE_ / _BATCH_SIZE_)
		for epoch in xrange(_EPOCH_):
			for step in xrange(max_steps):
				batch_z = np.random.uniform(-1, 1, [_BATCH_SIZE_, _Z_DIM_]).astype(np.float32)

				_, summary_str, error_adv_loss = session.run([dis_opt, sum_adv, discriminator_loss], feed_dict = {
					z: batch_z})
				file_writer.add_summary(summary_str, counter)

				_, summary_str, error_gen_loss = session.run([gen_opt, sum_gen, generator_loss], feed_dict = {
					z: batch_z})
				file_writer.add_summary(summary_str, counter)

				_, summary_str, error_gen_loss = session.run([gen_opt, sum_gen, generator_loss], feed_dict = {
					z: batch_z})
				file_writer.add_summary(summary_str, counter)

				file_writer.flush()

				counter += 1

				if counter % _OUTPUT_INFO_FREQUENCE_ == 0:
					print 'step: (%d, %d), adver_loss: %f, gen_loss: %f'%(epoch, step, error_adv_loss, error_gen_loss)
					sys.stdout.flush()

				if counter % _OUTPUT_IMAGE_FREQUENCE_ == 0:
					batch_z = np.random.uniform(-1, 1, [_BATCH_SIZE_, _Z_DIM_]).astype(np.float32)
					generated_image_eval = session.run(test_generate_images, {z: batch_z})
					filename = os.path.join('./result', 'out_%03d_%05d.png' %(epoch, step))
					with open(filename, 'wb') as f:
						f.write(generated_image_eval)
					print 'output generated image: %s'%(filename)
					sys.stdout.flush()

		print 'training done!'
		file_writer.close()
		coord.request_stop()
		coord.join(threads)
		session.close()

if __name__ == '__main__':
	train()
