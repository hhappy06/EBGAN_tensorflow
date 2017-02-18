from __future__ import absolute_import
import tensorflow as tf
import util.util as util

_MARGIN_ = 10
_PULLAWAY_WEIGHT_ = 0.1

class EBGANLoss:
	def loss(self, images, decode_images, generate_images, decode_generate_images, generate_images_embeddings):
		real_mse_loss = util.mes_loss(images, decode_images)
		z_mse_loss = util.mes_loss(generate_images, decode_generate_images)
		z_pullaway_loss = util.pullaway_loss(generate_images_embeddings)

		discriminator_loss = real_mse_loss + _MARGIN_ - z_mse_loss
		# generator_loss = z_mse_loss + _PULLAWAY_WEIGHT_ * z_pullaway_loss
		generator_loss = z_mse_loss
		return discriminator_loss, generator_loss