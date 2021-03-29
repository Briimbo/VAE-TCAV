import tensorflow as tf


class VAE(tf.keras.Model):
    """VAE base class"""
    def sample(self, eps=None):
        pass

    def encode(self, x):
        pass

    def decode(self, z, apply_sigmoid=False):
        pass

    # noinspection PyMethodOverriding
    def save(self, file=None, epoch=None):
        pass

    def load(self, file=None, epoch=None):
        pass


class MlpVae(VAE):
    """Multilayer-Perceptron Vae - mostly taken from https://www.tensorflow.org/tutorials/generative/cvae"""
    def __init__(self, latent_dim, save_path=None, multi_layered=False):
        super(MlpVae, self).__init__()
        self.latent_dim = latent_dim
        self.save_path = save_path
        if multi_layered:
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(units=588, activation='relu'),
                    tf.keras.layers.Dense(units=147, activation='relu'),
                    tf.keras.layers.Dense(units=36, activation='relu'),
                    tf.keras.layers.Dense(units=10, activation='relu'),

                    # tf.keras.layers.Dense(units=158, activation='relu'),

                    # No activation
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=10, activation='relu'),
                    tf.keras.layers.Dense(units=36, activation='relu'),
                    tf.keras.layers.Dense(units=147, activation='relu'),
                    tf.keras.layers.Dense(units=588, activation='relu'),
                    # tf.keras.layers.Dense(units=158, activation='relu'),
                    tf.keras.layers.Dense(units=2352, activation='relu'),
                    tf.keras.layers.Reshape(target_shape=(28, 28, 3)),
                ]
            )
        else:
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(units=158, activation='relu'),

                    # No activation
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
            )
            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=158, activation='relu'),
                    tf.keras.layers.Dense(units=2352, activation='relu'),
                    tf.keras.layers.Reshape(target_shape=(28, 28, 3)),
                ]
            )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = tf.clip_by_value(self.decoder(z), 0., 1.)
        # if apply_sigmoid:
        #     probs = tf.sigmoid(logits)
        #     return probs
        return logits

    @staticmethod
    def _get_checkpoint_paths(file, epoch):
        epoch = '' if epoch is None else '_{}'.format(epoch)
        enc_path = file + '_mlp_enc' + epoch
        dec_path = file + '_mlp_dec' + epoch
        return enc_path, dec_path

    # noinspection PyMethodOverriding
    def save(self, file=None, epoch=None):
        if file is None and self.save_path is None:
            raise Exception("file path for save needs to be specified")
        if file is None:
            file = self.save_path
        enc_path, dec_path = self._get_checkpoint_paths(file, epoch)
        self.encoder.save_weights(enc_path)
        self.decoder.save_weights(dec_path)

    def load(self, file=None, epoch=None):
        if file is None and self.save_path is None:
            raise Exception("file path for load needs to be specified")
        if file is None:
            file = self.save_path
        enc_path, dec_path = self._get_checkpoint_paths(file, epoch)
        self.encoder.load_weights(enc_path)
        self.decoder.load_weights(dec_path)

    def __call__(self, x, *args, **kwargs):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.sample(z)
