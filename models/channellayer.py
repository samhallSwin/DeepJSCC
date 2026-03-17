import tensorflow as tf
import random


def _snr_db_to_linear(snrdB):
    snrdB = tf.cast(snrdB, tf.float32)
    return tf.pow(10.0, snrdB / 10.0)


def _prepare_snr_for_batch(snrdB, batch_size):
    snrdB = tf.cast(snrdB, tf.float32)
    if snrdB.shape.rank == 0:
        snrdB = tf.fill([batch_size], snrdB)
    else:
        snrdB = tf.reshape(snrdB, [batch_size])
    return _snr_db_to_linear(snrdB)

class RayleighChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, clip_snrdB=5):
        super().__init__()
        self.default_snrdB = float(snrdB)
        self.snr = 10 ** (snrdB / 10) # in dB
        self.clip_snr = 10 ** (clip_snrdB / 10)
    

    def call(self, x, snrdB=None):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        Assumes slow rayleigh fading, where h does not change for single batch data

        We clip the coefficient h to generate short-term SNR between +-5 dB of given long-term SNR.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"
        
        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        batch_size = tf.shape(x)[0]
        sig_power = tf.reduce_mean(i ** 2 + q ** 2, axis=1, keepdims=True)
        sig_power = tf.reshape(sig_power, [batch_size, 1, 1])
        
        # batch-wise slow fading
        h = tf.random.normal(
            (batch_size, 1, 2),
            mean=0,
            stddev=tf.math.sqrt(0.5)
        )

        snr = _prepare_snr_for_batch(self.default_snrdB if snrdB is None else snrdB, batch_size)
        snr = tf.reshape(snr, [batch_size, 1, 1])

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        yhat = h * x + n

        return yhat
    

    def get_config(self):
        config = super().get_config()
        return config

    def set_snr(self, snrdB):
        self.default_snrdB = float(snrdB)
        self.snr = 10 ** (snrdB / 10)


class AWGNChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None):
        super().__init__()
        self.default_snrdB = float(snrdB)
        self.set_snr(snrdB)
    

    def call(self, x, snrdB=None):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"

        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        batch_size = tf.shape(x)[0]
        sig_power = tf.reduce_mean(i ** 2 + q ** 2, axis=1, keepdims=True)
        sig_power = tf.reshape(sig_power, [batch_size, 1, 1])
        snr = _prepare_snr_for_batch(self.default_snrdB if snrdB is None else snrdB, batch_size)
        snr = tf.reshape(snr, [batch_size, 1, 1])

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        y = x + n
        return y
    
    def set_snr(self, snrdB):
        self.default_snrdB = float(snrdB)
        self.snr = 10 ** (snrdB / 10)  # Convert dB to linear scale

    def get_config(self):
        config = super().get_config()
        return config


class RicianChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, k=2):
        super().__init__()
        self.default_snrdB = float(snrdB)
        self.snr = 10 ** (snrdB / 10) # in dB
        self.k = k
    

    def call(self, x, snrdB=None):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        Assumes slow rayleigh fading (for NLOS part), where h does not change for single batch data

        We clip the coefficient h to generate short-term SNR between +-5 dB of given long-term SNR.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"
        
        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        batch_size = tf.shape(x)[0]
        sig_power = tf.reduce_mean(i ** 2 + q ** 2, axis=1, keepdims=True)
        sig_power = tf.reshape(sig_power, [batch_size, 1, 1])
        
        # batch-wise slow fading
        h = tf.random.normal(
            (batch_size, 1, 2),
            mean=0,
            stddev=tf.math.sqrt(0.5)
        )

        snr = _prepare_snr_for_batch(self.default_snrdB if snrdB is None else snrdB, batch_size)
        snr = tf.reshape(snr, [batch_size, 1, 1])

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        k = self.k

        yhat = tf.math.sqrt(1 / (1+k)) * h * x + tf.math.sqrt(k / (1+k)) * x + n

        return yhat

    def set_snr(self, snrdB):
        self.default_snrdB = float(snrdB)
        self.snr = 10 ** (snrdB / 10)
    
def compute_bandwidth_ratio(input_size, num_symbols):
    """
    Computes the bandwidth ratio for the autoencoder.
    :param input_size: Tuple of (H, W, C), representing height, width, and channels of input.
    :param num_symbols: Number of symbols produced by the encoder.
    :return: Bandwidth ratio.
    """
    H, W, C = input_size
    B_input = H * W * C  # Input bandwidth
    B_output = num_symbols * 2  # Output bandwidth (I and Q components)
    return B_output / B_input
