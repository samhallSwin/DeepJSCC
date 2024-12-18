import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
from models.channellayer import RayleighChannel, AWGNChannel, RicianChannel
#from models.vitblock import VitBlock

class deepJSCC(tf.keras.Model):
    def __init__(self, has_gdn=True,
                 num_symbols=512, snrdB=25, channel='AWGN', input_size=32,
                 encoder_config=None, decoder_config=None):
        """
        :param encoder_config: List of dictionaries for encoder layers configuration.
        :param decoder_config: List of dictionaries for decoder layers configuration.
        """
    
        super().__init__()

        # Default configurations
        encoder_config = encoder_config or [
            {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
            {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
            {"filters": 256, "kernel_size": 5, "stride": 2, "block_type": "C"},
            {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"}
        ]

        decoder_config = decoder_config or [
            {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
            {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
            {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
            {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None}
        ]

        # GDN layers
        if has_gdn:
            gdn_func = tfc.layers.GDN()
            igdn_func = tfc.layers.GDN(inverse=True)
        else:
            gdn_func = tf.keras.layers.Lambda(lambda x: x)
            igdn_func = tf.keras.layers.Lambda(lambda x: x)

        # Encoder
        self.encoder = Encoder(
            config=encoder_config,
            num_symbols=num_symbols,
            input_size=input_size,
            gdn_func=gdn_func
        )

        # Channel
        if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        elif channel == 'Rician':
            self.channel = RicianChannel(snrdB, k=2)
        else:
            self.channel = tf.identity

        # Decoder
        self.decoder = Decoder(
            config=decoder_config,
            input_size=input_size,
            gdn_func=igdn_func
        )

    def call(self, x):
        x = self.encoder(x)
        x = self.channel(x)
        x = self.decoder(x)
        return x

    def save_latent_representation(self, input_image, file_path):
        """
        Passes the input through the encoder and saves the latent features to a file.
        """
        latent = self.encoder(input_image)
        np.save(file_path, latent.numpy())  # Save latent features to file

    def load_and_decode(self, file_path):
        """
        Loads latent features from a file and decodes them.
        """
        latent = np.load(file_path)  # Load latent features from file
        latent_tensor = tf.convert_to_tensor(latent, dtype=tf.float32)  # Convert to tensor
        output_image = self.decoder(latent_tensor)
        return output_image

    #Slows down everything and doesn't seem useful..
    #def get_latent_features(self, x):
    #    """
    #    Passes the input through the encoder and returns latent features.
    #    """
    #    return self.encoder(x)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, num_symbols, gdn_func=None, input_size=32, **kwargs):
        super().__init__()
        self.layers = []
        current_size = input_size

        for idx, layer_cfg in enumerate(config):
            self.layers.append(build_block(
                filters=layer_cfg["filters"],
                kernel_size=layer_cfg["kernel_size"],
                stride=layer_cfg["stride"],
                block_type=layer_cfg["block_type"],
                gdn_func=gdn_func
                )
            )


        # Final layer to map to constellation
        self.layers.append(
            tf.keras.layers.Conv2D(
                filters=num_symbols // (input_size // (2 ** len(config))) ** 2 * 2,      
                kernel_size=1
            )
        )

    def call(self, x):
        for sublayer in self.layers:
            x = sublayer(x)

        b, h, w, c = x.shape
        x = tf.reshape(x, (-1, h * w * c // 2, 2))
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, gdn_func=None, input_size=32, **kwargs):
        super().__init__()
        self.layers = []
        current_size = input_size // 4  # Assuming a 2x2 downsampling factor

        for idx, layer_cfg in enumerate(config):
            if "upsample_size" in layer_cfg and layer_cfg["upsample_size"] is not None:
                self.layers.append(tf.keras.layers.Resizing(
                    layer_cfg["upsample_size"], layer_cfg["upsample_size"]
                ))
                current_size = layer_cfg["upsample_size"]

            self.layers.append(build_block(
                filters=layer_cfg["filters"],
                kernel_size=layer_cfg["kernel_size"],
                stride=layer_cfg["stride"],
                block_type=layer_cfg["block_type"],
                gdn_func=gdn_func
            ))

        # Final layer to reconstruct the image
        self.layers.append(
            tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=1,
                activation='sigmoid'
            )
        )

    def call(self, x):
        b, c, _ = x.shape
        x = tf.reshape(x, (-1, 8, 8, c * 2 // 64))

        for sublayer in self.layers:
            x = sublayer(x)
        return x


def build_block(filters, kernel_size, stride, block_type, gdn_func=None):
    """
    Dynamically build a block based on the configuration.
    """
    assert block_type in ("C", "V"), "Block type must be 'C' or 'V'."

    x = tf.keras.Sequential()
    x.add(tfc.SignalConv2D(
        filters,
        kernel_size,
        corr=True,
        strides_down=stride,
        padding="same_zeros",
        use_bias=True,
    ))

    if gdn_func:
        x.add(tfc.layers.GDN())
    else:
        x.add(tf.keras.layers.Lambda(lambda x: x))  # No GDN fallback

    x.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    return x

def simulate_channel(latent_path, output_path, channel, snr_db):
    """
    Simulates a channel on the latent data saved in a file.
    """
    latent = np.load(latent_path)
    latent_tensor = tf.convert_to_tensor(latent, dtype=tf.float32)

    # Apply the channel model
    if channel == 'AWGN':
        simulated_latent = AWGNChannel(snr_db)(latent_tensor)
    elif channel == 'Rayleigh':
        simulated_latent = RayleighChannel(snr_db)(latent_tensor)
    elif channel == 'Rician':
        simulated_latent = RicianChannel(snr_db, k=2)(latent_tensor)
    else:
        raise ValueError("Unsupported channel type.")

    # Save the simulated latent features
    np.save(output_path, simulated_latent.numpy())
