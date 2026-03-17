import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
from models.channellayer import RayleighChannel, AWGNChannel, RicianChannel
#from models.vitblock import VitBlock


class FiLMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
        ])
        self.gamma_layer = None
        self.beta_layer = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma_layer = tf.keras.layers.Dense(channels)
        self.beta_layer = tf.keras.layers.Dense(channels)
        super().build(input_shape)

    def call(self, x, snr_db):
        if snr_db is None:
            return x

        snr_db = tf.cast(snr_db, tf.float32)
        if snr_db.shape.rank == 0:
            snr_db = tf.reshape(snr_db, [1, 1])
            snr_db = tf.repeat(snr_db, tf.shape(x)[0], axis=0)
        else:
            snr_db = tf.reshape(snr_db, [-1, 1])

        conditioning = self.mlp(snr_db)
        gamma = self.gamma_layer(conditioning)
        beta = self.beta_layer(conditioning)
        gamma = tf.reshape(gamma, [-1, 1, 1, tf.shape(x)[-1]])
        beta = tf.reshape(beta, [-1, 1, 1, tf.shape(x)[-1]])
        return x * (1.0 + gamma) + beta

class deepJSCC(tf.keras.Model):
    def __init__(self, has_gdn=True,
                 num_symbols=512, snrdB=25, channel='AWGN', input_size=32,
                 encoder_config=None, set_channel_filters=False, channel_filters=64, decoder_config=None, debug_file=None,
                 use_snr_side_info=False, film_hidden_units=64, rician_k_factor=2.0):
        """
        :param encoder_config: List of dictionaries for encoder layers configuration.
        :param decoder_config: List of dictionaries for decoder layers configuration.
        """
        super().__init__()

        # Debugging attributes
        self.debug = True
        self.first_call = True
        self.debug_file = debug_file  # File to write debug logs
        self.use_snr_side_info = use_snr_side_info
        self.default_snrdB = float(snrdB)
        self.rician_k_factor = float(rician_k_factor)

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

        # Encoder
        self.encoder = Encoder(
            config=encoder_config,
            num_symbols=num_symbols,
            input_size=input_size,
            set_channel_filters=set_channel_filters,
            channel_filters=channel_filters,
            gdn_func="forward" if has_gdn else "None",
            debug=self.debug,
            debug_file=debug_file,
            use_snr_side_info=use_snr_side_info,
            film_hidden_units=film_hidden_units,
        )

        # Channel
        if self.debug:
            self.log_debug(f'Building Channel Layer, Type: {channel}, snrdB: {snrdB}')
        if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        elif channel == 'Rician':
            self.channel = RicianChannel(snrdB, k=self.rician_k_factor)
        else:
            self.channel = tf.identity

        # Decoder
        self.decoder = Decoder(
            config=decoder_config,
            input_size=input_size,
            gdn_func="inverse" if has_gdn else "None",
            debug=self.debug,
            debug_file=debug_file,
            use_snr_side_info=use_snr_side_info,
            film_hidden_units=film_hidden_units,
        )

        # Disable debug after initialization
        self.debug = False

    def call(self, x):
        snr_db = None
        if isinstance(x, (tuple, list)):
            x, snr_db = x
        elif self.use_snr_side_info:
            snr_db = tf.fill([tf.shape(x)[0]], tf.cast(self.default_snrdB, tf.float32))

        # Debug shapes only during the first call
        if self.first_call:
            self.log_debug(f"Input shape = {x.shape}")
        x = self.encoder(x, snr_db=snr_db)
        if self.first_call:
            self.log_debug(f"Encoder output shape = {x.shape}")
        if callable(getattr(self.channel, "call", None)) and self.channel is not tf.identity:
            x = self.channel(x, snrdB=snr_db)
        else:
            x = self.channel(x)
        if self.first_call:
            self.log_debug(f"Channel output shape = {x.shape}")
        x = self.decoder(x, snr_db=snr_db)
        if self.first_call:
            self.log_debug(f"Decoder output shape = {x.shape}")

        # Disable first_call after the first execution
        self.first_call = False
        return x

    def log_debug(self, message):
        """Log debugging messages to the specified file."""
        if self.debug_file:
            with open(self.debug_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)  # Fallback to console if no file is provided

    def save_latent_representation(self, input_image, file_path):
        """
        Passes the input through the encoder and saves the latent features to a file.
        """
        latent = self.encoder(input_image, snr_db=None)
        np.save(file_path, latent.numpy())  # Save latent features to file

    def load_and_decode(self, file_path):
        """
        Loads latent features from a file and decodes them.
        """
        latent = np.load(file_path)  # Load latent features from file
        latent_tensor = tf.convert_to_tensor(latent, dtype=tf.float32)  # Convert to tensor
        output_image = self.decoder(latent_tensor, snr_db=None)
        return output_image

    #Slows down everything and doesn't seem useful..
    #def get_latent_features(self, x):
    #    """
    #    Passes the input through the encoder and returns latent features.
    #    """
    #    return self.encoder(x)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, num_symbols, gdn_func=None, input_size=32, set_channel_filters=False, channel_filters=64, debug=False, debug_file=None, use_snr_side_info=False, film_hidden_units=64, **kwargs):
        super().__init__()
        self.layers = []
        self.film_layers = []
        self.debug = debug
        self.first_call = True
        self.debug_file = debug_file  # File to write debug logs
        self.use_snr_side_info = use_snr_side_info
        current_size = input_size

        for idx, layer_cfg in enumerate(config):
            if self.debug:
                self.log_debug(f'Building Encoder Layer {idx + 1}, Config: {layer_cfg}, GDN: {gdn_func}')
            self.layers.append(build_block(
                filters=layer_cfg["filters"],
                kernel_size=layer_cfg["kernel_size"],
                stride=layer_cfg["stride"],
                block_type=layer_cfg["block_type"],
                section="Encoder",
                gdn_func=gdn_func
            ))
            self.film_layers.append(FiLMLayer(hidden_units=film_hidden_units) if use_snr_side_info else None)

        if self.debug: #Hacky way of getting this layer into the visualisation tool
            self.log_debug(f"Building Encoder Layer output, Config: {{'filters': {num_symbols // (input_size // (2 ** len(config))) ** 2 * 2}, 'kernel_size': 1, 'stride': 1, 'block_type': 'C'}}, GDN: None")

        #Calculate if not being set otherwise
        if not set_channel_filters:
            channel_filters = num_symbols // (input_size // (2 ** len(config))) ** 2 * 2

        # Final layer to map to constellation
        self.layers.append(
            tf.keras.layers.Conv2D(
                filters=channel_filters,
                kernel_size=1
            )
        )
        self.film_layers.append(FiLMLayer(hidden_units=film_hidden_units) if use_snr_side_info else None)

    def call(self, x, snr_db=None):
        for idx, sublayer in enumerate(self.layers):
            x = sublayer(x)
            if self.use_snr_side_info and self.film_layers[idx] is not None:
                x = self.film_layers[idx](x, snr_db)
            if self.debug and self.first_call:
                print(f"Layer {idx + 1}:")
                print(f"  Name: {sublayer.name}")
                print(f"  Type: {type(sublayer).__name__}")
                print(f"  Output Shape: {x.shape}")  # After the layer is called
                if hasattr(sublayer, 'count_params'):
                    print(f"  Trainable Parameters: {sublayer.count_params()}")
            
                self.log_debug(f"Encoder layer output shape: {x.shape}")
        b, h, w, c = x.shape
        x = tf.reshape(x, (-1, h * w * c // 2, 2))

        # Disable further debug prints
        self.first_call = False
        return x

    def log_debug(self, message):
        """Log debugging messages to the specified file."""
        if self.debug_file:
            with open(self.debug_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)  # Fallback to console if no file is provided


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, gdn_func=None, input_size=32, debug=False, debug_file=None, use_snr_side_info=False, film_hidden_units=64, **kwargs):
        super().__init__()
        self.layers = []
        self.film_layers = []
        self.debug = debug  # Store the debug flag
        self.first_call = True  # Track the first execution
        self.debug_file = debug_file
        self.use_snr_side_info = use_snr_side_info
        current_size = input_size // 4  # Assuming a 2x2 downsampling factor

        for idx, layer_cfg in enumerate(config):
            if "upsample_size" in layer_cfg and layer_cfg["upsample_size"] is not None:
                if self.debug:
                    self.log_debug(f'Building Decoder Layer {idx + 1}, Config: {layer_cfg}, GDN: {gdn_func}')
                self.layers.append(tf.keras.layers.Resizing(
                    layer_cfg["upsample_size"], layer_cfg["upsample_size"]
                ))
                self.film_layers.append(None)
                current_size = layer_cfg["upsample_size"]

            self.layers.append(build_block(
                filters=layer_cfg["filters"],
                kernel_size=layer_cfg["kernel_size"],
                stride=layer_cfg["stride"],
                block_type=layer_cfg["block_type"],
                section="Decoder",
                gdn_func=gdn_func
            ))
            self.film_layers.append(FiLMLayer(hidden_units=film_hidden_units) if use_snr_side_info else None)

        # Final layer to reconstruct the image
        if self.debug:
            self.log_debug(f"Building Decoder Layer Output, Config: {{'filters': 3, 'kernel_size': 1, 'stride': 1, 'block_type': 'C'}}, GDN: None")
        self.layers.append(
            tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=1,
                activation='sigmoid'
            )
        )
        self.film_layers.append(FiLMLayer(hidden_units=film_hidden_units) if use_snr_side_info else None)

    def call(self, x, snr_db=None):
        b, c, _ = x.shape
        x = tf.reshape(x, (-1, 8, 8, c * 2 // 64))

        for idx, sublayer in enumerate(self.layers):
            x = sublayer(x)
            if self.use_snr_side_info and self.film_layers[idx] is not None:
                x = self.film_layers[idx](x, snr_db)
            if self.debug and self.first_call:
                print(f"Layer {idx + 1}:")
                print(f"  Name: {sublayer.name}")
                print(f"  Type: {type(sublayer).__name__}")
                print(f"  Output Shape: {x.shape}")  # After the layer is called
                if hasattr(sublayer, 'count_params'):
                    print(f"  Trainable Parameters: {sublayer.count_params()}")
            
                self.log_debug(f"Decoder layer output shape: {x.shape}")
        
        # Disable further debug prints
        self.first_call = False
        return x

    def log_debug(self, message):
        """Log debugging messages to the specified file."""
        if self.debug_file:
            with open(self.debug_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)  # Fallback to console if no file is provided

def build_block(filters, kernel_size, stride, block_type, gdn_func=None, section="None"):
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

    if gdn_func == "forward":
        x.add(tfc.layers.GDN())
    elif gdn_func == "inverse":
        x.add(tfc.layers.GDN(inverse=True))
    elif gdn_func == "none":
        x.add(tf.keras.layers.Lambda(lambda x: x))  # No GDN fallback
    else:
        print("GDN function parsing issue")

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
