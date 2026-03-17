import math
import os
import shutil
import subprocess
import tempfile
import zlib

import numpy as np
import tensorflow as tf
from PIL import Image
from sionna.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.mapping import Constellation, Demapper, Mapper
from sionna.utils import ebnodb2no
from models.channellayer import AWGNChannel, RayleighChannel, RicianChannel


def _require_binary(binary_name):
    binary_path = shutil.which(binary_name)
    if binary_path is None:
        raise RuntimeError(f"Required binary not found on PATH: {binary_name}")
    return binary_path


def _run_command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{stderr}")


def _make_baseline_channel(channel_name, snr_db, rician_k_factor=2.0):
    if channel_name == "AWGN":
        return AWGNChannel(snr_db)
    if channel_name == "Rayleigh":
        return RayleighChannel(snr_db)
    if channel_name == "Rician":
        return RicianChannel(snr_db, k=rician_k_factor)
    raise ValueError(f"Unsupported baseline channel type: {channel_name}")


def _symbols_to_iq(symbols):
    if symbols.dtype.is_complex:
        return tf.stack([tf.math.real(symbols), tf.math.imag(symbols)], axis=-1)
    return tf.stack([tf.cast(symbols, tf.float32), tf.zeros_like(tf.cast(symbols, tf.float32))], axis=-1)


def _iq_to_symbols(iq_tensor, original_dtype):
    if original_dtype.is_complex:
        return tf.complex(iq_tensor[..., 0], iq_tensor[..., 1])
    return iq_tensor[..., 0]


class BPGEncoder:
    def __init__(self):
        self.bpgenc = _require_binary("bpgenc")

    def _encode_at_qp(self, input_path, output_path, qp):
        if os.path.exists(output_path):
            os.remove(output_path)

        _run_command([self.bpgenc, input_path, "-q", str(qp), "-o", output_path, "-f", "444"])
        return os.path.getsize(output_path)

    def encode(self, image_array, max_bytes, header_bytes=22):
        image_u8 = np.clip(np.asarray(image_array), 0, 255).astype(np.uint8)
        target_bytes = max(1, int(max_bytes) + header_bytes)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.png")
            output_path = os.path.join(temp_dir, "output.bpg")
            Image.fromarray(image_u8).save(input_path)

            best_qp = 51
            low = 0
            high = 51

            while low <= high:
                mid = (low + high) // 2
                encoded_size = self._encode_at_qp(input_path, output_path, mid)
                if encoded_size <= target_bytes:
                    best_qp = mid
                    high = mid - 1
                else:
                    low = mid + 1

            self._encode_at_qp(input_path, output_path, best_qp)
            encoded_bytes = np.fromfile(output_path, dtype=np.uint8)

        return np.unpackbits(encoded_bytes).astype(np.float32)


class BPGDecoder:
    def __init__(self, image_shape):
        self.bpgdec = _require_binary("bpgdec")
        self.image_shape = tuple(image_shape)
        self.fallback_image = np.full(self.image_shape, 127, dtype=np.uint8)

    def decode(self, bit_array, image_shape=None, filename=None):
        del filename
        expected_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        packed_bytes = np.packbits(np.asarray(bit_array).astype(np.uint8))

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.bpg")
            output_path = os.path.join(temp_dir, "output.png")

            with open(input_path, "wb") as handle:
                handle.write(packed_bytes.tobytes())

            try:
                _run_command([self.bpgdec, input_path, "-o", output_path])
                decoded = np.array(Image.open(output_path).convert("RGB"), dtype=np.uint8)
            except RuntimeError:
                return self.fallback_image.copy()

        if decoded.shape != expected_shape:
            return self.fallback_image.copy()
        return decoded


class LDPCTransmitter:
    def __init__(self, k, n, m, esno_db, channel="AWGN", rician_k_factor=2.0):
        self.k = int(k)
        self.n = int(n)
        self.m = int(m)
        self.esno_db = float(esno_db)
        self.channel_name = channel
        self.rician_k_factor = float(rician_k_factor)
        self.num_bits_per_symbol = int(round(math.log2(self.m)))
        self.coderate = self.k / self.n

        constellation_type = "qam" if self.m != 2 else "pam"
        constellation = Constellation(
            constellation_type,
            num_bits_per_symbol=self.num_bits_per_symbol,
        )

        self.mapper = Mapper(constellation=constellation)
        self.demapper = Demapper("app", constellation=constellation)
        self.channel = _make_baseline_channel(channel, esno_db, rician_k_factor=self.rician_k_factor)
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)

    def send(self, source_bits):
        decoded_bits, _ = self.send_with_stats(source_bits)
        return decoded_bits

    def send_with_stats(self, source_bits):
        source_bits = tf.convert_to_tensor(source_bits, dtype=tf.float32)
        source_length = int(source_bits.shape[0])

        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        padded_length = math.ceil(source_length / lcm) * lcm
        source_bits_padded = tf.pad(source_bits, [[0, padded_length - source_length]])
        source_blocks = tf.reshape(source_bits_padded, (-1, self.k))

        noise_variance = ebnodb2no(
            self.esno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=self.coderate,
        )

        encoded_blocks = self.encoder(source_blocks)
        mapped_symbols = self.mapper(encoded_blocks)
        mapped_iq = _symbols_to_iq(mapped_symbols)
        received_iq = self.channel(mapped_iq, snrdB=self.esno_db)
        received_symbols = _iq_to_symbols(received_iq, mapped_symbols.dtype)
        llrs = self.demapper([received_symbols, noise_variance])
        decoded_blocks = self.decoder(llrs)
        decoded_bits = tf.reshape(decoded_blocks, (-1,))[:source_length]

        decoded_bits_np = decoded_bits.numpy()
        source_bits_np = source_bits.numpy()
        bit_error_rate = float(np.mean(np.abs(decoded_bits_np - source_bits_np)))
        stats = {
            "bit_error_rate": bit_error_rate,
            "source_bits": source_length,
            "coded_blocks": int(source_blocks.shape[0]),
        }

        return decoded_bits, stats


class RawTransmitter:
    def __init__(self, m, esno_db, channel="AWGN", rician_k_factor=2.0):
        self.m = int(m)
        self.esno_db = float(esno_db)
        self.channel_name = channel
        self.rician_k_factor = float(rician_k_factor)
        self.num_bits_per_symbol = int(round(math.log2(self.m)))

        constellation_type = "qam" if self.m != 2 else "pam"
        constellation = Constellation(
            constellation_type,
            num_bits_per_symbol=self.num_bits_per_symbol,
        )

        self.mapper = Mapper(constellation=constellation)
        self.demapper = Demapper("app", constellation=constellation)
        self.channel = _make_baseline_channel(channel, esno_db, rician_k_factor=self.rician_k_factor)

    def send_with_stats(self, source_bits):
        source_bits = tf.convert_to_tensor(source_bits, dtype=tf.float32)
        source_length = int(source_bits.shape[0])

        padded_length = math.ceil(source_length / self.num_bits_per_symbol) * self.num_bits_per_symbol
        source_bits_padded = tf.pad(source_bits, [[0, padded_length - source_length]])
        source_blocks = tf.reshape(source_bits_padded, (-1, self.num_bits_per_symbol))

        noise_variance = ebnodb2no(
            self.esno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=1.0,
        )

        mapped_symbols = self.mapper(source_blocks)
        mapped_iq = _symbols_to_iq(mapped_symbols)
        received_iq = self.channel(mapped_iq, snrdB=self.esno_db)
        received_symbols = _iq_to_symbols(received_iq, mapped_symbols.dtype)
        llrs = self.demapper([received_symbols, noise_variance])
        # Sionna LLRs are positive when bit=1 is more likely.
        decoded_blocks = tf.cast(llrs > 0, tf.float32)
        decoded_bits = tf.reshape(decoded_blocks, (-1,))[:source_length]

        decoded_bits_np = decoded_bits.numpy()
        source_bits_np = source_bits.numpy()
        bit_error_rate = float(np.mean(np.abs(decoded_bits_np - source_bits_np)))
        stats = {
            "bit_error_rate": bit_error_rate,
            "source_bits": source_length,
            "symbols": int(source_blocks.shape[0]),
        }

        return decoded_bits, stats


def append_crc32(payload_bytes):
    payload_bytes = bytes(payload_bytes)
    checksum = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    return payload_bytes + checksum.to_bytes(4, byteorder="big")


def verify_and_strip_crc32(payload_bytes):
    payload_bytes = bytes(payload_bytes)
    if len(payload_bytes) < 4:
        return None, False

    message = payload_bytes[:-4]
    received_crc = int.from_bytes(payload_bytes[-4:], byteorder="big")
    computed_crc = zlib.crc32(message) & 0xFFFFFFFF
    return message, received_crc == computed_crc
