import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from utils import analysis_tools
from utils.datasets import dataset_generator


def _to_uint8_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _save_image(image, filepath):
    Image.fromarray(_to_uint8_image(image)).save(filepath)


def _compute_psnr_ssim(reference_u8, candidate_u8):
    psnr_value = peak_signal_noise_ratio(reference_u8, candidate_u8, data_range=255)
    ssim_value = structural_similarity(
        reference_u8,
        candidate_u8,
        channel_axis=-1,
        data_range=255,
    )
    return float(psnr_value), float(ssim_value)


def _extract_images(batch_inputs):
    if isinstance(batch_inputs, (tuple, list)):
        return batch_inputs[0]
    return batch_inputs


def _model_predict(model, images, snr_db=None):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    if getattr(model, "use_snr_side_info", False):
        if snr_db is None:
            snr_db = getattr(model, "default_snrdB", 0.0)
        snr_tensor = tf.fill((tf.shape(images)[0],), tf.cast(snr_db, tf.float32))
        return model((images, snr_tensor), training=False).numpy()
    return model(images, training=False).numpy()


def _get_clip_handle(config):
    if not getattr(config, "enable_clip_metric", False):
        return None

    from utils.clip_metrics import CLIPHandle

    return CLIPHandle(
        model_name=getattr(config, "clip_model_name", "ViT-B/32"),
        device=getattr(config, "clip_device", "cpu"),
    )


def _compute_clip_similarity(reference_f01, candidate_f01, clip_handle):
    if clip_handle is None:
        return None

    from utils.clip_metrics import clip_cosine_similarities

    similarities = clip_cosine_similarities(
        [np.asarray(reference_f01, dtype=np.float32)],
        [np.asarray(candidate_f01, dtype=np.float32)],
        clip_handle,
        batch_size=1,
    )
    return float(similarities[0])


def _collect_eval_images(test_ds, num_images):
    collected = []

    for images, _ in test_ds:
        originals = _extract_images(images).numpy()
        for image in originals:
            collected.append(image)
            if len(collected) >= num_images:
                return collected

    return collected


def _collect_eval_samples_with_labels(config, num_images):
    labeled_ds = dataset_generator(config.test_dir, config, shuffle=False)

    images_out = []
    labels_out = []

    for images, labels in labeled_ds:
        images_np = images.numpy().astype(np.float32) / 255.0
        labels_np = labels.numpy()
        for image, label in zip(images_np, labels_np):
            images_out.append(image)
            labels_out.append(int(label))
            if len(images_out) >= num_images:
                return images_out, labels_out

    return images_out, labels_out


def _dataset_class_names(config):
    return sorted(
        entry
        for entry in os.listdir(config.test_dir)
        if os.path.isdir(os.path.join(config.test_dir, entry))
    )


def _get_downstream_classifier(config):
    if not getattr(config, "enable_downstream_metric", False):
        return None

    import timm
    import torch
    from timm.data import create_transform, resolve_model_data_config

    class DownstreamClassifier:
        def __init__(self, model_id, device):
            self.device = device
            self.model = timm.create_model(f"hf-hub:{model_id}", pretrained=True).to(device)
            self.model.eval()
            self.data_config = resolve_model_data_config(self.model)
            self.transform = create_transform(**self.data_config, is_training=False)
            self.label_names = list(self.model.pretrained_cfg.get("label_names", []))

        def predict(self, image_f01):
            with torch.no_grad():
                pil_image = Image.fromarray((np.clip(image_f01, 0.0, 1.0) * 255.0).astype(np.uint8))
                tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=-1)
                pred = int(torch.argmax(probs, dim=-1).item())
                conf = float(torch.max(probs, dim=-1).values.item())
                return pred, conf

    return DownstreamClassifier(
        model_id=getattr(config, "downstream_model_id", "cm93/resnet18-eurosat"),
        device=getattr(config, "downstream_device", "cpu"),
    )


def _build_snr_values(snr_range, snr_step):
    snr_min, snr_max = snr_range
    snr_min = float(snr_min)
    snr_max = float(snr_max)
    snr_step = float(snr_step)

    if snr_step <= 0:
        raise ValueError(f"snr_eval_step must be positive, got {snr_step}")

    snr_values = []
    current = snr_min
    epsilon = snr_step * 1e-6

    while current <= snr_max + epsilon:
        snr_values.append(round(current, 6))
        current += snr_step

    return snr_values


def _write_snr_summary_plots(summary_rows, output_dir):
    snr_values = [row["snr_db"] for row in summary_rows]
    model_psnr = [row["model_psnr_mean"] for row in summary_rows]
    baseline_psnr = [row["bpg_ldpc_psnr_mean"] for row in summary_rows]
    raw_bpg_psnr = [row["bpg_raw_psnr_mean"] for row in summary_rows]
    model_ssim = [row["model_ssim_mean"] for row in summary_rows]
    baseline_ssim = [row["bpg_ldpc_ssim_mean"] for row in summary_rows]
    raw_bpg_ssim = [row["bpg_raw_ssim_mean"] for row in summary_rows]
    baseline_ber = [row["bpg_ldpc_ber_mean"] for row in summary_rows]
    raw_bpg_ber = [row["bpg_raw_ber_mean"] for row in summary_rows]
    baseline_crc_ok = [row["bpg_ldpc_crc_ok_mean"] for row in summary_rows]
    has_clip = "model_clip_mean" in summary_rows[0]
    has_downstream = "model_acc_mean" in summary_rows[0]
    if has_clip:
        model_clip = [row["model_clip_mean"] for row in summary_rows]
        baseline_clip = [row["bpg_ldpc_clip_mean"] for row in summary_rows]
        raw_bpg_clip = [row["bpg_raw_clip_mean"] for row in summary_rows]
    if has_downstream:
        original_acc = [row["original_acc_mean"] for row in summary_rows]
        model_acc = [row["model_acc_mean"] for row in summary_rows]
        baseline_acc = [row["bpg_ldpc_acc_mean"] for row in summary_rows]
        raw_bpg_acc = [row["bpg_raw_acc_mean"] for row in summary_rows]

    psnr_plot_path = os.path.join(output_dir, "mean_psnr_vs_snr.png")
    ssim_plot_path = os.path.join(output_dir, "mean_ssim_vs_snr.png")
    ber_plot_path = os.path.join(output_dir, "mean_bpg_ldpc_ber_vs_snr.png")
    crc_plot_path = os.path.join(output_dir, "mean_bpg_ldpc_crc_ok_vs_snr.png")
    clip_plot_path = os.path.join(output_dir, "mean_clip_vs_snr.png") if has_clip else None
    acc_plot_path = os.path.join(output_dir, "mean_downstream_acc_vs_snr.png") if has_downstream else None

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, model_psnr, marker="o", label="Model")
    plt.plot(snr_values, baseline_psnr, marker="o", label="BPG+LDPC")
    plt.plot(snr_values, raw_bpg_psnr, marker="o", label="BPG raw")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean PSNR")
    plt.title("Mean PSNR vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(psnr_plot_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, model_ssim, marker="o", label="Model")
    plt.plot(snr_values, baseline_ssim, marker="o", label="BPG+LDPC")
    plt.plot(snr_values, raw_bpg_ssim, marker="o", label="BPG raw")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean SSIM")
    plt.title("Mean SSIM vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ssim_plot_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, baseline_ber, marker="o", label="BPG+LDPC BER")
    plt.plot(snr_values, raw_bpg_ber, marker="o", label="BPG raw BER")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean BER")
    plt.title("Mean BPG+LDPC BER vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ber_plot_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, baseline_crc_ok, marker="o", label="BPG+LDPC CRC success")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean CRC success rate")
    plt.title("Mean BPG+LDPC CRC Success vs SNR")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(crc_plot_path, bbox_inches="tight")
    plt.close()

    if has_clip:
        plt.figure(figsize=(8, 5))
        plt.plot(snr_values, model_clip, marker="o", label="Model")
        plt.plot(snr_values, baseline_clip, marker="o", label="BPG+LDPC")
        plt.plot(snr_values, raw_bpg_clip, marker="o", label="BPG raw")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean CLIP cosine")
        plt.title("Mean CLIP Similarity vs SNR")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(clip_plot_path, bbox_inches="tight")
        plt.close()

    if has_downstream:
        plt.figure(figsize=(8, 5))
        plt.plot(snr_values, original_acc, marker="o", label="Original")
        plt.plot(snr_values, model_acc, marker="o", label="Model")
        plt.plot(snr_values, baseline_acc, marker="o", label="BPG+LDPC")
        plt.plot(snr_values, raw_bpg_acc, marker="o", label="BPG raw")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean downstream accuracy")
        plt.title("Mean Downstream Accuracy vs SNR")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_plot_path, bbox_inches="tight")
        plt.close()

    return psnr_plot_path, ssim_plot_path, ber_plot_path, crc_plot_path, clip_plot_path, acc_plot_path


def save_reconstructions(model, test_ds, config):
    num_images = getattr(config, "num_visual_eval_images", 8)
    output_dir = getattr(config, "visual_eval_output_dir", "outputs/visual_eval")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    saved = 0
    clip_handle = _get_clip_handle(config)

    for images, _ in test_ds:
        predictions = model(images, training=False).numpy()
        originals = _extract_images(images).numpy()

        for idx in range(images.shape[0]):
            if saved >= num_images:
                break

            original_u8 = _to_uint8_image(originals[idx])
            reconstructed_u8 = _to_uint8_image(predictions[idx])

            psnr_value, ssim_value = _compute_psnr_ssim(original_u8, reconstructed_u8)
            clip_value = _compute_clip_similarity(originals[idx], predictions[idx], clip_handle)

            image_id = saved + 1
            original_path = os.path.join(output_dir, f"image_{image_id:03d}_original.png")
            reconstructed_path = os.path.join(output_dir, f"image_{image_id:03d}_reconstructed.png")

            _save_image(original_u8, original_path)
            _save_image(reconstructed_u8, reconstructed_path)

            rows.append({
                "image_id": image_id,
                "original_path": original_path,
                "reconstructed_path": reconstructed_path,
                "psnr": float(psnr_value),
                "ssim": float(ssim_value),
            })
            if clip_value is not None:
                rows[-1]["clip"] = clip_value
            saved += 1

        if saved >= num_images:
            break

    if not rows:
        raise RuntimeError("No images were available from the test dataset.")

    fieldnames = ["image_id", "original_path", "reconstructed_path", "psnr", "ssim"]
    if clip_handle is not None:
        fieldnames.append("clip")

    metrics_path = os.path.join(output_dir, "metrics.tsv")
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
        summary_row = {
            "image_id": "mean",
            "original_path": "",
            "reconstructed_path": "",
            "psnr": float(np.mean([row["psnr"] for row in rows])),
            "ssim": float(np.mean([row["ssim"] for row in rows])),
        }
        if clip_handle is not None:
            summary_row["clip"] = float(np.mean([row["clip"] for row in rows]))
        writer.writerow(summary_row)

    print(f"Saved {len(rows)} original/reconstruction pairs to {output_dir}")
    print(f"Saved PSNR/SSIM metrics to {metrics_path}")

    return {
        "output_dir": output_dir,
        "metrics_path": metrics_path,
    }


def _channel_uses_for_model(model, sample_image):
    latent = model.encoder(tf.expand_dims(sample_image, axis=0))
    return int(latent.shape[1])


def _max_bpg_payload_bytes(channel_uses, mcs):
    k, n, m = mcs
    bits_per_symbol = int(np.log2(m))
    source_bits = channel_uses * bits_per_symbol * (k / n)
    transport_bytes = max(1, int(source_bits // 8))
    crc_bytes = 4
    return max(1, transport_bytes - crc_bytes)


def _max_raw_bpg_payload_bytes(channel_uses, m):
    bits_per_symbol = int(np.log2(m))
    payload_bits = channel_uses * bits_per_symbol
    return max(1, int(payload_bits // 8))


def _select_adaptive_mcs(config, snr_db):
    adaptive_enabled = getattr(config, "adaptive_bpg_ldpc", False)
    if not adaptive_enabled:
        return config.mcs

    mcs_table = getattr(config, "adaptive_mcs_table", None)
    if not mcs_table:
        return config.mcs

    selected_mcs = tuple(mcs_table[0][1])
    for snr_threshold, mcs in sorted(mcs_table, key=lambda item: float(item[0])):
        if float(snr_db) >= float(snr_threshold):
            selected_mcs = tuple(mcs)
        else:
            break

    return selected_mcs


def _run_bpg_ldpc_baseline(image_u8, channel_uses, snr_db, mcs, channel_type, rician_k_factor=2.0):
    max_bytes = _max_bpg_payload_bytes(channel_uses, mcs)
    k, n, m = mcs

    encoder = analysis_tools.BPGEncoder()
    decoder = analysis_tools.BPGDecoder(image_shape=image_u8.shape)
    transmitter = analysis_tools.LDPCTransmitter(
        k, n, m, snr_db, channel_type, rician_k_factor=rician_k_factor
    )

    payload_bits = encoder.encode(image_u8, max_bytes)
    payload_bytes = np.packbits(payload_bits.astype(np.uint8)).tobytes()
    framed_bytes = analysis_tools.append_crc32(payload_bytes)
    framed_bits = np.unpackbits(np.frombuffer(framed_bytes, dtype=np.uint8)).astype(np.float32)

    received_bits, tx_stats = transmitter.send_with_stats(framed_bits)
    received_bytes = np.packbits(np.asarray(received_bits.numpy()).astype(np.uint8)).tobytes()
    decoded_payload_bytes, crc_ok = analysis_tools.verify_and_strip_crc32(received_bytes)

    if crc_ok:
        decoded_payload_bits = np.unpackbits(
            np.frombuffer(decoded_payload_bytes, dtype=np.uint8)
        ).astype(np.float32)
    else:
        decoded_payload_bits = np.zeros_like(payload_bits)

    decoded_image = decoder.decode(decoded_payload_bits, image_u8.shape, "bpg_ldpc_processed")
    decoded_image = np.clip(np.asarray(decoded_image), 0, 255).astype(np.uint8)

    return decoded_image, max_bytes, tx_stats["bit_error_rate"], crc_ok, mcs


def _run_bpg_raw_baseline(image_u8, channel_uses, snr_db, mcs, channel_type, rician_k_factor=2.0):
    _, _, m = mcs
    max_bytes = _max_raw_bpg_payload_bytes(channel_uses, m)

    encoder = analysis_tools.BPGEncoder()
    decoder = analysis_tools.BPGDecoder(image_shape=image_u8.shape)
    transmitter = analysis_tools.RawTransmitter(
        m, snr_db, channel_type, rician_k_factor=rician_k_factor
    )

    payload_bits = encoder.encode(image_u8, max_bytes)
    received_bits, tx_stats = transmitter.send_with_stats(payload_bits)
    decoded_image = decoder.decode(received_bits.numpy(), image_u8.shape, "bpg_raw_processed")
    decoded_image = np.clip(np.asarray(decoded_image), 0, 255).astype(np.uint8)

    return decoded_image, max_bytes, tx_stats["bit_error_rate"], m


def compare_to_BPG_LDPC(model, test_ds, train_ds, config):
    del train_ds

    num_images = getattr(config, "num_visual_eval_images", 8)
    output_dir = getattr(config, "bpg_ldpc_eval_output_dir", "outputs/bpg_ldpc_eval")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    saved = 0
    channel_uses = None
    clip_handle = _get_clip_handle(config)
    downstream_classifier = _get_downstream_classifier(config)
    labeled_images = None
    labeled_targets = None

    if downstream_classifier is not None:
        labeled_images, labeled_targets = _collect_eval_samples_with_labels(config, num_images)
        if not labeled_images:
            raise RuntimeError("No labeled evaluation images were available for downstream metrics.")
        dataset_class_names = _dataset_class_names(config)
        label_mapping = {idx: downstream_classifier.label_names.index(name) for idx, name in enumerate(dataset_class_names)}
        predictions = _model_predict(model, np.asarray(labeled_images, dtype=np.float32), snr_db=config.train_snrdB)
        originals = np.asarray(labeled_images, dtype=np.float32)
        channel_uses = _channel_uses_for_model(model, originals[0])

    for images, _ in test_ds if downstream_classifier is None else []:
        predictions = _model_predict(model, _extract_images(images), snr_db=config.train_snrdB)
        originals = _extract_images(images).numpy()

        if channel_uses is None:
            channel_uses = _channel_uses_for_model(model, originals[0])

        for idx in range(images.shape[0]):
            if saved >= num_images:
                break

            original_u8 = _to_uint8_image(originals[idx])
            reconstructed_u8 = _to_uint8_image(predictions[idx])
            selected_mcs = _select_adaptive_mcs(config, config.train_snrdB)
            baseline_u8, max_bytes, bit_error_rate, crc_ok, _ = _run_bpg_ldpc_baseline(
                original_u8,
                channel_uses=channel_uses,
                snr_db=config.train_snrdB,
                mcs=selected_mcs,
                channel_type=config.channel_type,
                rician_k_factor=getattr(config, "rician_k_factor", 2.0),
            )
            raw_bpg_u8, raw_bpg_max_bytes, raw_bpg_ber, raw_bpg_m = _run_bpg_raw_baseline(
                original_u8,
                channel_uses=channel_uses,
                snr_db=config.train_snrdB,
                mcs=selected_mcs,
                channel_type=config.channel_type,
                rician_k_factor=getattr(config, "rician_k_factor", 2.0),
            )

            image_id = saved + 1
            original_path = os.path.join(output_dir, f"image_{image_id:03d}_original.png")
            model_path = os.path.join(output_dir, f"image_{image_id:03d}_model.png")
            baseline_path = os.path.join(output_dir, f"image_{image_id:03d}_bpg_ldpc.png")
            raw_bpg_path = os.path.join(output_dir, f"image_{image_id:03d}_bpg_raw.png")

            _save_image(original_u8, original_path)
            _save_image(reconstructed_u8, model_path)
            _save_image(baseline_u8, baseline_path)
            _save_image(raw_bpg_u8, raw_bpg_path)

            model_psnr, model_ssim = _compute_psnr_ssim(original_u8, reconstructed_u8)
            baseline_psnr, baseline_ssim = _compute_psnr_ssim(original_u8, baseline_u8)
            raw_bpg_psnr, raw_bpg_ssim = _compute_psnr_ssim(original_u8, raw_bpg_u8)
            model_clip = _compute_clip_similarity(originals[idx], predictions[idx], clip_handle)
            bpg_ldpc_clip = _compute_clip_similarity(originals[idx], baseline_u8.astype(np.float32) / 255.0, clip_handle)
            bpg_raw_clip = _compute_clip_similarity(originals[idx], raw_bpg_u8.astype(np.float32) / 255.0, clip_handle)

            mcs_k, mcs_n, mcs_m = selected_mcs
            rows.append({
                "image_id": image_id,
                "channel_uses": channel_uses,
                "bpg_max_bytes": max_bytes,
                "bpg_ldpc_ber": bit_error_rate,
                "bpg_ldpc_crc_ok": int(crc_ok),
                "bpg_raw_max_bytes": raw_bpg_max_bytes,
                "bpg_raw_ber": raw_bpg_ber,
                "mcs_k": mcs_k,
                "mcs_n": mcs_n,
                "mcs_m": mcs_m,
                "snr_db": config.train_snrdB,
                "original_path": original_path,
                "model_path": model_path,
                "bpg_ldpc_path": baseline_path,
                "bpg_raw_path": raw_bpg_path,
                "model_psnr": float(model_psnr),
                "model_ssim": float(model_ssim),
                "bpg_ldpc_psnr": float(baseline_psnr),
                "bpg_ldpc_ssim": float(baseline_ssim),
                "bpg_raw_psnr": float(raw_bpg_psnr),
                "bpg_raw_ssim": float(raw_bpg_ssim),
            })
            if clip_handle is not None:
                rows[-1]["model_clip"] = model_clip
                rows[-1]["bpg_ldpc_clip"] = bpg_ldpc_clip
                rows[-1]["bpg_raw_clip"] = bpg_raw_clip
            saved += 1

        if saved >= num_images:
            break

    if not rows:
        if downstream_classifier is not None:
            for idx in range(len(originals)):
                if saved >= num_images:
                    break

                original_u8 = _to_uint8_image(originals[idx])
                reconstructed_u8 = _to_uint8_image(predictions[idx])
                selected_mcs = _select_adaptive_mcs(config, config.train_snrdB)
                baseline_u8, max_bytes, bit_error_rate, crc_ok, _ = _run_bpg_ldpc_baseline(
                    original_u8,
                    channel_uses=channel_uses,
                    snr_db=config.train_snrdB,
                    mcs=selected_mcs,
                    channel_type=config.channel_type,
                    rician_k_factor=getattr(config, "rician_k_factor", 2.0),
                )
                raw_bpg_u8, raw_bpg_max_bytes, raw_bpg_ber, _ = _run_bpg_raw_baseline(
                    original_u8,
                    channel_uses=channel_uses,
                    snr_db=config.train_snrdB,
                    mcs=selected_mcs,
                    channel_type=config.channel_type,
                    rician_k_factor=getattr(config, "rician_k_factor", 2.0),
                )

                image_id = saved + 1
                original_path = os.path.join(output_dir, f"image_{image_id:03d}_original.png")
                model_path = os.path.join(output_dir, f"image_{image_id:03d}_model.png")
                baseline_path = os.path.join(output_dir, f"image_{image_id:03d}_bpg_ldpc.png")
                raw_bpg_path = os.path.join(output_dir, f"image_{image_id:03d}_bpg_raw.png")

                _save_image(original_u8, original_path)
                _save_image(reconstructed_u8, model_path)
                _save_image(baseline_u8, baseline_path)
                _save_image(raw_bpg_u8, raw_bpg_path)

                model_psnr, model_ssim = _compute_psnr_ssim(original_u8, reconstructed_u8)
                baseline_psnr, baseline_ssim = _compute_psnr_ssim(original_u8, baseline_u8)
                raw_bpg_psnr, raw_bpg_ssim = _compute_psnr_ssim(original_u8, raw_bpg_u8)
                model_clip = _compute_clip_similarity(originals[idx], predictions[idx], clip_handle)
                bpg_ldpc_clip = _compute_clip_similarity(originals[idx], baseline_u8.astype(np.float32) / 255.0, clip_handle)
                bpg_raw_clip = _compute_clip_similarity(originals[idx], raw_bpg_u8.astype(np.float32) / 255.0, clip_handle)
                model_pred, _ = downstream_classifier.predict(predictions[idx])
                bpg_ldpc_pred, _ = downstream_classifier.predict(baseline_u8.astype(np.float32) / 255.0)
                bpg_raw_pred, _ = downstream_classifier.predict(raw_bpg_u8.astype(np.float32) / 255.0)
                original_pred, _ = downstream_classifier.predict(originals[idx])
                target_model_label = label_mapping[labeled_targets[idx]]

                mcs_k, mcs_n, mcs_m = selected_mcs
                rows.append({
                    "image_id": image_id,
                    "channel_uses": channel_uses,
                    "bpg_max_bytes": max_bytes,
                    "bpg_ldpc_ber": bit_error_rate,
                    "bpg_ldpc_crc_ok": int(crc_ok),
                    "bpg_raw_max_bytes": raw_bpg_max_bytes,
                    "bpg_raw_ber": raw_bpg_ber,
                    "mcs_k": mcs_k,
                    "mcs_n": mcs_n,
                    "mcs_m": mcs_m,
                    "snr_db": config.train_snrdB,
                    "original_path": original_path,
                    "model_path": model_path,
                    "bpg_ldpc_path": baseline_path,
                    "bpg_raw_path": raw_bpg_path,
                    "model_psnr": float(model_psnr),
                    "model_ssim": float(model_ssim),
                    "bpg_ldpc_psnr": float(baseline_psnr),
                    "bpg_ldpc_ssim": float(baseline_ssim),
                    "bpg_raw_psnr": float(raw_bpg_psnr),
                    "bpg_raw_ssim": float(raw_bpg_ssim),
                    "target_label": labeled_targets[idx],
                    "target_model_label": target_model_label,
                    "original_pred": original_pred,
                    "model_pred": model_pred,
                    "bpg_ldpc_pred": bpg_ldpc_pred,
                    "bpg_raw_pred": bpg_raw_pred,
                    "model_acc": float(model_pred == target_model_label),
                    "bpg_ldpc_acc": float(bpg_ldpc_pred == target_model_label),
                    "bpg_raw_acc": float(bpg_raw_pred == target_model_label),
                    "original_acc": float(original_pred == target_model_label),
                })
                if clip_handle is not None:
                    rows[-1]["model_clip"] = model_clip
                    rows[-1]["bpg_ldpc_clip"] = bpg_ldpc_clip
                    rows[-1]["bpg_raw_clip"] = bpg_raw_clip
                saved += 1
        else:
            raise RuntimeError("No images were available from the test dataset.")

    metrics_path = os.path.join(output_dir, "metrics.tsv")
    with open(metrics_path, "w", newline="") as handle:
        fieldnames = [
            "image_id",
            "channel_uses",
            "bpg_max_bytes",
            "bpg_ldpc_ber",
            "bpg_ldpc_crc_ok",
            "bpg_raw_max_bytes",
            "bpg_raw_ber",
            "mcs_k",
            "mcs_n",
            "mcs_m",
            "snr_db",
            "original_path",
            "model_path",
            "bpg_ldpc_path",
            "bpg_raw_path",
            "model_psnr",
            "model_ssim",
            "bpg_ldpc_psnr",
            "bpg_ldpc_ssim",
            "bpg_raw_psnr",
            "bpg_raw_ssim",
        ]
        if clip_handle is not None:
            fieldnames.extend(["model_clip", "bpg_ldpc_clip", "bpg_raw_clip"])
        if downstream_classifier is not None:
            fieldnames.extend([
                "target_label",
                "original_pred",
                "model_pred",
                "bpg_ldpc_pred",
                "bpg_raw_pred",
                "original_acc",
                "model_acc",
                "bpg_ldpc_acc",
                "bpg_raw_acc",
            ])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        summary_row = {
            "image_id": "mean",
            "channel_uses": channel_uses,
            "bpg_max_bytes": float(np.mean([row["bpg_max_bytes"] for row in rows])),
            "bpg_ldpc_ber": float(np.mean([row["bpg_ldpc_ber"] for row in rows])),
            "bpg_ldpc_crc_ok": float(np.mean([row["bpg_ldpc_crc_ok"] for row in rows])),
            "bpg_raw_max_bytes": float(np.mean([row["bpg_raw_max_bytes"] for row in rows])),
            "bpg_raw_ber": float(np.mean([row["bpg_raw_ber"] for row in rows])),
            "mcs_k": rows[0]["mcs_k"],
            "mcs_n": rows[0]["mcs_n"],
            "mcs_m": rows[0]["mcs_m"],
            "snr_db": config.train_snrdB,
            "original_path": "",
            "model_path": "",
            "bpg_ldpc_path": "",
            "bpg_raw_path": "",
            "model_psnr": float(np.mean([row["model_psnr"] for row in rows])),
            "model_ssim": float(np.mean([row["model_ssim"] for row in rows])),
            "bpg_ldpc_psnr": float(np.mean([row["bpg_ldpc_psnr"] for row in rows])),
            "bpg_ldpc_ssim": float(np.mean([row["bpg_ldpc_ssim"] for row in rows])),
            "bpg_raw_psnr": float(np.mean([row["bpg_raw_psnr"] for row in rows])),
            "bpg_raw_ssim": float(np.mean([row["bpg_raw_ssim"] for row in rows])),
        }
        if clip_handle is not None:
            summary_row["model_clip"] = float(np.mean([row["model_clip"] for row in rows]))
            summary_row["bpg_ldpc_clip"] = float(np.mean([row["bpg_ldpc_clip"] for row in rows]))
            summary_row["bpg_raw_clip"] = float(np.mean([row["bpg_raw_clip"] for row in rows]))
        if downstream_classifier is not None:
            summary_row["target_label"] = ""
            summary_row["original_pred"] = ""
            summary_row["model_pred"] = ""
            summary_row["bpg_ldpc_pred"] = ""
            summary_row["bpg_raw_pred"] = ""
            summary_row["original_acc"] = float(np.mean([row["original_acc"] for row in rows]))
            summary_row["model_acc"] = float(np.mean([row["model_acc"] for row in rows]))
            summary_row["bpg_ldpc_acc"] = float(np.mean([row["bpg_ldpc_acc"] for row in rows]))
            summary_row["bpg_raw_acc"] = float(np.mean([row["bpg_raw_acc"] for row in rows]))
        writer.writerow(summary_row)

    print(f"Saved {len(rows)} model/BPG+LDPC comparison triplets to {output_dir}")
    print(f"Model channel uses per image: {channel_uses}")
    print(f"Saved comparison metrics to {metrics_path}")

    return {
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "channel_uses": channel_uses,
    }


def compare_to_BPG_LDPC_sweep(model, test_ds, config):
    num_images = getattr(config, "num_snr_eval_images", 8)
    output_dir = getattr(config, "snr_sweep_output_dir", "outputs/snr_sweep")
    os.makedirs(output_dir, exist_ok=True)

    downstream_classifier = _get_downstream_classifier(config)
    if downstream_classifier is not None:
        eval_images, eval_labels = _collect_eval_samples_with_labels(config, num_images)
        dataset_class_names = _dataset_class_names(config)
        label_mapping = {idx: downstream_classifier.label_names.index(name) for idx, name in enumerate(dataset_class_names)}
    else:
        eval_images = _collect_eval_images(test_ds, num_images)
        eval_labels = None
    if not eval_images:
        raise RuntimeError("No images were available from the test dataset.")

    channel_uses = _channel_uses_for_model(model, eval_images[0])
    snr_step = getattr(config, "snr_eval_step", 1)
    snr_values = _build_snr_values(config.snr_range, snr_step)
    clip_handle = _get_clip_handle(config)

    print(
        "Running compare_to_BPG_LDPC_sweep "
        f"from {snr_values[0]} to {snr_values[-1]} dB "
        f"in steps of {snr_step} dB "
        f"({len(snr_values)} SNR values total)."
    )

    per_image_rows = []
    summary_rows = []

    for snr_db in snr_values:
        print(f"Processing at {snr_db} SNR")
        selected_mcs = _select_adaptive_mcs(config, snr_db)
        mcs_k, mcs_n, mcs_m = selected_mcs
        model_psnr_values = []
        model_ssim_values = []
        baseline_psnr_values = []
        baseline_ssim_values = []
        raw_bpg_psnr_values = []
        raw_bpg_ssim_values = []
        model_clip_values = []
        bpg_ldpc_clip_values = []
        bpg_raw_clip_values = []
        original_acc_values = []
        model_acc_values = []
        bpg_ldpc_acc_values = []
        bpg_raw_acc_values = []

        snr_dir = os.path.join(output_dir, f"snr_{snr_db}")
        os.makedirs(snr_dir, exist_ok=True)

        for idx, image in enumerate(eval_images, start=1):
            image_tensor = tf.expand_dims(image, axis=0)
            model.channel.set_snr(snr_db)
            reconstructed = _model_predict(model, image_tensor, snr_db=snr_db)[0]

            original_u8 = _to_uint8_image(image)
            reconstructed_u8 = _to_uint8_image(reconstructed)
            baseline_u8, max_bytes, bit_error_rate, crc_ok, _ = _run_bpg_ldpc_baseline(
                original_u8,
                channel_uses=channel_uses,
                snr_db=snr_db,
                mcs=selected_mcs,
                channel_type=config.channel_type,
                rician_k_factor=getattr(config, "rician_k_factor", 2.0),
            )
            raw_bpg_u8, raw_bpg_max_bytes, raw_bpg_ber, _ = _run_bpg_raw_baseline(
                original_u8,
                channel_uses=channel_uses,
                snr_db=snr_db,
                mcs=selected_mcs,
                channel_type=config.channel_type,
                rician_k_factor=getattr(config, "rician_k_factor", 2.0),
            )

            model_psnr, model_ssim = _compute_psnr_ssim(original_u8, reconstructed_u8)
            baseline_psnr, baseline_ssim = _compute_psnr_ssim(original_u8, baseline_u8)
            raw_bpg_psnr, raw_bpg_ssim = _compute_psnr_ssim(original_u8, raw_bpg_u8)
            model_clip = _compute_clip_similarity(image, reconstructed, clip_handle)
            bpg_ldpc_clip = _compute_clip_similarity(image, baseline_u8.astype(np.float32) / 255.0, clip_handle)
            bpg_raw_clip = _compute_clip_similarity(image, raw_bpg_u8.astype(np.float32) / 255.0, clip_handle)

            model_psnr_values.append(model_psnr)
            model_ssim_values.append(model_ssim)
            baseline_psnr_values.append(baseline_psnr)
            baseline_ssim_values.append(baseline_ssim)
            raw_bpg_psnr_values.append(raw_bpg_psnr)
            raw_bpg_ssim_values.append(raw_bpg_ssim)
            if clip_handle is not None:
                model_clip_values.append(model_clip)
                bpg_ldpc_clip_values.append(bpg_ldpc_clip)
                bpg_raw_clip_values.append(bpg_raw_clip)
            if downstream_classifier is not None:
                target_label = eval_labels[idx - 1]
                target_model_label = label_mapping[target_label]
                original_pred, _ = downstream_classifier.predict(image)
                model_pred, _ = downstream_classifier.predict(reconstructed)
                bpg_ldpc_pred, _ = downstream_classifier.predict(baseline_u8.astype(np.float32) / 255.0)
                bpg_raw_pred, _ = downstream_classifier.predict(raw_bpg_u8.astype(np.float32) / 255.0)
                original_acc = float(original_pred == target_model_label)
                model_acc = float(model_pred == target_model_label)
                bpg_ldpc_acc = float(bpg_ldpc_pred == target_model_label)
                bpg_raw_acc = float(bpg_raw_pred == target_model_label)
                original_acc_values.append(original_acc)
                model_acc_values.append(model_acc)
                bpg_ldpc_acc_values.append(bpg_ldpc_acc)
                bpg_raw_acc_values.append(bpg_raw_acc)

            original_path = os.path.join(snr_dir, f"image_{idx:03d}_original.png")
            model_path = os.path.join(snr_dir, f"image_{idx:03d}_model.png")
            baseline_path = os.path.join(snr_dir, f"image_{idx:03d}_bpg_ldpc.png")
            raw_bpg_path = os.path.join(snr_dir, f"image_{idx:03d}_bpg_raw.png")

            _save_image(original_u8, original_path)
            _save_image(reconstructed_u8, model_path)
            _save_image(baseline_u8, baseline_path)
            _save_image(raw_bpg_u8, raw_bpg_path)

            per_image_rows.append({
                "snr_db": snr_db,
                "image_id": idx,
                "channel_uses": channel_uses,
                "bpg_max_bytes": max_bytes,
                "bpg_ldpc_ber": bit_error_rate,
                "bpg_ldpc_crc_ok": int(crc_ok),
                "bpg_raw_max_bytes": raw_bpg_max_bytes,
                "bpg_raw_ber": raw_bpg_ber,
                "mcs_k": mcs_k,
                "mcs_n": mcs_n,
                "mcs_m": mcs_m,
                "original_path": original_path,
                "model_path": model_path,
                "bpg_ldpc_path": baseline_path,
                "bpg_raw_path": raw_bpg_path,
                "model_psnr": model_psnr,
                "model_ssim": model_ssim,
                "bpg_ldpc_psnr": baseline_psnr,
                "bpg_ldpc_ssim": baseline_ssim,
                "bpg_raw_psnr": raw_bpg_psnr,
                "bpg_raw_ssim": raw_bpg_ssim,
            })
            if clip_handle is not None:
                per_image_rows[-1]["model_clip"] = model_clip
                per_image_rows[-1]["bpg_ldpc_clip"] = bpg_ldpc_clip
                per_image_rows[-1]["bpg_raw_clip"] = bpg_raw_clip
            if downstream_classifier is not None:
                per_image_rows[-1]["target_label"] = target_label
                per_image_rows[-1]["target_model_label"] = target_model_label
                per_image_rows[-1]["original_pred"] = original_pred
                per_image_rows[-1]["model_pred"] = model_pred
                per_image_rows[-1]["bpg_ldpc_pred"] = bpg_ldpc_pred
                per_image_rows[-1]["bpg_raw_pred"] = bpg_raw_pred
                per_image_rows[-1]["original_acc"] = original_acc
                per_image_rows[-1]["model_acc"] = model_acc
                per_image_rows[-1]["bpg_ldpc_acc"] = bpg_ldpc_acc
                per_image_rows[-1]["bpg_raw_acc"] = bpg_raw_acc

        summary_row = {
            "snr_db": snr_db,
            "num_images": len(eval_images),
            "channel_uses": channel_uses,
            "mcs_k": mcs_k,
            "mcs_n": mcs_n,
            "mcs_m": mcs_m,
            "bpg_max_bytes_mean": float(np.mean([row["bpg_max_bytes"] for row in per_image_rows if row["snr_db"] == snr_db])),
            "bpg_ldpc_ber_mean": float(np.mean([row["bpg_ldpc_ber"] for row in per_image_rows if row["snr_db"] == snr_db])),
            "bpg_ldpc_ber_std": float(np.std([row["bpg_ldpc_ber"] for row in per_image_rows if row["snr_db"] == snr_db])),
            "bpg_ldpc_crc_ok_mean": float(np.mean([row["bpg_ldpc_crc_ok"] for row in per_image_rows if row["snr_db"] == snr_db])),
            "bpg_raw_max_bytes_mean": float(np.mean([row["bpg_raw_max_bytes"] for row in per_image_rows if row["snr_db"] == snr_db])),
            "bpg_raw_ber_mean": float(np.mean([row["bpg_raw_ber"] for row in per_image_rows if row["snr_db"] == snr_db])),
            "bpg_raw_ber_std": float(np.std([row["bpg_raw_ber"] for row in per_image_rows if row["snr_db"] == snr_db])),
            "model_psnr_mean": float(np.mean(model_psnr_values)),
            "model_psnr_std": float(np.std(model_psnr_values)),
            "model_ssim_mean": float(np.mean(model_ssim_values)),
            "model_ssim_std": float(np.std(model_ssim_values)),
            "bpg_ldpc_psnr_mean": float(np.mean(baseline_psnr_values)),
            "bpg_ldpc_psnr_std": float(np.std(baseline_psnr_values)),
            "bpg_ldpc_ssim_mean": float(np.mean(baseline_ssim_values)),
            "bpg_ldpc_ssim_std": float(np.std(baseline_ssim_values)),
            "bpg_raw_psnr_mean": float(np.mean(raw_bpg_psnr_values)),
            "bpg_raw_psnr_std": float(np.std(raw_bpg_psnr_values)),
            "bpg_raw_ssim_mean": float(np.mean(raw_bpg_ssim_values)),
            "bpg_raw_ssim_std": float(np.std(raw_bpg_ssim_values)),
        }
        if clip_handle is not None:
            summary_row["model_clip_mean"] = float(np.mean(model_clip_values))
            summary_row["model_clip_std"] = float(np.std(model_clip_values))
            summary_row["bpg_ldpc_clip_mean"] = float(np.mean(bpg_ldpc_clip_values))
            summary_row["bpg_ldpc_clip_std"] = float(np.std(bpg_ldpc_clip_values))
            summary_row["bpg_raw_clip_mean"] = float(np.mean(bpg_raw_clip_values))
            summary_row["bpg_raw_clip_std"] = float(np.std(bpg_raw_clip_values))
        if downstream_classifier is not None:
            summary_row["original_acc_mean"] = float(np.mean(original_acc_values))
            summary_row["model_acc_mean"] = float(np.mean(model_acc_values))
            summary_row["bpg_ldpc_acc_mean"] = float(np.mean(bpg_ldpc_acc_values))
            summary_row["bpg_raw_acc_mean"] = float(np.mean(bpg_raw_acc_values))
        summary_rows.append(summary_row)

    per_image_metrics_path = os.path.join(output_dir, "metrics_per_image.tsv")
    summary_metrics_path = os.path.join(output_dir, "metrics_summary.tsv")

    with open(per_image_metrics_path, "w", newline="") as handle:
        fieldnames = [
            "snr_db",
            "image_id",
            "channel_uses",
            "bpg_max_bytes",
            "bpg_ldpc_ber",
            "bpg_ldpc_crc_ok",
            "bpg_raw_max_bytes",
            "bpg_raw_ber",
            "mcs_k",
            "mcs_n",
            "mcs_m",
            "original_path",
            "model_path",
            "bpg_ldpc_path",
            "bpg_raw_path",
            "model_psnr",
            "model_ssim",
            "bpg_ldpc_psnr",
            "bpg_ldpc_ssim",
            "bpg_raw_psnr",
            "bpg_raw_ssim",
        ]
        if clip_handle is not None:
            fieldnames.extend(["model_clip", "bpg_ldpc_clip", "bpg_raw_clip"])
        if downstream_classifier is not None:
            fieldnames.extend([
                "target_label",
                "target_model_label",
                "original_pred",
                "model_pred",
                "bpg_ldpc_pred",
                "bpg_raw_pred",
                "original_acc",
                "model_acc",
                "bpg_ldpc_acc",
                "bpg_raw_acc",
            ])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(per_image_rows)

    with open(summary_metrics_path, "w", newline="") as handle:
        fieldnames = [
            "snr_db",
            "num_images",
            "channel_uses",
            "mcs_k",
            "mcs_n",
            "mcs_m",
            "bpg_max_bytes_mean",
            "bpg_ldpc_ber_mean",
            "bpg_ldpc_ber_std",
            "bpg_ldpc_crc_ok_mean",
            "bpg_raw_max_bytes_mean",
            "bpg_raw_ber_mean",
            "bpg_raw_ber_std",
            "model_psnr_mean",
            "model_psnr_std",
            "model_ssim_mean",
            "model_ssim_std",
            "bpg_ldpc_psnr_mean",
            "bpg_ldpc_psnr_std",
            "bpg_ldpc_ssim_mean",
            "bpg_ldpc_ssim_std",
            "bpg_raw_psnr_mean",
            "bpg_raw_psnr_std",
            "bpg_raw_ssim_mean",
            "bpg_raw_ssim_std",
        ]
        if clip_handle is not None:
            fieldnames.extend([
                "model_clip_mean",
                "model_clip_std",
                "bpg_ldpc_clip_mean",
                "bpg_ldpc_clip_std",
                "bpg_raw_clip_mean",
                "bpg_raw_clip_std",
            ])
        if downstream_classifier is not None:
            fieldnames.extend([
                "original_acc_mean",
                "model_acc_mean",
                "bpg_ldpc_acc_mean",
                "bpg_raw_acc_mean",
            ])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    psnr_plot_path, ssim_plot_path, ber_plot_path, crc_plot_path, clip_plot_path, acc_plot_path = _write_snr_summary_plots(summary_rows, output_dir)

    print(f"Saved SNR sweep image triplets to {output_dir}")
    print(f"Saved per-image SNR sweep metrics to {per_image_metrics_path}")
    print(f"Saved summary SNR sweep metrics to {summary_metrics_path}")
    print(f"Saved mean PSNR plot to {psnr_plot_path}")
    print(f"Saved mean SSIM plot to {ssim_plot_path}")
    print(f"Saved mean BER plot to {ber_plot_path}")
    print(f"Saved mean CRC success plot to {crc_plot_path}")
    if clip_plot_path is not None:
        print(f"Saved mean CLIP plot to {clip_plot_path}")
    if acc_plot_path is not None:
        print(f"Saved mean downstream accuracy plot to {acc_plot_path}")

    return {
        "output_dir": output_dir,
        "per_image_metrics_path": per_image_metrics_path,
        "summary_metrics_path": summary_metrics_path,
        "psnr_plot_path": psnr_plot_path,
        "ssim_plot_path": ssim_plot_path,
        "ber_plot_path": ber_plot_path,
        "crc_plot_path": crc_plot_path,
        "clip_plot_path": clip_plot_path,
        "acc_plot_path": acc_plot_path,
        "channel_uses": channel_uses,
    }
