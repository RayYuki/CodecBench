from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import sys
from utils import load_and_fix_codec_model


class CodecForClassification:
    def __init__(self, args):
        self.model_name = args.model_name
        self.hidden_size = args.hidden_size
        self.vq = args.vq
        self.device = args.device
        self.sampling_rate = args.sampling_rate
        self.model_path = args.codec_ckpt
        self.model = load_and_fix_codec_model(args)

    def encode(self, inputs, mask):
        if self.model_name == "mimi":
            inputs = inputs.unsqueeze(1)
            z = self.model.encode_to_latent(inputs)
            outputs = z.transpose(1, 2).float()
            batch_size, max_length = mask.shape
            input_lengths = mask.sum(dim=1)  # [B]
            if torch.all(input_lengths == max_length):
                # No padding: directly compute mean
                embeddings = outputs.mean(dim=1)  # 均值池化
            else:
                # Padding exists: adjust mask for downsampling
                downsampled_lengths = []
                for length in input_lengths:
                    T = length.item()
                    for ratio in self.model.encoder.ratios:
                        T = np.ceil(T / ratio)
                    downsampled_lengths.append(int(T))
                downsampled_lengths = (
                    torch.tensor(downsampled_lengths)
                    .clamp(max=outputs.shape[1])
                    .to(self.device)
                )  # 限制不超过 T'
                # Generate downsampled mask
                downsampled_mask = torch.arange(
                    outputs.shape[1], device=self.device
                ).expand(batch_size, outputs.shape[1]) < downsampled_lengths.unsqueeze(
                    1
                )
                downsampled_mask = downsampled_mask.to(torch.float)  # [B, T']

                # Compute mean over time dimension, excluding padded regions
                masked_z = outputs * downsampled_mask.unsqueeze(-1)  # [B, T', D]
                valid_lengths = downsampled_mask.sum(dim=1).clamp(min=1)  # [B]
                embeddings = masked_z.sum(dim=1) / valid_lengths.unsqueeze(-1)

            return embeddings

        elif self.model_name == "dac" or self.model_name == "flowdec":
            from audiotools import AudioSignal

            inputs = inputs.unsqueeze(1)
            signal = AudioSignal(inputs, self.sampling_rate)
            signal.to(self.device)
            x = self.model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = self.model.encode(x, n_quantizers=self.vq)
            outputs = z.transpose(1, 2).float()
            batch_size, max_length = mask.shape
            input_lengths = mask.sum(dim=1)  # [B]
            if torch.all(input_lengths == max_length):
                # No padding: directly compute mean
                embeddings = outputs.mean(dim=1)  # 均值池化
            else:
                # Padding exists: adjust mask for downsampling
                downsampled_lengths = []
                for length in input_lengths:
                    T = length.item()
                    for stride in self.model.encoder_rates:
                        padding = math.ceil(stride / 2)
                        T = (T + 2 * padding - (2 * stride - 1)) // stride + 1
                    downsampled_lengths.append(int(T))
                downsampled_lengths = (
                    torch.tensor(downsampled_lengths)
                    .clamp(max=outputs.shape[1])
                    .to(self.device)
                )  # 限制不超过 T'
                # Generate downsampled mask
                downsampled_mask = torch.arange(
                    outputs.shape[1], device=self.device
                ).expand(batch_size, outputs.shape[1]) < downsampled_lengths.unsqueeze(
                    1
                )
                downsampled_mask = downsampled_mask.to(torch.float)  # [B, T']

                # Compute mean over time dimension, excluding padded regions
                masked_z = outputs * downsampled_mask.unsqueeze(-1)  # [B, T', D]
                valid_lengths = downsampled_mask.sum(dim=1).clamp(min=1)  # [B]
                embeddings = masked_z.sum(dim=1) / valid_lengths.unsqueeze(-1)

            return embeddings

        elif self.model_name == "maskgct":
            inputs = inputs.unsqueeze(1)

            with torch.no_grad():
                codes = self.model["encoder"](inputs)
                quantized_out, vq = self.model["decoder"].quantize(
                    codes, n_quantizers=self.vq
                )

            outputs = quantized_out.transpose(1, 2).float()
            batch_size, max_length = mask.shape
            input_lengths = mask.sum(dim=1)  # [B]
            if torch.all(input_lengths == max_length):
                # No padding: directly compute mean
                embeddings = outputs.mean(dim=1)  # 均值池化
            else:
                # Padding exists: adjust mask for downsampling
                downsampled_lengths = []
                for length in input_lengths:
                    T = length.item()
                    for stride in [3, 4, 5, 8]:
                        padding = math.ceil(stride / 2)
                        T = (T + 2 * padding - (2 * stride - 1)) // stride + 1
                    downsampled_lengths.append(int(T))
                downsampled_lengths = (
                    torch.tensor(downsampled_lengths)
                    .clamp(max=outputs.shape[1])
                    .to(self.device)
                )  # 限制不超过 T'
                # Generate downsampled mask
                downsampled_mask = torch.arange(
                    outputs.shape[1], device=self.device
                ).expand(batch_size, outputs.shape[1]) < downsampled_lengths.unsqueeze(
                    1
                )
                downsampled_mask = downsampled_mask.to(torch.float)  # [B, T']

                # Compute mean over time dimension, excluding padded regions
                masked_z = outputs * downsampled_mask.unsqueeze(-1)  # [B, T', D]
                valid_lengths = downsampled_mask.sum(dim=1).clamp(min=1)  # [B]
                embeddings = masked_z.sum(dim=1) / valid_lengths.unsqueeze(-1)

            return embeddings

        elif self.model_name == "bigcodec":
            # print('input shape', inputs.shape)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() == 2:
                if inputs.shape[0] == 2:  # [2, T] 立体声
                    inputs = torch.mean(inputs, dim=0, keepdim=True)  # [1, T]
            inputs = F.pad(inputs, (0, (200 - (inputs.shape[1] % 200))))
            with torch.no_grad():
                vq_emb = self.model["encoder"](inputs.unsqueeze(1))
                vq_post_emb, vq_code, _ = self.model["decoder"](vq_emb, vq=True)
            outputs = vq_post_emb.transpose(1, 2)
            bs, max_length = mask.shape
            input_lengths = mask.sum(dim=1)  # [B]
            if torch.all(input_lengths == max_length):
                # No padding: directly compute mean
                embeddings = outputs.mean(dim=1)  # 均值池化
            else:
                # print('bs, len', bs, max_length)
                # print('output shape', outputs.shape)
                len_prime = max_length // 200 + 1
                # print(len_prime)
                re_length = len_prime * 200
                # print(re_length)
                mask = F.pad(
                    mask, (0, re_length - max_length), mode="constant", value=0
                )
                # print('mask shape', mask.shape)
                # print('output shape', outputs.shape)
                mask_reshaped = mask.view(bs, len_prime, 200)
                mask_downsampled = mask_reshaped.max(dim=-1)[0]
                # print('mask downsample shape', mask_downsampled.shape)
                mask_downsampled = mask_downsampled.unsqueeze(-1)  # [bs, len', 1]
                masked_sum = (outputs * mask_downsampled).sum(dim=1)  # [bs, dim]
                # print('mask sum', masked_sum.shape)
                mask_sum = mask_downsampled.sum(dim=1)  # [bs, 1]
                mask_sum = torch.clamp(mask_sum, min=1.0)  # 避免除以0
                embeddings = masked_sum / mask_sum
                # print('embed', embeddings.shape)

            return embeddings

        elif self.model_name == "xcodec2":
            # print('input shape', inputs.shape)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() == 2:
                if inputs.shape[0] == 2:  # [2, T] 立体声
                    inputs = torch.mean(inputs, dim=0, keepdim=True)  # [1, T]
            inputs = F.pad(inputs, (0, (320 - (inputs.shape[1] % 320))))
            bs, _ = inputs.shape
            feats = []
            for i in range(bs):
                feat = (
                    self.model["feature_extractor"](
                        F.pad(inputs[i, i + 1 :].cpu(), (160, 160)),
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                    )
                    .data["input_features"]
                    .to(self.device)
                )
                feats.append(feat)
            feat = torch.stack(feats, dim=0).squeeze(1)
            vq_emb = self.model["encoder"](inputs.unsqueeze(1))
            vq_emb = vq_emb.transpose(1, 2)
            semantic_target = self.model["semantic_model"](feat[:, :, :])
            semantic_target = semantic_target.hidden_states[16]
            semantic_target = semantic_target.transpose(1, 2)
            semantic_target = self.model["SemanticEncoder_module"](semantic_target)
            # print(semantic_target.shape, vq_emb.shape)
            vq_emb = torch.cat([semantic_target, vq_emb], dim=1)
            vq_emb = self.model["fc_prior"](vq_emb.transpose(1, 2)).transpose(1, 2)
            _, vq_code, _ = self.model["decoder"](vq_emb, vq=True)  # vq_code here !!!!
            vq_post_emb = self.model["decoder"].quantizer.get_output_from_indices(
                vq_code.transpose(1, 2)
            )
            vq_post_emb = self.model["fc_post_a"](vq_post_emb)

            outputs = vq_post_emb
            bs, max_length = mask.shape
            input_lengths = mask.sum(dim=1)  # [B]
            if torch.all(input_lengths == max_length):
                # No padding: directly compute mean
                embeddings = outputs.mean(dim=1)  # 均值池化
            else:
                # print('bs, len', bs, max_length)
                # print('output shape', outputs.shape)
                len_prime = max_length // 320 + 1
                # print(len_prime)
                re_length = len_prime * 320
                # print(re_length)
                mask = F.pad(
                    mask, (0, re_length - max_length), mode="constant", value=0
                )
                # print('mask shape', mask.shape)
                # print('output shape', outputs.shape)
                mask_reshaped = mask.view(bs, len_prime, 320)
                mask_downsampled = mask_reshaped.max(dim=-1)[0]
                # print('mask downsample shape', mask_downsampled.shape)
                mask_downsampled = mask_downsampled.unsqueeze(-1)  # [bs, len', 1]
                masked_sum = (outputs * mask_downsampled).sum(dim=1)  # [bs, dim]
                # print('mask sum', masked_sum.shape)
                mask_sum = mask_downsampled.sum(dim=1)  # [bs, 1]
                mask_sum = torch.clamp(mask_sum, min=1.0)  # 避免除以0
                embeddings = masked_sum / mask_sum
                # print('embed', embeddings.shape)

            return embeddings

        elif self.model_name == "whisper":
            inputs = inputs.cpu().numpy()
            input_feature = self.model["feature_extractor"](
                inputs,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                truncation=True,
            ).input_features

            input_feature = input_feature.to(self.device)
            res = self.model["model"].encoder(input_feature)
            z = res.last_hidden_state
            n_fft = self.model["feature_extractor"].n_fft
            hop_length = self.model["feature_extractor"].hop_length
            outputs = z.float()
            batch_size, max_length = mask.shape
            input_lengths = mask.sum(dim=1)  # [B]
            if torch.all(input_lengths == max_length):
                # No padding: directly compute mean
                embeddings = outputs.mean(dim=1)  # 均值池化
            else:
                # Padding exists: adjust mask for downsampling
                downsampled_lengths = []
                for length in input_lengths:
                    T = length.item()
                    if T == 0:
                        downsampled_lengths.append(0)
                        continue
                    T_prime = (T - n_fft) // hop_length + 1  # mel frames
                    T_double_prime = (T_prime + 1) // 2  # conv2 stride=2
                    downsampled_lengths.append(int(T_double_prime))
                downsampled_lengths = (
                    torch.tensor(downsampled_lengths)
                    .clamp(max=outputs.shape[1])
                    .to(self.device)
                )  # 限制不超过 T'
                # Generate downsampled mask
                downsampled_mask = torch.arange(
                    outputs.shape[1], device=self.device
                ).expand(batch_size, outputs.shape[1]) < downsampled_lengths.unsqueeze(
                    1
                )
                downsampled_mask = downsampled_mask.to(torch.float)  # [B, T']

                # Compute mean over time dimension, excluding padded regions
                masked_z = outputs * downsampled_mask.unsqueeze(-1)  # [B, T', D]
                valid_lengths = downsampled_mask.sum(dim=1).clamp(min=1)  # [B]
                embeddings = masked_z.sum(dim=1) / valid_lengths.unsqueeze(-1)

            return embeddings
