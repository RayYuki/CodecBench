try:
    import s3prl.hub as hub
except ImportError:
    hub = None
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from torchaudio.transforms import Resample
from tqdm import tqdm

class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        result = torch.cat(out, dim=1)
        return result


""" Conv1d + BatchNorm1d + ReLU
"""


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


""" The SE connection of 1D case.
"""


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out

class SE_Res2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale,
        se_bottleneck_dim,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels, kernel_size, stride, padding, dilation, scale=scale
        )
        self.Conv1dReluBn2 = Conv1dReluBn(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, attention_channels, kernel_size=1
            )  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, attention_channels, kernel_size=1
            )  # equals W and b in the paper
        self.linear2 = nn.Conv1d(
            attention_channels, in_dim, kernel_size=1
        )  # equals V and k in the paper

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10
            ).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        channels=512,
        emb_dim=192,
        global_context_att=False,
        feat_type="fbank",
        sr=16000,
        feature_selection="hidden_states",
        update_extract=False,
        config_path=None,
    ):
        if hub is None:
            raise ImportError("Please `pip install s3prl` to run ECAPA models")
        super().__init__()

        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        if feat_type == "fbank" or feat_type == "mfcc":
            self.update_extract = False

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == "fbank":
            self.feature_extract = trans.MelSpectrogram(
                sample_rate=sr,
                n_fft=512,
                win_length=win_len,
                hop_length=hop_len,
                f_min=0.0,
                f_max=sr // 2,
                pad=0,
                n_mels=feat_dim,
            )
        elif feat_type == "mfcc":
            melkwargs = {
                "n_fft": 512,
                "win_length": win_len,
                "hop_length": hop_len,
                "f_min": 0.0,
                "f_max": sr // 2,
                "pad": 0,
            }
            self.feature_extract = trans.MFCC(
                sample_rate=sr, n_mfcc=feat_dim, log_mels=False, melkwargs=melkwargs
            )
        else:
            self.feature_extract = getattr(hub, feat_type)()  # hub.wavlm_large()
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
                self.feature_extract.model.encoder.layers[23].self_attn,
                "fp32_attention",
            ):
                self.feature_extract.model.encoder.layers[
                    23
                ].self_attn.fp32_attention = False
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
                self.feature_extract.model.encoder.layers[11].self_attn,
                "fp32_attention",
            ):
                self.feature_extract.model.encoder.layers[
                    11
                ].self_attn.fp32_attention = False

            self.feat_num = self.get_feat_num()
            self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type != "fbank" and feat_type != "mfcc":
            freeze_list = [
                "final_proj",
                "label_embs_concat",
                "mask_emb",
                "project_q",
                "quantizer",
            ]
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[0],
            self.channels[1],
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            self.channels[1],
            self.channels[2],
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            self.channels[2],
            self.channels[3],
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(
            self.channels[-1],
            attention_channels=128,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type == "fbank" or self.feat_type == "mfcc":
                    x = self.feature_extract(x) + 1e-6  # B x feat_dim x time_len
                else:
                    x = self.feature_extract([sample for sample in x])

        if self.feat_type == "fbank":
            x = x.log()

        if self.feat_type != "fbank" and self.feat_type != "mfcc":
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = (
                F.softmax(self.feature_weight, dim=-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out

def ECAPA_TDNN_SMALL(
    feat_dim,
    emb_dim=256,
    feat_type="fbank",
    sr=16000,
    feature_selection="hidden_states",
    update_extract=False,
    config_path=None,
):
    return ECAPA_TDNN(
        feat_dim=feat_dim,
        channels=512,
        emb_dim=emb_dim,
        feat_type=feat_type,
        sr=sr,
        feature_selection=feature_selection,
        update_extract=update_extract,
        config_path=config_path,
    )

def init_model(model_name, checkpoint=None):
    if model_name == "unispeech_sat":
        config_path = "config/unispeech_sat.th"
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="unispeech_sat", config_path=config_path
        )
    elif model_name == "wavlm_base_plus":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=768, feat_type="wavlm_base_plus", config_path=config_path
        )
    elif model_name == "wavlm_large":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wavlm_large", config_path=config_path
        )
    elif model_name == "hubert_large":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="hubert_large_ll60k", config_path=config_path
        )
    elif model_name == "wav2vec2_xlsr":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wav2vec2_xlsr", config_path=config_path
        )
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type="fbank")

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"], strict=False)
    return model

class ValleEncoder(nn.Module):
    def __init__(self, ckpt_path, model_name="wavlm_large", use_cuda=True):
        super(ValleEncoder, self).__init__()
        self.model = init_model(model_name, ckpt_path)
        if use_cuda and torch.cuda.is_available():
            self.model.cuda()

    def forward(self, audio_paths, is_path=True):
        assert is_path, f"Only pass path to Valle encoder"
        self.eval()
        if is_path:
            embs = []
            for i_audio, audio_path in tqdm(enumerate(audio_paths)):
                audio = self.load_audio(audio_path)
                audio = audio.cuda()
                with torch.no_grad():
                    emb = self.model(audio)
                embs.append(emb.cpu().numpy())
            embs = np.concatenate(embs, axis=0)
        else:
            raise ValueError(f"Only accepts audio paths for Valle encoder")
        return embs

    def load_audio(self, audio_path):
        wav1, sr1 = sf.read(audio_path)
        if wav1.ndim == 2 and wav1.shape[1] > 1:
            wav1 = np.mean(wav1, axis=1)
        wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=16000)
        wav1 = resample1(wav1)
        return wav1
    
def compute_cosine_similarity(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    xs and ys arrays can be multi-dimensional; the similarity is computed along the last dimension.
    Their expected dimension is [batch_size, embedding_dim].
    """
    # xs/ys: [N, F]
    xnorm, ynorm = np.linalg.norm(xs, axis=-1), np.linalg.norm(ys, axis=-1)
    result = (
        np.matmul(np.expand_dims(xs, axis=-2), np.expand_dims(ys, axis=-1))
        .squeeze(axis=-1)
        .squeeze(axis=-1)
    )
    result = result / np.multiply(xnorm, ynorm)
    return result