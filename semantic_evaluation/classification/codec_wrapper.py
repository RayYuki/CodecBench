from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from codec_for_classification import CodecForClassification
from classification_model import Model

class CodecWrapper(Model):
    def __init__(self, args):
        
        self.hidden_size = args.hidden_size
        self.device = args.device
        self.sampling_rate = args.sampling_rate
        self.train_backbone = args.train_backbone

        self.model = CodecForClassification(args)

    def get_embeddings(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
        sampling_rate: Optional[int] = None,
    ) -> torch.Tensor:

        inputs = audio.to(self.device)
        mask = mask.to(self.device)
        if self.train_backbone:
            outputs = self.model.encode(inputs, mask)
        else:
            with torch.no_grad():
                outputs = self.model.encode(inputs, mask)
            
        return outputs

    # 其他方法保持不变
    def get_classification_embedding_size(self) -> int:
        return self.hidden_size

    def get_token_embedding_size(self) -> int:
        return self.hidden_size

    def get_sampling_rate(self) -> int:
        return self.sampling_rate

    def get_embedding_layer(self) -> int:
        return self.hidden_size