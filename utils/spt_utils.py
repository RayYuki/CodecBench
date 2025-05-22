import logging
import torch
import sys

def load_and_fix_speechtokenizer(config_path, ckpt_path, device=torch.device("cuda")):
    from speechtokenizer.model import SpeechTokenizer
    speechtokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    speechtokenizer = speechtokenizer.to(device)
    speechtokenizer.eval()
    
    for param in speechtokenizer.parameters():
        param.requires_grad = False
    
    logging.info(f"Load and fix speechtokenizer of config: {config_path} from checkpoint: {ckpt_path} success")
    
    return speechtokenizer

def load_and_fix_stablecodec(config_path, ckpt_path, device=torch.device("cuda")):
    from stable_codec import StableCodec
    
    model = StableCodec(model_config_path=config_path, ckpt_path=ckpt_path, device=device)
    model.set_posthoc_bottleneck('1x46656_400bps')
    # model.set_posthoc_bottleneck()
    return model

def load_and_fix_xcodec(ckpt_path, device=torch.device("cuda")):
    sys.path.append('/inspire/hdd/project/embodied-multimodality/public/X-Codec-2.0')
    from vq.codec_encoder import CodecEncoder
    from vq.codec_decoder_vocos import CodecDecoderVocos
    from vq.module import SemanticDecoder,SemanticEncoder
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
    import torch.nn as nn
    
    ckpt=torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    from collections import OrderedDict
    # 步骤 2：提取并过滤 'codec_enc' 和 'generator' 部分
    filtered_state_dict_codec = OrderedDict()
    filtered_state_dict_semantic_encoder = OrderedDict()
    filtered_state_dict_gen = OrderedDict()
    filtered_state_dict_fc_post_a = OrderedDict()
    filtered_state_dict_fc_prior = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('CodecEnc.'):
            # 去掉 'codec_enc.' 前缀，以匹配 CodecEncoder 的参数名
            new_key = key[len('CodecEnc.'):]
            filtered_state_dict_codec[new_key] = value
        elif key.startswith('generator.'):
            # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
            new_key = key[len('generator.'):]
            filtered_state_dict_gen[new_key] = value
        elif key.startswith('fc_post_a.'):
            # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
            new_key = key[len('fc_post_a.'):]
            filtered_state_dict_fc_post_a[new_key] = value
        elif key.startswith('SemanticEncoder_module.'):
            # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
            new_key = key[len('SemanticEncoder_module.'):]
            filtered_state_dict_semantic_encoder[new_key] = value
        elif key.startswith('fc_prior.'):
            # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
            new_key = key[len('fc_prior.'):]
            filtered_state_dict_fc_prior[new_key] = value
    semantic_model = Wav2Vec2BertModel.from_pretrained("/inspire/hdd/project/embodied-multimodality/public/model/w2vbert2", output_hidden_states=True)

    semantic_model.eval().cuda()
    SemanticEncoder_module = SemanticEncoder(1024,1024,1024)
    SemanticEncoder_module.load_state_dict(filtered_state_dict_semantic_encoder)
    SemanticEncoder_module = SemanticEncoder_module.eval().cuda()
    encoder = CodecEncoder()
    encoder.load_state_dict(filtered_state_dict_codec)
    encoder = encoder.eval().cuda()
    decoder = CodecDecoderVocos()
    decoder.load_state_dict(filtered_state_dict_gen)
    decoder = decoder.eval().cuda()
    fc_post_a = nn.Linear( 2048, 1024 )
    fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)
    fc_post_a = fc_post_a.eval().cuda()
    fc_prior = nn.Linear( 2048, 2048 )
    fc_prior.load_state_dict(filtered_state_dict_fc_prior)
    fc_prior = fc_prior.eval().cuda()
    feature_extractor = AutoFeatureExtractor.from_pretrained("/inspire/hdd/project/embodied-multimodality/public/model/w2vbert2")
    
    return {"feature_extractor": feature_extractor, "encoder": encoder, "semantic_model": semantic_model, "SemanticEncoder_module": SemanticEncoder_module, "fc_prior": fc_prior, "decoder": decoder, "sample_rate": 16000}

def load_and_fix_glm(ckpt_path, device=torch.device("cuda")):
    sys.path.append('/inspire/hdd/project/embodied-multimodality/public/GLM-4-Voice')
    from transformers import WhisperFeatureExtractor
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from speech_tokenizer.utils import extract_speech_token
    
    encoder = WhisperVQEncoder.from_pretrained(ckpt_path).eval().cuda()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(ckpt_path)
    
    return {"encoder": encoder, "feature_extractor": feature_extractor}

def load_and_fix_mimi(ckpt_path, device=torch.device("cuda")):
    from moshi.models import loaders
    model = loaders.get_mimi(ckpt_path, device=device)
    model.set_num_codebooks(32)  # up to 32 for mimi, but limited to 8 for moshi.
    model.cuda()
    model.eval()

    return model

def load_and_fix_dac(ckpt_path, device=torch.device("cuda")):
    import dac
    model = dac.DAC.load(ckpt_path)
    model.to(device)
    model.eval()

    return model

def load_and_fix_xcodec2(ckpt_path, device=torch.device("cuda")):
    sys.path.append('/inspire/hdd/project/embodied-multimodality/public/CodecEvaluation/x-codec2')
    from model_x_codec import X_Codec2
    model = X_Codec2(semantic_position='after_fsq_before_post_proj', ckpt_path=ckpt_path)
    model.to(device)
    model.eval()
    
    return model

def load_and_fix_bigcodec(ckpt_path, device=torch.device("cuda")):
    sys.path.append('/inspire/hdd/project/embodied-multimodality/public/CodecEvaluation/BigCodec')
    from model_bigcodec import BigCodec
    model = BigCodec(ckpt_path)
    model.to(device)
    model.eval()
    
    return model
    
def load_and_fix_codec_model(args):
    if args.model_type == "SpeechTokenizer":
        codec_model = load_and_fix_speechtokenizer(args.config, args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == "xcodec":
        codec_model = load_and_fix_xcodec(args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == "stablecodec":
        codec_model = load_and_fix_stablecodec(args.config, args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == "glm4voice":
        codec_model = load_and_fix_glm(args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == 'mimi':
        codec_model = load_and_fix_mimi(args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == 'dac':
        codec_model = load_and_fix_dac(args.codec_ckpt)
        target_frame_rate_before_ctc = 75
    elif args.model_type == "xcodec2":
        codec_model = load_and_fix_xcodec2(args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == "bigcodec":
        codec_model = load_and_fix_bigcodec(args.codec_ckpt)
        target_frame_rate_before_ctc = 80
    elif args.model_type == '<your model>':
        """
        sys.path.append("<model dir>")
        codec_model = ...
        target_frame_rate_before_ctc = ...
        """
    else:
        assert False, f'model type {args.model_type} not support !'
        
    if isinstance(codec_model, dict):
        pass
    else:
        for param in codec_model.parameters():
            param.requires_grad = False
    
    return codec_model, target_frame_rate_before_ctc