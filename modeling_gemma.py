import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SigLipVisionConfig, SiglipVisionModel
class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size, 
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim = 256,
        max_position_embeddings = 8192,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        seld.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig():
    
    def __init__(
        self,
        vision_config=None, 
        text_config=None,
        ignore_index = 100,
        image_token_index = 256000,
        vocab_size = 257152,
        projection_dim = 2048,
        hidden_size = 2048,
        pad_token_id = None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        
        self.vision_config = SigLipVisionConfig(**vision_config)
        self.text_config = text_config
        
        self.text_config = GemmaCofig(**text_config, pad_token=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class 
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lmhead = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lmhead.weight = self.model.embed_tokens.weight
    
    def forward(self,
                attention_mask,
                position_ids,
                input_embeds,
                kv_cache):
        
        outputs = self.model(attention_mask=attention_mask,
                             position_ids=position_ids,
                             input_embeds=input_embeds,
                             kv_cache=kv_cache
                             )
        
        hidden_states = outputs
        logits = self.lmhead(hidden_states)
        return {"logits": logits.float()}        
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states
    
class PaliGemmaForConditionalGeneration(nn.Module):
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        language_model = GemmaForCausaLM(config.text_config)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1


    def tie_weights(self):
        return self.language_model.tie_weights()

    def merge_input_ids_with_image_features(self,
                                            image_features,
                                            input_embeds,
                                            input_ids,
                                            attention_mask,
                                            kv_cache):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        scaled_image_features = image_features/(self.config.hidden_size)**0.5
        final_embeddings = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)
        #shape: [Batch_size, sequence_length]
        #input_ids = [[567, 567, 567, 567, 567, 1, 3, 4, 6, 7, 4, 2],[...]] where 1 is <bos> and 2 is \n
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id
        
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1), embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1), embed_dim)
        #copying text embeddings into the final embeddings
        final_embeddings = torch.where(text_mask_expanded, input_embeds, final_embeddings)
        #copying the image embeddings
        final_embeddings = final_embeddings.masked_scatter(image_mask_expanded, scaled_image_features)
    
        #estimating the attention mask
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]
        
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)
        
        causal_mask = causal_mask.unsqueeze(1)
        
        if kv_cache is not None or kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).mask_fill_((attention_mask == 0), 1).to(device)
        
        return final_embeddings, causal_mask, position_ids        
    
    def forward(
        self,
        input_ids: torch.LongTensor = None, 
        pixel_values: torch.FloatTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"
        #1. Extract the input embeddings, shape: (batch_size, seq_len, hidden_states)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        #2. Merge text and images
        # [batch_size, channels, h, w] -> [batch_size, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_states]
        image_features = self.multi_modal_projector(selected_image_feature)
        
        inputs_embeds, attention_mask, position_ids = self.merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )
        
        return outputs