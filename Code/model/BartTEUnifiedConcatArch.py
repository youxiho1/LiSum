import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BartForSequenceClassification, BartForConditionalGeneration
from typing import Optional, List
from transformers import BartModel
from utils.UnifiedModelOutput import UnifiedModelOutput
from transformers import BartPretrainedModel

import os

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartTEUnifiedConcatArch(BartPretrainedModel):
    _keys_to_ignore_on_load_unexpected = []
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BartModel(config)
        self.classification_loss_weight = config.classification_loss_weight
        self.summarization_loss_weight = config.summarization_loss_weight
        self.dropout_ratio = config.dropout
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.concat_mode = config.concat_mode
        self.num_classifier = config.classifiers_num
        self.num_labels = config.classification_output_dim
        self.d_model = config.d_model

        # task embedding attention matrix
        # initialize randomly
        self.attention_matrix_encoder = nn.Parameter(torch.rand(config.d_model, config.classifiers_num))
        self.attention_matrix_decoder = nn.Parameter(torch.rand(config.d_model, config.classifiers_num))
        # fix mode
        # self.attention_matrix_encoder = torch.rand(config.d_model, config.classifiers_num).cuda()
        # self.attention_matrix_decoder = torch.rand(config.d_model, config.classifiers_num).cuda()
        
        # self.attention_matrix = nn.Parameter(torch.rand(config.classifiers_num, config.d_model))
        
        # only one classifier
        # dense layer 1s
        if self.concat_mode == 'concat':
            self.fc1 = nn.Linear(config.d_model*2, 512)
        else:
            self.fc1 = nn.Linear(config.d_model, 512)
        # relu activation function
        self.relu = nn.ReLU()
        # dropout layer
        self.dropout = nn.Dropout(self.dropout_ratio)
        # dense layer 2 (output layer)
        self.fc2 = nn.Linear(512, config.classification_output_dim)
        # Cross Entropy自带softmax
        # self.softmax = nn.LogSoftmax(dim=1)
        self.post_init()

    def mean_pooling_with_attention_mask(self, x, attention_mask):
        """Calc mean value of input tensor with attention mask

        Args:
            x: torch.FloatTensor, the last hidden states of the current batch
            attention_mask: torch.IntTensor, the input attention mask

        Returns:
            obj:'torch.FloatTensor' of shape:obj:'(batch_size, hidden_dim)': mean value of the input
        """
        return (x * attention_mask.unsqueeze(-1)).sum(dim=1)/attention_mask.float().sum(-1).unsqueeze(-1)
    
    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        actions: Optional[torch.LongTensor]  = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=0.1)
            # loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if self.concat_mode == 'encoder':
            encoder_last_hidden_state = outputs["encoder_last_hidden_state"]
            hidden_size = encoder_last_hidden_state.shape[2]
            mat = torch.matmul(encoder_last_hidden_state, self.attention_matrix_encoder, out=None)
            mat = F.softmax(mat, dim=1)
            mat = mat.permute(2,0,1)
            mat = mat.unsqueeze(mat.dim()).repeat(1,1,1,hidden_size)
            encoder_last_hidden_state = encoder_last_hidden_state.unsqueeze(0).repeat(self.num_classifier,1,1,1)
            hidden_state = (encoder_last_hidden_state * mat).sum(2)
            classification_input = hidden_state
        elif self.concat_mode == 'decoder':
            decoder_last_hidden_state = outputs["last_hidden_state"]
            hidden_size = decoder_last_hidden_state.shape[2]
            mat = torch.matmul(decoder_last_hidden_state, self.attention_matrix_decoder, out=None)
            mat = F.softmax(mat, dim=1)
            mat = mat.permute(2,0,1)
            mat = mat.unsqueeze(mat.dim()).repeat(1,1,1,hidden_size)
            decoder_last_hidden_state = decoder_last_hidden_state.unsqueeze(0).repeat(self.num_classifier,1,1,1)
            hidden_state = (decoder_last_hidden_state * mat).sum(2)
            classification_input = hidden_state
            # if labels is not None:
            #     eos_mask = decoder_input_ids.eq(self.config.eos_token_id)
            # else:
            
            #     eos_mask = 
            # eos_mask = decoder_input_ids.eq(self.config.eos_token_id)
            # # print(eos_mask.shape)

            # if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            #     raise ValueError("All examples must have the same number of <eos> tokens.")
            # t = hidden_states[eos_mask, :   ]
            # sentence_representation = t.view(hidden_states.size(0), -1, hidden_states.size(-1))[
            #     :, -1, :
            # ]
            
        elif self.concat_mode == 'concat':
            encoder_last_hidden_state = outputs["encoder_last_hidden_state"]
            hidden_size = encoder_last_hidden_state.shape[2]
            mat = torch.matmul(encoder_last_hidden_state, self.attention_matrix_encoder, out=None)
            mat = F.softmax(mat, dim=1)
            mat = mat.permute(2,0,1)
            mat = mat.unsqueeze(mat.dim()).repeat(1,1,1,hidden_size)
            encoder_last_hidden_state = encoder_last_hidden_state.unsqueeze(0).repeat(self.num_classifier,1,1,1)
            hidden_state_encoder = (encoder_last_hidden_state * mat).sum(2)
            
            decoder_last_hidden_state = outputs["last_hidden_state"]
            hidden_size = decoder_last_hidden_state.shape[2]
            mat = torch.matmul(decoder_last_hidden_state, self.attention_matrix_decoder, out=None)
            mat = F.softmax(mat, dim=1)
            mat = mat.permute(2,0,1)
            mat = mat.unsqueeze(mat.dim()).repeat(1,1,1,hidden_size)
            decoder_last_hidden_state = decoder_last_hidden_state.unsqueeze(0).repeat(self.num_classifier,1,1,1)
            hidden_state_decoder = (decoder_last_hidden_state * mat).sum(2)
            # print(hidden_state_encoder.shape)       
            # print(hidden_state_decoder.shape)
            # hidden_state_encoder.shape = hidden_state_decoder.shape = c * b * h
            classification_input = torch.concat((hidden_state_encoder, hidden_state_decoder),2)
            print(classification_input.shape)
            # classification_input.shape = c * b * 2h
        
        x = self.fc1(classification_input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x.shape = c * b * num_labels
        classification_logits = x
            
        # Calculate Cross Entropy Loss
        classification_loss = torch.tensor(0.0).cuda()
        if actions is not None: 
            ce_loss = nn.CrossEntropyLoss()
            for i in range(self.num_classifier):
                loss = ce_loss(x[i,:,:], actions[:,i])
                classification_loss = classification_loss + loss
            classification_loss = classification_loss / self.num_classifier
            
        total_loss = None
        if actions is not None and labels is not None:
            total_loss = self.summarization_loss_weight * masked_lm_loss + self.classification_loss_weight * classification_loss / (classification_loss / masked_lm_loss).detach()

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return UnifiedModelOutput(
            loss=total_loss,
            classification_loss=classification_loss,
            summarization_loss=masked_lm_loss,
            logits=lm_logits,
            classification_logits=classification_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )    
        
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def init_attention_matrix_by_task(self, tokenizer, max_fulltext_length):
        with torch.no_grad():
            encoder = self.get_encoder()
            decoder = self.get_decoder()
            action_list = ['Distribute', 'Modify', 'Commercial Use', 'Hold Liable', 'Include Copyright', 
                       'Include License', 'Sublicense', 'Use Trademark', 'Private Use', 'Disclose Source',
                       'State Changes', 'Place Warranty', 'Include Notice', 'Include Original', 'Give Credit',
                       'Use Patent Claims', 'Rename', 'Relicense', 'Contact Author', 'Include Install Instructions',
                       'Compensate for Damages', 'Statically Link', 'Pay Above Use Threshold']
            
            action_tokenized = tokenizer(action_list, padding=False, truncation=True, max_length=max_fulltext_length)
            input_ids_list = action_tokenized['input_ids']
            attention_mask_list = action_tokenized['attention_mask']
            for i in range(len(input_ids_list)):
                input_ids = torch.tensor(input_ids_list[i]).unsqueeze(0)
                attention_mask = torch.tensor(attention_mask_list[i]).unsqueeze(0)
                encoder_outputs = encoder(input_ids, attention_mask)
                encoder_last_hidden_state = encoder_outputs["last_hidden_state"]
                # encoder_last_hidden_state.shape = batch_size * sequence_length * hidden_size
                # hidden_state.shape = batch_size * hidden_size
                
                hidden_state_encoder = self.mean_pooling_with_attention_mask(encoder_last_hidden_state[:,1:-2], attention_mask[:,1:-2])
                # print(hidden_state.shape)
                # self.attention_matrix[:,i] = hidden_state
                
                self.attention_matrix_encoder[:,i] = hidden_state_encoder
                
                decoder_last_hidden_state = decoder(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask = attention_mask
                )["last_hidden_state"]
                hidden_state_decoder = self.mean_pooling_with_attention_mask(decoder_last_hidden_state, attention_mask)
                self.attention_matrix_decoder[:,i] = hidden_state_decoder
                
                # attention_matrix: c*h
        return  
    
    def init_attention_matrix_by_desp(self, tokenizer, max_fulltext_length):
        with torch.no_grad():
            encoder = self.get_encoder()
            decoder = self.get_decoder()
            action_desp_list = [
                'the ability to distribute original or modified (derivative) works',
                'the ability to modify the software and create derivatives',
                'the ability to use the software for commercial purposes',
                'the warranty and if the software/license owner can be charged for damages',
                'whether the original copyright must be retained',
                'including the full text of license in modified software',
                'the ability for you to grant/extend a license to the software',
                'the allowance of using contributors\' names, trademarks or logos',
                'may use the work for private use',
                'whether you must disclose your source code when you distribute the software',
                'stating significant changes made to software',
                'the ability to place warranty on the software licensed',
                'include a \"NOTICE\" file with attribution notes included',
                'whether copies of the original software or instructions to obtain copies must be distributed with the software',
                'the author of licensed binaries must display their name, professional identification, and a relevant url on software launch',
                'the rights to practice patent claims of contributors to the code',
                'the ability to change software\'s name if modified/distributed',
                'the ability to freely relicense the software',
                'your ability to contact the author about the module you\'re using',
                'if the software is part of a consumer device, you must include the installation information necessary to modify and reinstall the software',
                'the obligation to compensate the author for any damages cased by your work',
                'whether the library can be compiled into the program (linked at compile time rather than runtime, aka dynamic linking)',
                'the obligation to pay the licensor after a certain amount of use'   
            ]
            
            action_tokenized = tokenizer(action_desp_list, padding=False, truncation=True, max_length=max_fulltext_length)
            input_ids_list = action_tokenized['input_ids']
            attention_mask_list = action_tokenized['attention_mask']
            for i in range(len(input_ids_list)):
                input_ids = torch.tensor(input_ids_list[i]).unsqueeze(0)
                attention_mask = torch.tensor(attention_mask_list[i]).unsqueeze(0)
                encoder_outputs = encoder(input_ids, attention_mask)
                encoder_last_hidden_state = encoder_outputs["last_hidden_state"]
                # encoder_last_hidden_state.shape = batch_size * sequence_length * hidden_size
                # hidden_state.shape = batch_size * hidden_size
                
                hidden_state_encoder = self.mean_pooling_with_attention_mask(encoder_last_hidden_state, attention_mask)
                # print(hidden_state.shape)
                # self.attention_matrix[:,i] = hidden_state
                self.attention_matrix_encoder[:,i] = hidden_state_encoder
                decoder_last_hidden_state = decoder(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask = attention_mask
                )["last_hidden_state"]
                hidden_state_decoder = self.mean_pooling_with_attention_mask(decoder_last_hidden_state, attention_mask)
                self.attention_matrix_decoder[:,i] = hidden_state_decoder
                
                # attention_matrix: c*h
        return  
    