# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Adapted from fairseq-py

from logging import getLogger
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.layer_norm import LayerNorm
from ..modules.multihead_attention import MultiheadAttention
from ..modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from ..sequence_generator import SequenceGenerator

from . import LatentState


logger = getLogger()


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    ENC_ATTR = ['n_langs', 'n_words', 'dropout', 'padding_idx']

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout

        self.n_langs = args.n_langs
        self.n_langs = 2 #FIXME
        self.n_words = args.n_words
        embed_dim = args.encoder_embed_dim
        if args.share_lang_emb:
            assert len(set(args.n_words)) == 1
            logger.info("Sharing encoder input embeddings")
            layer_0 = Embedding(args.n_words[0], embed_dim - 32*0, args.pad_index)#TODO
            embeddings = [layer_0 for _ in range(self.n_langs)]
        else:
            #embeddings = [Embedding(n_words, embed_dim - 32*0, padding_idx=args.pad_index) for n_words in self.n_words]
            layer_0 = Embedding(args.n_words[0], embed_dim - 32*0, args.pad_index)#TODO
            embeddings = [layer_0 for _ in range(self.n_langs)]

        self.embeddings = nn.ModuleList(embeddings)
        self.freeze_enc_emb = args.freeze_enc_emb

        self.args = args
        domain_embed_dim = 512
        task_embed_dim = 512
        domain_layer_0 = Embedding(5, domain_embed_dim, 4) #TODO set this manually
        domain_embeddings = [domain_layer_0 for _ in range(self.n_langs)]
        task_layer_0 = Embedding(5, task_embed_dim, 4) #TODO set this manually
        task_embeddings = [task_layer_0 for _ in range(self.n_langs)]
        self.domain_embeddings = nn.ModuleList(domain_embeddings)
        self.task_embeddings = nn.ModuleList(task_embeddings)
        self.domain_embed_scale = math.sqrt(domain_embed_dim)
        self.task_embed_scale = math.sqrt(task_embed_dim)

        #lang_layer_0 = Embedding(5, 512, 4) #TODO set this manually
        #lang_embeddings = [lang_layer_0 for _ in range(self.n_langs)]
        #self.lang_embeddings = nn.ModuleList(lang_embeddings)
        #self.lang_embed_scale = math.sqrt(512)

        self.padding_idx = args.pad_index
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim -32*0, self.padding_idx, #TODO
            left_pad=args.left_pad_source,
        )

        self.layers = nn.ModuleList()
        self.domain_layers = nn.ModuleList()
        self.task_layers = nn.ModuleList()
        #self.lang_layers = nn.ModuleList()
        for k in range(args.encoder_layers):
            domain_layer = Embedding(5, domain_embed_dim, 4) #TODO set this manually
            self.domain_layers.append(domain_layer)
            task_layer = Embedding(5, task_embed_dim, 4) #TODO set this manually
            self.task_layers.append(task_layer)
            #lang_layer = Embedding(5, 512, 4) #TODO set this manually
            #self.lang_layers.append(lang_layer)


        for k in range(args.encoder_layers):
            # share top share_enc layers
            #layer_is_shared = (k >= (args.encoder_layers - args.share_enc))
            layer_is_shared = True
            if layer_is_shared:
                logger.info("Sharing encoder transformer parameters for layer %i" % k)

            self.layers.append(nn.ModuleList([
                TransformerEncoderLayer(args)
            ]))
            for i in range(1, self.n_langs):
                self.layers[k].append(self.layers[k][0])

        self.domain_fc = [Linear(embed_dim*2, embed_dim) for _ in range(args.encoder_layers)]
        self.domain_fc = nn.ModuleList(self.domain_fc)
        self.task_fc = [Linear(embed_dim*2, embed_dim) for _ in range(args.encoder_layers)]
        self.task_fc = nn.ModuleList(self.task_fc)
        #self.lang_fc = [Linear(embed_dim*2, embed_dim) for _ in range(args.encoder_layers)]
        #self.lang_fc = nn.ModuleList(self.lang_fc)
        self.domain_embed_fc = Linear(embed_dim*2, embed_dim)
        self.task_embed_fc = Linear(embed_dim*2, embed_dim)
        #self.lang_embed_fc = Linear(embed_dim*2, embed_dim)

    def forward(self, src_tokens, src_lengths, lang_id):
        assert type(lang_id) is int


        name = self.args.id2lang[lang_id]
        lang = name
        if '1' in name:
            domain_id = 0
        elif '2' in name:
            domain_id = 1
        elif '3' in name:
            domain_id = 2
        else:
            domain_id = 3

        if self.args.use_id != -1:
            domain_id = self.args.use_id

        if 'lm' in name:
            task = 'lm'
        else:
            task = 'nmt'
        if 'lm' in task:
            task_id = 0
        elif 'nmt' in task:
            task_id = 1
        else:
            assert(0)
        if 'en' in lang:
            lang_id = 0
        elif 'de' in lang or 'cs' in lang:
            lang_id = 1
        else:
            assert(0)

        embed_tokens = self.embeddings[lang_id]
        domain_embed_tokens = self.domain_embeddings[lang_id]
        task_embed_tokens = self.task_embeddings[lang_id]
        #lang_embed_tokens = self.lang_embeddings[lang_id]

        domain_tokens = torch.ones_like(src_tokens) * domain_id if domain_id >= 1 else torch.zeros_like(src_tokens)
        domain_tokens[src_tokens==4] = 4
        task_tokens = torch.ones_like(src_tokens) * task_id if task_id == 1 else torch.zeros_like(src_tokens)
        task_tokens[src_tokens==4] = 4
        #lang_tokens = torch.ones_like(src_tokens) * lang_id if lang_id >= 1 else torch.zeros_like(src_tokens)
        #lang_tokens[src_tokens==4] = 4
        domain_x = self.domain_embed_scale * domain_embed_tokens(domain_tokens) 
        task_x = self.task_embed_scale * task_embed_tokens(task_tokens) 
        #lang_x = self.lang_embed_scale * lang_embed_tokens(lang_tokens) 

        # embed tokens and positions
        x = self.embed_scale * embed_tokens(src_tokens)
        #print(x.shape)
        #print(domain_x.shape)
        #1/0
        x = x.detach() if self.freeze_enc_emb else x
        x += self.embed_positions(src_tokens)
        domain_concat = torch.cat([x, domain_x], dim=-1)
        domain_lambda = self.domain_embed_fc(domain_concat)
        task_concat = torch.cat([x, task_x], dim=-1)
        task_lambda = self.task_embed_fc(task_concat)
        #print(x.shape)
        #x = torch.cat([x, domain_x, task_x], -1)
        #x += domain_x + task_x + lang_x
        #x += domain_x
        #x = F.dropout(x, p=0.1, training=self.training) #add

        #y1 = F.dropout(domain_lambda*domain_x, p=0.1, training=self.training) #add
        #y2 = F.dropout(task_lambda*task_x, p=0.1, training=self.training)
        #x += y1 + y2
        #x += domain_lambda*domain_x + task_lambda*task_x
        if self.args.embed == True:
            if self.args.gate == True:
                x += domain_lambda*domain_x + task_lambda*task_x
            else:
                x += domain_x + task_x #+ lang_x
        #print(x.shape)
        #1/0
        #print(domain_tokens[0])
        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        encoder_padding_mask = src_tokens.t().eq(self.padding_idx)

        # encoder layers
        for i, layer in enumerate(self.layers):
            x = layer[lang_id](x, encoder_padding_mask)
            domain_x = self.domain_embed_scale * self.domain_layers[i](domain_tokens)
            task_x = self.task_embed_scale * self.task_layers[i](task_tokens)
            #lang_x = self.lang_embed_scale * self.lang_layers[i](lang_tokens)
            domain_concat = torch.cat([x, domain_x], dim=-1)
            domain_lambda = self.domain_fc[i](domain_concat)
            task_concat = torch.cat([x, task_x], dim=-1)
            task_lambda = self.task_fc[i](task_concat)
            #y1 = F.dropout(domain_lambda*domain_x, p=0.1, training=self.training) #add
            #y2 = F.dropout(task_lambda*task_x, p=0.1, training=self.training)
            #x = F.dropout(x, p=self.dropout, training=self.training) #add
            #x += y1 + y2# domain_lambda*domain_x + task_lambda*task_x
            #x += domain_lambda*domain_x + task_lambda*task_x
            #x += domain_x + task_x + lang_x
            if self.args.embed == True:
                if self.args.gate == True:
                    x += domain_lambda*domain_x + task_lambda*task_x
                else:
                    x += domain_x + task_x #+ lang_x

        return LatentState(
            input_len=src_lengths,
            dec_input={
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
            },
            dis_input=x,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    @staticmethod
    def expand_encoder_out_(encoder_out, beam_size):
        T, B, C = encoder_out['encoder_out'].size()
        assert encoder_out['encoder_padding_mask'].size() == (B, T)
        encoder_out['encoder_out'] = encoder_out['encoder_out'].repeat(1, 1, beam_size).view(T, -1, C)
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].repeat(1, beam_size).view(-1, T)


class TransformerDecoder(nn.Module):
    """Transformer decoder."""

    DEC_ATTR = ['n_langs', 'n_words', ('share_lang_emb', False), ('share_encdec_emb', False), ('share_decpro_emb', False), ('share_dec', False), 'dropout', 'eos_index', 'pad_index', 'bos_index']

    def __init__(self, args, encoder):
        super().__init__()
        self.dropout = args.dropout
        self.n_langs = args.n_langs
        self.n_langs = 2 #FIXME
        self.n_words = args.n_words
        self.share_lang_emb = args.share_lang_emb
        self.share_encdec_emb = args.share_encdec_emb
        self.share_decpro_emb = args.share_decpro_emb
        self.share_output_emb = args.share_output_emb
        self.share_dec = args.share_dec
        self.freeze_dec_emb = args.freeze_dec_emb
        self.encoder_class = encoder.__class__
        self.beam_size = args.beam_size
        self.length_penalty = args.length_penalty
    
        self.args = args

        # indexes
        self.eos_index = args.eos_index
        self.pad_index = args.pad_index
        self.bos_index = args.bos_index

        # words allowed for generation
        self.vocab_mask_neg = args.vocab_mask_neg if len(args.vocab) > 0 else None  # TODO: implement

        # embedding layers
        self.emb_dim = args.decoder_embed_dim
        if self.share_encdec_emb:
            logger.info("Sharing encoder and decoder input embeddings")
            embeddings = encoder.embeddings
        else:
            if self.share_lang_emb:
                logger.info("Sharing decoder input embeddings")
                layer_0 = Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
                embeddings = [layer_0 for _ in range(self.n_langs)]
            else:
                embeddings = [Embedding(n_words, self.emb_dim, padding_idx=self.pad_index) for n_words in self.n_words]
            embeddings = nn.ModuleList(embeddings)
        self.embeddings = embeddings
        self.embed_scale = math.sqrt(self.emb_dim)
        self.embed_positions = PositionalEmbedding(
            1024, self.emb_dim, self.pad_index,
            left_pad=args.left_pad_target,
        )

        domain_embed_dim = 512
        task_embed_dim = 512
        embed_dim = 512
        '''domain_layer_0 = Embedding(5, domain_embed_dim, 4) #TODO set this manually
        domain_embeddings = [domain_layer_0 for _ in range(self.n_langs)]
        task_layer_0 = Embedding(5, task_embed_dim, 4) #TODO set this manually
        task_embeddings = [task_layer_0 for _ in range(self.n_langs)]
        self.domain_embeddings = nn.ModuleList(domain_embeddings)
        self.task_embeddings = nn.ModuleList(task_embeddings)
        self.domain_embed_scale = math.sqrt(domain_embed_dim)
        self.task_embed_scale = math.sqrt(task_embed_dim)'''

        '''self.domain_layers = nn.ModuleList()
        self.task_layers = nn.ModuleList()
        for k in range(args.decoder_layers):
            domain_layer = Embedding(5, domain_embed_dim, 4) #TODO set this manually
            self.domain_layers.append(domain_layer)
            task_layer = Embedding(5, task_embed_dim, 4) #TODO set this manually
            self.task_layers.append(task_layer)'''

        proj_embed_dim = self.n_words[0] 
        '''domain_proj = Embedding(5, proj_embed_dim, 4) #TODO set this manually
        task_proj = Embedding(5, proj_embed_dim, 4) #TODO set this manually
        self.proj_embed_scale = math.sqrt(proj_embed_dim)
        self.domain_proj = domain_proj#nn.ModuleList(domain_projs)
        self.task_proj = task_proj#nn.ModuleList(task_projs)'''

        self.layers = nn.ModuleList()
        for k in range(args.decoder_layers):
            # share bottom share_dec layers
            layer_is_shared = True#(k < args.share_dec)
            if layer_is_shared:
                logger.info("Sharing decoder transformer parameters for layer %i" % k)

            #domain shared
            if True:# args.decoder_layers/2: #domain
                self.layers.append(nn.ModuleList([
                    TransformerDecoderLayer(args)
                ]))
                for i in range(1, self.n_langs):
                    self.layers[k].append(self.layers[k][0])
            else:
                self.layers.append(nn.ModuleList([
                    TransformerDecoderLayer(args)
                ]))
                #name_0 = args.id2lang[0]
                #lang, domain, task = name_0.split('_')
                #task = lang + task
            
                task_layer = {}
                #task_layer[task] = self.layers[k][0]
                for i in range(0, self.n_langs):
                    # layer for lang i
                    t_name = args.id2lang[i]
                    t_lang, t_domain, t_task = t_name.split('_')
                    t_task = t_domain #+ t_task
                    #t_task = t_lang + t_task
                        
                    if t_task in task_layer:
                        self.layers[k].append(task_layer[t_task])
                    else:
                        t_layer = (TransformerDecoderLayer(args))
                        task_layer[t_task] = t_layer
                        self.layers[k].append(task_layer[t_task])

                self.new_task_layers.append(nn.ModuleList([
                    TransformerDecoderLayer(args)
                ]))
                name_0 = args.id2lang[0]
                lang, domain, task = name_0.split('_')
                #task = lang + task
            
                new_task_layer = {}
                new_task_layer[task] = self.new_task_layers[k - 4 + my_layer_share_decoder][0]
                for i in range(1, self.n_langs):
                    # layer for lang i
                    t_name = args.id2lang[i]
                    t_lang, t_domain, t_task = t_name.split('_')
                    #t_task = t_domain #+ t_task
                    #t_task = t_lang + t_task
                        
                    if t_task in new_task_layer:
                        self.new_task_layers[k - 4 + my_layer_share_decoder].append(new_task_layer[t_task])
                    else:
                        t_layer = (TransformerDecoderLayer(args))
                        new_task_layer[t_task] = t_layer
                        self.new_task_layers[k - 4 + my_layer_share_decoder].append(new_task_layer[t_task])

        '''self.domain_fc = [Linear(embed_dim*2, embed_dim) for _ in range(args.encoder_layers)]
        self.domain_fc = nn.ModuleList(self.domain_fc)
        self.task_fc = [Linear(embed_dim*2, embed_dim) for _ in range(args.encoder_layers)]
        self.task_fc = nn.ModuleList(self.task_fc)
        self.lang_fc = [Linear(embed_dim*2, embed_dim) for _ in range(args.encoder_layers)]
        self.lang_fc = nn.ModuleList(self.lang_fc)
        self.domain_embed_fc = Linear(embed_dim*2, embed_dim)
        self.task_embed_fc = Linear(embed_dim*2, embed_dim)
        self.lang_embed_fc = Linear(embed_dim*2, embed_dim)'''

        # projection layers
        proj = [nn.Linear(self.emb_dim, n_words) for n_words in self.n_words]
        proj = [proj[i] for i in range(self.n_langs)]
        if self.share_decpro_emb:
            logger.info("Sharing input embeddings and projection matrix in the decoder")
            for i in range(self.n_langs):
                proj[i].weight = self.embeddings[i].weight
            if self.share_lang_emb:
                assert self.share_output_emb
                logger.info("Sharing decoder projection matrices")
                for i in range(1, self.n_langs):
                    proj[i].bias = proj[0].bias
        elif self.share_output_emb:
            assert self.share_lang_emb
            logger.info("Sharing decoder projection matrices")
            for i in range(1, self.n_langs):
                proj[i].weight = proj[0].weight
                proj[i].bias = proj[0].bias
        self.proj = nn.ModuleList(proj)


    def forward(self, encoded, y, lang_id, one_hot=False, incremental_state=None):
        assert not one_hot, 'one_hot=True has not been implemented for transformer'
        assert type(lang_id) is int


        name = self.args.id2lang[lang_id]
        try:
            lang, domain, task = name.split('_')
        except:
            domain='out'
            task='nmt'
            lang='en'
        if 'in' in domain:
            domain_id = 0
        elif 'out' in domain:
            domain_id = 1
        else:
            assert(0)
        if 'lm' in task:
            task_id = 0
        elif 'nmt' in task:
            task_id = 1
        else:
            assert(0)
        if 'en' in lang:
            lang_id = 0
        elif 'de' in lang or 'cs' in lang:
            lang_id = 1
        else:
            assert(0)

        '''lang = name
        if '1' in name:
            domain_id = 0
        elif '2' in name:
            domain_id = 1
        elif '3' in name:
            domain_id = 2
        else:
            domain_id = 3

        if 'lm' in name:
            task = 'lm'
        else:
            task = 'nmt'
        if 'lm' in task:
            task_id = 0
        elif 'nmt' in task:
            task_id = 1
        else:
            assert(0)
        if 'en' in lang:
            lang_id = 0
        elif 'de' in lang or 'cs' in lang:
            lang_id = 1
        else:
            assert(0)'''

        '''domain_embed_tokens = self.domain_embeddings[lang_id]
        task_embed_tokens = self.task_embeddings[lang_id]'''

        prev_output_tokens = y  # T x B
        encoder_out = encoded.dec_input
        embed_tokens = self.embeddings[lang_id]
        proj_layer = self.proj[lang_id]

        # embed positions
        positions = self.embed_positions(prev_output_tokens, incremental_state)

        # embed tokens and positions
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[-1:, :]  # only keep last time step

        '''domain_tokens = torch.ones_like(prev_output_tokens) if domain_id == 1 else torch.zeros_like(prev_output_tokens)
        domain_tokens[prev_output_tokens==4] = 4
        task_tokens = torch.ones_like(prev_output_tokens) if task_id == 1 else torch.zeros_like(prev_output_tokens)
        task_tokens[prev_output_tokens==4] = 4
        domain_x = self.domain_embed_scale * domain_embed_tokens(domain_tokens) 
        task_x = self.task_embed_scale * task_embed_tokens(task_tokens) '''

        x = self.embed_scale * embed_tokens(prev_output_tokens)
        x = x.detach() if self.freeze_dec_emb else x
        x += positions

        '''domain_concat = torch.cat([x, domain_x], dim=-1)
        domain_lambda = self.domain_embed_fc(domain_concat)
        task_concat = torch.cat([x, task_x], dim=-1)
        task_lambda = self.task_embed_fc(task_concat)'''
        #x += task_x + domain_x
        #x += task_lambda*task_x + domain_lambda*domain_x

        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for i, layer in enumerate(self.layers):
            x, attn = layer[lang_id](
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state=incremental_state,
            )
            '''if i < 0:
                domain_x = self.domain_embed_scale * self.domain_layers[i](domain_tokens)
                task_x = self.task_embed_scale * self.task_layers[i](task_tokens)
                #x += domain_x + task_x
                domain_concat = torch.cat([x, domain_x], dim=-1)
                domain_lambda = self.domain_fc[i](domain_concat)
                task_concat = torch.cat([x, task_x], dim=-1)
                task_lambda = self.task_fc[i](task_concat)
                #x += task_lambda*task_x + domain_lambda*domain_x'''

        # project back to size of vocabulary
        x = proj_layer(x)

        #domain_x = self.proj_embed_scale * self.domain_proj(domain_tokens)
        #task_x = self.proj_embed_scale * self.task_proj(task_tokens)
        #x += domain_x + task_x

        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def reorder_incremental_state_(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )
        self.apply(apply_reorder_incremental_state)

    def reorder_encoder_out_(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)

    def generate(self, encoded, lang_id, max_len=200, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        if self.beam_size > 0:
            return self.generate_beam(encoded, lang_id, self.beam_size, max_len, sample, temperature)

        encoder_out = encoded.dec_input
        latent = encoder_out['encoder_out']

        x_len = encoded.input_len
        is_cuda = latent.is_cuda
        one_hot = None

        # check inputs
        assert type(lang_id) is int
        assert latent.size() == (x_len.max(), x_len.size(0), self.emb_dim)
        assert (sample is True) ^ (temperature is None)

        # initialize generated sentences batch
        slen, bs = latent.size(0), latent.size(1)
        assert x_len.max() == slen and x_len.size(0) == bs
        cur_len = 1
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index)
        unfinished_sents = torch.LongTensor(bs).fill_(1)
        lengths = torch.LongTensor(bs).fill_(1)
        if is_cuda:
            decoded = decoded.cuda()
            unfinished_sents = unfinished_sents.cuda()
            lengths = lengths.cuda()
        decoded[0] = self.bos_index[lang_id]

        incremental_state = {}
        while cur_len < max_len:

            # previous word embeddings
            scores = self.forward(encoded, decoded[:cur_len], lang_id, one_hot, incremental_state)
            scores = scores.data[-1, :, :]  # T x B x V -> B x V

            # select next words: sample or one-hot
            if sample:
                next_words = torch.multinomial((scores / temperature).exp(), 1).squeeze(1)
            else:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            assert next_words.size() == (bs,)
            decoded[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            lengths.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len += 1

            # stop when there is a </s> in each sentence
            if unfinished_sents.max() == 0:
                break

        if cur_len == max_len:
            decoded[max_len - 1].masked_fill_(unfinished_sents.byte(), self.eos_index)
        assert (decoded == self.eos_index).sum() == bs

        return decoded[:cur_len], lengths, one_hot

    def generate_beam(self, encoded, lang_id, beam_size=20, max_len=175, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        self.encoder_class.expand_encoder_out_(encoded.dec_input, beam_size)

        x_len = encoded.input_len
        is_cuda = encoded.dec_input['encoder_out'].is_cuda
        one_hot = None

        # check inputs
        assert type(lang_id) is int
        # assert latent.size() == (x_len.max(), x_len.size(0) * beam_size, self.emb_dim)
        assert (sample is True) ^ (temperature is None)
        assert temperature is None, 'not supported'

        generator = SequenceGenerator(
            self, self.bos_index[lang_id], self.pad_index, self.eos_index,
            self.n_words[lang_id], beam_size=beam_size, maxlen=max_len, sampling=sample,
            len_penalty=self.length_penalty,
        )
        if is_cuda:
            x_len = x_len.cuda()
        results = generator.generate(x_len, encoded, lang_id)

        lengths = torch.LongTensor([sent[0]['tokens'].numel() for sent in results])
        lengths.add_(1)  # for BOS
        max_len = lengths.max()
        bsz = len(results)
        decoded = results[0][0]['tokens'].new(max_len, bsz).fill_(0)
        decoded[0, :] = self.bos_index[lang_id]
        for i, sent in enumerate(results):
            ntoks = sent[0]['tokens'].numel()  # pick the top beam result
            decoded[1:ntoks + 1, i] = sent[0]['tokens']

        return decoded, lengths, one_hot


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x, mask_future_timesteps=True,
            incremental_state=incremental_state, need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state, static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, init_size=num_embeddings)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m
