import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.modules.container import ModuleList
from transformers import BertModel, BertTokenizer
from model_emfh import ResEMFH
from models.model_transformer import make_model
import pickle
from models.attn import DistSparseAtt
import matplotlib.pyplot as plt
import seaborn as sns


def save_pkl(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class ArgModule(nn.Module):
    def __init__(self, arg_layer, n_layers):
        super(ArgModule, self).__init__()
        self.layers = _get_clones(arg_layer, n_layers)
        self.n_layers = n_layers

    def forward(self, encoded, predicate, pred_mask=None):
        output = encoded
        for layer_idx in range(self.n_layers):
            output = self.layers[layer_idx](target=output, source=predicate,
                                            key_mask=pred_mask)
        return output


class ArgExtractorLayer(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 d_feedforward=2048,
                 dropout=0.1,
                 activation='relu'):
        super(ArgExtractorLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.ds_attn = DistSparseAtt(False, 5, attention_dropout=0.05, output_attention=True)
        self.n_heads = n_heads

    def forward(self, target, source, key_mask=None):
        max_len, bs, d_hidden = target.shape
        target = target.view(max_len, bs, self.n_heads, d_hidden // self.n_heads)
        target = target.transpose(0, 1)
        source = source.view(max_len, bs, self.n_heads, d_hidden // self.n_heads)
        source = source.transpose(0, 1)
        attended, attn_rmh = self.ds_attn(target, source, source, key_mask)

        attended = attended.reshape(bs, max_len, d_hidden)
        attended = attended.transpose(0, 1)
        target = target.transpose(0, 1)
        target = target.reshape(max_len, bs, d_hidden)
        # source = source.transpose(0, 1)
        # source = source.reshape(max_len, bs, d_hidden)

        skipped = target + self.dropout1(attended)
        normed = self.norm1(skipped)

        projected = self.linear2(self.dropout2(self.activation(self.linear1(normed))))
        skipped = normed + self.dropout1(projected)
        normed = self.norm2(skipped)
        return normed


class InteractionAware(nn.Module):
    def __init__(self,
                 bert_config='bert-base-cased',
                 mh_dropout=0.1,
                 pred_clf_dropout=0.,
                 arg_clf_dropout=0.3,
                 n_arg_heads=8,
                 n_arg_layers=4,
                 pos_emb_dim=64,
                 pred_n_labels=3,
                 arg_n_labels=9,
                 emfh_args=None):
        super(InteractionAware, self).__init__()
        self.pred_n_labels = pred_n_labels
        self.arg_n_labels = arg_n_labels

        self.bert = BertModel.from_pretrained(bert_config, output_hidden_states=True)
        self.bert_p = BertModel.from_pretrained(bert_config, output_hidden_states=True)
        self.bert_p.resize_token_embeddings(28998)
        d_model = self.bert.config.hidden_size
        self.d_bert = d_model
        self.pred_dropout = nn.Dropout(pred_clf_dropout)

        self.fc_pred = nn.Linear(d_model + 8 + 16, d_model)
        self.pred_classifier = nn.Linear(d_model, self.pred_n_labels)

        self.position_emb = nn.Embedding(3, pos_emb_dim, padding_idx=0)
        d_model = d_model + d_model + pos_emb_dim + 176
        # d_model = d_model + 32
        arg_layer = ArgExtractorLayer(
            d_model=d_model,
            n_heads=n_arg_heads,
            dropout=mh_dropout)
        self.arg_module = ArgModule(arg_layer, n_arg_layers)
        self.arg_dropout = nn.Dropout(arg_clf_dropout)
        self.arg_classifier = nn.Linear(d_model, arg_n_labels)

        self.EMFH = ResEMFH(emfh_args, self.d_bert)

        self.change_ent = nn.Linear(200, self.d_bert)
        self.fc_fuse = nn.Linear(8, 128)
        self.fc_word = nn.Linear(768, 12)

    def forward(self,
                input_ids,
                attention_mask,
                input_ids_p,
                attention_mask_p,
                idx_phrs,
                entity_embs,
                predicate_mask=None,
                predicate_hidden=None,
                total_pred_labels=None,
                arg_labels=None
                ):
        # predicate extraction
        bert_hidden, pooler_output, bert_hid_all = self.bert(input_ids, attention_mask)
        pred_logit = self.pred_classifier(self.pred_dropout(bert_hidden))

        # predicate loss
        if total_pred_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = pred_logit.view(-1, self.pred_n_labels)
            active_labels = torch.where(active_loss, total_pred_labels.view(-1),
                                        torch.tensor(loss_fct.ignore_index).type_as(total_pred_labels))
            pred_loss = loss_fct(active_logits, active_labels)

        pred_feature = _get_pred_feature(bert_hidden, predicate_mask)
        position_vectors = self.position_emb(
            _get_position_idxs(predicate_mask, input_ids))

        feat_sent = pooler_output
        feat_word = self.get_word_features(bert_hidden.shape[0], bert_hidden.shape[1], bert_hid_all)
        feat_word_fuse = torch.mean(feat_word, dim=1, keepdim=False)
        feat_phrs = self.get_phr_features(idx_phrs, input_ids_p, attention_mask_p)
        feat_ents = torch.mean(entity_embs, dim=1, keepdim=False).float()
        feat_ents = self.change_ent(feat_ents)
        fuse_feat = self.EMFH(feat_sent, feat_word_fuse, feat_phrs, feat_ents)
        fuse_feat = fuse_feat.view(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        fuse_feat = self.fc_fuse(fuse_feat)
        feat_sent = feat_sent.reshape(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        feat_word = self.fc_word(feat_word)
        feat_phrs = feat_phrs.reshape(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        feat_ents = feat_ents.reshape(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        multi_feat = torch.cat([feat_sent, feat_word, feat_phrs, feat_ents, fuse_feat],
                               dim=2)
        bert_hidden = torch.cat([bert_hidden, pred_feature, position_vectors, multi_feat],
                                dim=2)
        bert_hidden = bert_hidden.transpose(0, 1)
        arg_hidden = self.arg_module(bert_hidden, bert_hidden, predicate_mask)
        arg_hidden = arg_hidden.transpose(0, 1)
        arg_logit = self.arg_classifier(self.arg_dropout(arg_hidden))

        if arg_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = arg_logit.view(-1, self.arg_n_labels)
            active_labels = torch.where(active_loss, arg_labels.view(-1),
                                        torch.tensor(loss_fct.ignore_index).type_as(arg_labels))
            arg_loss = loss_fct(active_logits, active_labels)

        # total loss
        batch_loss = pred_loss + arg_loss
        outputs = (batch_loss, pred_loss, arg_loss)
        return outputs

    def get_word_features(self, batch_size, max_len, bert_hid_all):
        word_feature = []
        for batch_i in range(batch_size):
            token_embeddings = []
            for token_i in range(max_len):
                hidden_layers = []
                for layer_i in range(len(bert_hid_all)):
                    vec = bert_hid_all[layer_i][batch_i][token_i]
                    hidden_layers.append(vec)
                token_embeddings.append(hidden_layers)
            summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0).cpu().detach().numpy().tolist() for layer in
                                    token_embeddings]

            word_feature.append(summed_last_4_layers)
        word_feature = torch.tensor(word_feature, requires_grad=True, device='cuda:0')
        return word_feature

    def get_phr_features(self, idx_phrs, input_ids_p, attention_mask_p):
        bert_hidden_p, pooler_output_p, bert_hid_all_p = self.bert_p(input_ids_p, attention_mask_p)
        idx_phrs = idx_phrs.cpu().numpy().tolist()
        for i, idx_phr in enumerate(idx_phrs):
            num_phr = int((len(idx_phr) - idx_phr.count(-1)) / 2)
            if num_phr > 0:

                for j, idx in enumerate(idx_phr[:num_phr * 2]):
                    if j == 0:
                        feat_phr = bert_hidden_p[i, idx, :].unsqueeze(0)
                    else:
                        feat_phr = torch.cat((feat_phr, bert_hidden_p[i, idx, :].unsqueeze(0)), 0)
                feat_phr = torch.mean(feat_phr, dim=0, keepdim=True)
            else:
                feat_phr = torch.zeros((1, bert_hidden_p.shape[2]), requires_grad=True).type_as(
                    bert_hidden_p)
            if i == 0:
                feat_phrs = feat_phr
            else:
                feat_phrs = torch.cat((feat_phrs, feat_phr), dim=0)
        return feat_phrs

    def extract_predicate(self,
                          input_ids,
                          attention_mask,
                          input_ids_p,
                          attention_mask_p,
                          idx_phrs,
                          entity_embs,
                          ):
        bert_hidden, pooler_output, bert_hid_all = self.bert(input_ids, attention_mask)
        pred_logit = self.pred_classifier(bert_hidden)

        return pred_logit, bert_hidden, pooler_output, bert_hid_all

    def extract_argument(self,
                         input_ids,
                         bert_hidden,
                         predicate_mask,
                         input_ids_p,
                         attention_mask_p,
                         idx_phrs,
                         entity_embs,
                         pooler_output,
                         bert_hid_all):
        pred_feature = _get_pred_feature(bert_hidden, predicate_mask)
        position_vectors = self.position_emb(_get_position_idxs(predicate_mask, input_ids))

        feat_sent = pooler_output
        feat_word = self.get_word_features(bert_hidden.shape[0], bert_hidden.shape[1], bert_hid_all)
        feat_word_fuse = torch.mean(feat_word, dim=1, keepdim=False)
        feat_phrs = self.get_phr_features(idx_phrs, input_ids_p, attention_mask_p)
        feat_ents = torch.mean(entity_embs, dim=1, keepdim=False).float()
        feat_ents = self.change_ent(feat_ents)
        fuse_feat = self.EMFH(feat_sent, feat_word_fuse, feat_phrs, feat_ents)
        fuse_feat = fuse_feat.view(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        fuse_feat = self.fc_fuse(fuse_feat)
        feat_sent = feat_sent.reshape(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        feat_word = self.fc_word(feat_word)
        feat_phrs = feat_phrs.reshape(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        feat_ents = feat_ents.reshape(bert_hidden.shape[0], bert_hidden.shape[1], -1)
        multi_feat = torch.cat([feat_sent, feat_word, feat_phrs, feat_ents, fuse_feat],
                               dim=2)
        arg_input = torch.cat([bert_hidden, pred_feature, position_vectors, multi_feat],
                              dim=2)
        arg_input = arg_input.transpose(0, 1)
        arg_hidden = self.arg_module(arg_input, arg_input, predicate_mask)
        arg_hidden = arg_hidden.transpose(0, 1)
        return self.arg_classifier(arg_hidden)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_position_idxs(pred_mask, input_ids):
    position_idxs = torch.zeros(pred_mask.shape, dtype=int, device=pred_mask.device)
    for mask_idx, cur_mask in enumerate(pred_mask):
        position_idxs[mask_idx, :] += 2
        cur_nonzero = (cur_mask == 0).nonzero()
        start = torch.min(cur_nonzero).item()
        end = torch.max(cur_nonzero).item()
        position_idxs[mask_idx, start:end + 1] = 1
        pad_start = max(input_ids[mask_idx].nonzero()).item() + 1
        position_idxs[mask_idx, pad_start:] = 0
    return position_idxs


def _get_pred_feature(pred_hidden, pred_mask):
    B, L, D = pred_hidden.shape
    pred_features = torch.zeros((B, L, D), device=pred_mask.device)
    for mask_idx, cur_mask in enumerate(pred_mask):
        pred_position = (cur_mask == 0).nonzero().flatten()
        pred_feature = torch.mean(pred_hidden[mask_idx, pred_position], dim=0)
        pred_feature = torch.cat(L * [pred_feature.unsqueeze(0)])
        pred_features[mask_idx, :, :] = pred_feature
    return pred_features
