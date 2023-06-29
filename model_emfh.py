from utils.fc import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F


class MFB(nn.Module):
    def __init__(self, __C, d_words, d_text, is_first, d_bert):
        super(MFB, self).__init__()
        self.__C = __C
        self.is_first = is_first
        self.proj_i = nn.Linear(d_words, __C.MFB_K * __C.MFB_O)
        self.proj_q = nn.Linear(d_text, __C.MFB_K * __C.MFB_O)
        self.proj_ent = nn.Linear(d_bert, __C.MFB_K * __C.MFB_O)
        self.proj_struct = nn.Linear(d_bert, __C.MFB_K * __C.MFB_O)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.pool = nn.AvgPool1d(__C.MFB_K, stride=__C.MFB_K)

    def forward(self, words_feat, text_att_feat, ents_feat, struct_feat, exp_in=1):
        batch_size = words_feat.shape[0]
        words_feat = self.proj_i(words_feat)
        text_feat = self.proj_q(text_att_feat)
        ents_feat = self.proj_ent(ents_feat)
        struct_feat = self.proj_struct(struct_feat)

        exp_out = words_feat * text_feat * ents_feat * struct_feat
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)
        z = self.pool(exp_out) * self.__C.MFB_K
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))
        z = z.view(batch_size, -1, self.__C.MFB_O)
        return z, exp_out


class TextAtt(nn.Module):
    def __init__(self, __C, d_text):
        super(TextAtt, self).__init__()
        self.__C = __C
        self.mlp = MLP(
            in_size=d_text,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.Q_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.lstm = nn.LSTM(__C.Q_GLIMPSES, __C.Q_GLIMPSES, __C.n_layers, dropout=__C.DROPOUT_R, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, text_feat):
        text_att_maps = self.mlp(text_feat)
        text_att_maps, _ = self.lstm(text_att_maps)
        text_att_maps = self.relu(text_att_maps)
        text_att_maps = F.softmax(text_att_maps, dim=1)

        text_att_feat_list = []
        for i in range(self.__C.Q_GLIMPSES):
            mask = text_att_maps[:, :, i:i + 1]
            mask = mask * text_feat
            mask = torch.sum(mask, dim=1)
            text_att_feat_list.append(mask)
        text_att_feat = torch.cat(text_att_feat_list, dim=1)

        return text_att_feat


class WordsAtt(nn.Module):
    def __init__(self, __C, d_words, d_text_att, d_bert):
        super(WordsAtt, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.mfb = MFB(__C, d_words, d_text_att, True, d_bert)
        self.mlp = MLP(
            in_size=__C.MFB_O,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.I_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.lstm = nn.LSTM(__C.I_GLIMPSES, __C.I_GLIMPSES, __C.n_layers, dropout=__C.DROPOUT_R, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, words_feat, text_att_feat, ents_feat, struct_feat):
        text_att_feat = text_att_feat.unsqueeze(1)
        words_feat = self.dropout(words_feat)
        z, _ = self.mfb(words_feat, text_att_feat, ents_feat, struct_feat)

        words_att_maps = self.mlp(z)
        words_att_maps, _ = self.lstm(words_att_maps)
        words_att_maps = self.relu(words_att_maps)
        words_att_maps = F.softmax(words_att_maps, dim=1)

        words_att_feat_list = []
        for i in range(self.__C.I_GLIMPSES):
            mask = words_att_maps[:, :, i:i + 1]
            mask = mask * words_feat
            mask = torch.sum(mask, dim=1)
            words_att_feat_list.append(mask)
        words_att_feat = torch.cat(words_att_feat_list, dim=1)

        return words_att_feat


class ResEMFH(nn.Module):
    def __init__(self, __C, d_bert):
        super(ResEMFH, self).__init__()
        self.__C = __C
        self.d_words = d_bert
        self.d_text = d_bert
        self.d_words_att = self.d_words * __C.I_GLIMPSES
        self.d_text_att = self.d_words * __C.Q_GLIMPSES

        self.text_att = TextAtt(__C, self.d_text)
        self.words_att = WordsAtt(__C, self.d_words, self.d_text_att, d_bert)
        self.sig = nn.Sigmoid()

        if self.__C.HIGH_ORDER:
            self.mfh1 = MFB(__C, self.d_words_att, self.d_text_att, True, d_bert)
            self.mfh2 = MFB(__C, self.d_words_att, self.d_text_att, False, d_bert)
            self.mfh3 = MFB(__C, self.d_words_att, self.d_text_att, False, d_bert)
            self.mfh4 = MFB(__C, self.d_words_att, self.d_text_att, False, d_bert)
            self.mfh5 = MFB(__C, self.d_words_att, self.d_text_att, False, d_bert)
            self.mfh6 = MFB(__C, self.d_words_att, self.d_text_att, False, d_bert)
        else:
            self.mfb = MFB(__C, self.d_words_att, self.d_text_att, True, d_bert)

    def forward(self, words_feat, text_feat, ents_feat, struct_feat):
        words_feat = words_feat.unsqueeze(1)
        text_feat = text_feat.unsqueeze(1)
        ents_feat = ents_feat.unsqueeze(1)
        struct_feat = struct_feat.unsqueeze(1)

        text_att_feat = self.text_att(text_feat)
        words_att_feat = self.words_att(words_feat, text_att_feat, ents_feat, struct_feat)

        if self.__C.HIGH_ORDER:
            z1, exp1 = self.mfh1(words_att_feat.unsqueeze(1), text_att_feat.unsqueeze(1), ents_feat, struct_feat)
            residual = z1
            z2, exp2 = self.mfh1(words_att_feat.unsqueeze(1), text_att_feat.unsqueeze(1), ents_feat, struct_feat, exp1)
            z3, exp3 = self.mfh1(words_att_feat.unsqueeze(1), text_att_feat.unsqueeze(1), ents_feat, struct_feat, exp2)
            z3 = z3 + residual
            residual = z3
            z4, exp4 = self.mfh1(words_att_feat.unsqueeze(1), text_att_feat.unsqueeze(1), ents_feat, struct_feat, exp3)
            z5, exp5 = self.mfh1(words_att_feat.unsqueeze(1), text_att_feat.unsqueeze(1), ents_feat, struct_feat, exp4)
            z6, _ = self.mfh2(words_att_feat.unsqueeze(1), text_att_feat.unsqueeze(1), ents_feat, struct_feat, exp5)
            z6 = z6 + residual
            z = torch.mean(torch.cat((z1, z2, z3, z4, z5, z6), 1), dim=1, keepdim=False)
        else:
            z, _ = self.mfb(words_att_feat.unsqueeze(1), text_att_feat.unsqueeze(1), ents_feat, struct_feat)
            z = z.squeeze(1)

        return z
