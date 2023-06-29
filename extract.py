import os
import torch
import numpy as np
import utils.bio as bio
from transformers import BertTokenizer
from tqdm import tqdm
import logging

logger = logging.getLogger('root')



def extract(args,
            model,
            loader,
            output_path,
            constraint=False):

    model.eval()
    os.makedirs(output_path, exist_ok=True)
    extraction_path = os.path.join(output_path, "extraction.txt")
    tokenizer = BertTokenizer.from_pretrained(args.bert_config)
    new_tokens = ['<PHRASE>', '</PHRASE>']
    tokenizer.add_tokens(new_tokens)
    f = open(extraction_path, 'w')

    for step, batch in tqdm(enumerate(loader), desc='eval_steps', total=len(loader)):
        if len(batch) != 8:
            print('error!')
        token_strs = [[word for word in sent] for sent in
                      np.asarray(batch[2]).T]
        sentences = batch[3]
        token_ids, att_mask = map(lambda x: x.to(args.device), batch[:2])
        token_ids_padded_p, att_mask_p, idx_phr, entity_embs = map(lambda x: x.to(args.device), batch[4:8])

        with torch.no_grad():
            pred_logit, pred_hidden, pooler_output, bert_hid_all = model.extract_predicate(input_ids=token_ids,
                                                                                           attention_mask=att_mask,
                                                                                           input_ids_p=token_ids_padded_p,
                                                                                           attention_mask_p=att_mask_p,
                                                                                           idx_phrs=idx_phr,
                                                                                           entity_embs=entity_embs)
            pred_tags = torch.argmax(pred_logit, 2)
            pred_tags = bio.filter_pred_tags(pred_tags, token_strs)
            pred_tags = bio.get_single_predicate_idxs(pred_tags)
            pred_probs = torch.nn.Softmax(2)(pred_logit)

            num_arg = 0
            for cur_pred_tags, cur_pred_hidden, cur_att_mask, cur_token_id, cur_pred_probs, token_str, sentence, \
                cur_token_ids_padded_p, cur_att_mask_p, cur_idx_phr, cur_entity_embs, cur_pooler_output \
                    in zip(pred_tags, pred_hidden, att_mask, token_ids, pred_probs, token_strs, sentences,
                           token_ids_padded_p, att_mask_p, idx_phr, entity_embs, pooler_output):
                cur_pred_masks = bio.get_pred_mask(cur_pred_tags).to(args.device)
                n_predicates = cur_pred_masks.shape[0]
                if n_predicates == 0:
                    continue
                cur_pred_hidden = torch.cat(n_predicates * [cur_pred_hidden.unsqueeze(0)])
                cur_token_id = torch.cat(n_predicates * [cur_token_id.unsqueeze(0)])
                cur_token_ids_padded_p = torch.cat(n_predicates * [cur_token_ids_padded_p.unsqueeze(0)])
                cur_att_mask_p = torch.cat(n_predicates * [cur_att_mask_p.unsqueeze(0)])
                cur_idx_phr = torch.cat(n_predicates * [cur_idx_phr.unsqueeze(0)])
                cur_entity_embs = torch.cat(n_predicates * [cur_entity_embs.unsqueeze(0)])
                cur_pooler_output = torch.cat(n_predicates * [cur_pooler_output.unsqueeze(0)])
                cur_bert_hid_all = []
                for each_bert_hid_all in bert_hid_all:
                    temp_all = each_bert_hid_all[num_arg]
                    temp_all = torch.cat(n_predicates * [temp_all.unsqueeze(0)])
                    cur_bert_hid_all.append(temp_all)
                cur_bert_hid_all = tuple(cur_bert_hid_all)
                num_arg += 1
                cur_arg_logit = model.extract_argument(
                    input_ids=cur_token_id,
                    bert_hidden=cur_pred_hidden,
                    predicate_mask=cur_pred_masks,
                    input_ids_p=cur_token_ids_padded_p,
                    attention_mask_p=cur_att_mask_p,
                    idx_phrs=cur_idx_phr,
                    entity_embs=cur_entity_embs,
                    pooler_output=cur_pooler_output,
                    bert_hid_all=cur_bert_hid_all
                )

                cur_arg_tags = torch.argmax(cur_arg_logit, 2)
                cur_arg_probs = torch.nn.Softmax(2)(cur_arg_logit)
                cur_arg_tags = bio.filter_arg_tags(cur_arg_tags, cur_pred_tags, token_str, constraint)

                cur_extractions, cur_extraction_idxs = bio.get_tuple(sentence, cur_pred_tags, cur_arg_tags, tokenizer,
                                                                     constraint)
                cur_confidences = bio.get_confidence_score(cur_pred_probs, cur_arg_probs, cur_extraction_idxs)
                for extraction, confidence in zip(cur_extractions, cur_confidences):
                    if args.binary:
                        f.write("\t".join([sentence] + [str(1.0)] + extraction[:3]) + '\n')
                    else:
                        f.write("\t".join([sentence] + [str(confidence)] + extraction) + '\n')
    f.close()
    logger.info("Extraction Done.")
