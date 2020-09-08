import os.path
import random
from collections import defaultdict


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, BertForQuestionAnswering, BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import compute_predictions_logits, get_final_text, squad_evaluate
from transformers.modeling_utils import PreTrainedModel

TRAIN_DATA_FN = os.path.join(os.path.pardir, os.path.pardir, 'SQuAD2', 'cached_train_model_384')
EVAL_DATA_FN = os.path.join(os.path.pardir, os.path.pardir, 'SQuAD2', 'cached_dev_model_384')
QA_MODEL_DIR = os.path.join(os.path.pardir, os.path.pardir, 'BERT-Tiny', 'output')
SR_MODEL_DIR = os.path.join(os.path.pardir, os.path.pardir, 'BERT-Tiny', 'sr_model')
IR_MODEL_DIR_CLS = os.path.join(os.path.pardir, os.path.pardir, 'BERT-Tiny', 'ir_model', 'ifv')
IR_MODEL_DIR_SPAN = os.path.join(os.path.pardir, os.path.pardir, 'BERT-Tiny', 'ir_model', 'span')

#{version:str, data:[{title:str, paragraphs:[{qas:[Question...], context:str}, ...]}, ...]}
#Question = {plausible_answers:Answers, question:str, id:str, answers:Answers,  is_impossible:bool}
#Answers = [{text:str, answer_start:int}, ...]




class BertForQA(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.qa_bert = BertForQuestionAnswering(config)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layernorm = nn.LayerNorm(384, self.config.layer_norm_eps)

    config_class = BertConfig
    base_model_prefix = 'bert'
    def forward(self, inputs, start_positions=None, end_positions=None):
        start_logits, end_logits = self.qa_bert(**inputs)
        start_probs, end_probs = self.softmax(start_logits), self.softmax(end_logits)
        if start_positions is not None:
            start_logits, end_logits = self.layernorm(start_probs), self.layernorm(end_probs)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss = nn.CrossEntropyLoss(ignore_index=ignored_index)
            total_loss = loss(start_logits, start_positions) + loss(end_logits, end_positions)
            return total_loss / 2
        else:
            return start_probs, end_probs

class BERTForCLS(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.cls_bert = BertForSequenceClassification(config)
        self.loss = nn.BCEWithLogitsLoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layernorm = nn.LayerNorm(self.config.num_labels, self.config.layer_norm_eps)

    config_class = BertConfig
    base_model_prefix = 'bert'
    def forward(self, inputs, cls_target=None):
        outputs = self.cls_bert(**inputs)
        logits = outputs[0]
        probs = self.softmax(logits)
        print(probs)
        print(self.layernorm(probs))
        if cls_target is not None:
            logits = self.layernorm(probs)
            return self.loss(logits[:, 1], cls_target)
        else:
            return probs[:, 1]


class BERTWithSketchyReader(nn.Module):
    def __init__(self, model_dir=SR_MODEL_DIR):
        super().__init__()
        self.efv = BERTForCLS.from_pretrained(model_dir)

    def save_pretrained(self, model_dir=SR_MODEL_DIR):
        self.efv.save_pretrained(model_dir)
    
    def forward(self, inputs, cls_target=None):
        outputs = self.efv(inputs, cls_target)
        return outputs


class BERTWithIntensiveReader(nn.Module):
    def __init__(self, qa_model_dir=IR_MODEL_DIR_SPAN, ifv_model_dir=IR_MODEL_DIR_CLS):
        super().__init__()
        self.baseline = BertForQA.from_pretrained(qa_model_dir)
        self.ifv = BERTForCLS.from_pretrained(ifv_model_dir)
        self.config = self.baseline.config

    def save_pretrained(self, span_dir=IR_MODEL_DIR_SPAN, cls_dir=IR_MODEL_DIR_CLS):
        self.baseline.save_pretrained(span_dir)
        self.ifv.save_pretrained(cls_dir)
        
    #**args, (batch), (batch), (batch)
    def forward(self, inputs, start_target=None, end_target=None, cls_target=None):     
        if all((x is not None for x in (start_target, end_target, cls_target))):
            span_loss = self.baseline(inputs, start_target, end_target)
            ifv_loss = self.ifv(inputs, cls_target)
            return ifv_loss, span_loss
        else:
            start_logits, end_logits = self.baseline(inputs)
            ifv_logits = self.ifv(inputs)
            return ifv_logits, start_logits, end_logits

class SQuAD2Data:
    def __init__(self):
        self.batch_size = 8
        self.sequence_length = 384
        self.hidden_size = 128
        
        self.null_score_diff_threshold = 0.0
        self.do_lower_case=True
        self.max_answer_length=30
        self.n_best_size=5
        self.verbose_logging=False
        self.version_2_with_negative = True
        

        self.alpha1, self.alpha2, self.beta1, self.beta2, self.lambda1, self.lambda2 = (.5 for _ in range(6))
        
        output_dir = os.path.join('..', '..', 'NLP-QA-output')
        prefix = 'epoch_1_3'
        self.output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
        self.output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))
        self.output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))

        self.device = torch.device('cpu')

        adam_epsilon=1e-08
        learning_rate=0.0003
        weight_decay=0.0

        self.tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_DIR, do_lower_case=self.do_lower_case)
        self.sr_model = BERTWithSketchyReader(SR_MODEL_DIR)
        self.sr_optimizer = AdamW(self.sr_model.parameters(), lr=learning_rate, eps=adam_epsilon, weight_decay=weight_decay)
        self.ir_model = BERTWithIntensiveReader(IR_MODEL_DIR_SPAN, IR_MODEL_DIR_CLS)
        self.ir_optimizer = AdamW(self.ir_model.parameters(), lr=learning_rate, eps=adam_epsilon, weight_decay=weight_decay)

        self.num_train_epochs=1.0
        self.max_steps=-1
        self.max_grad_norm=1.0
        self.logging_steps=200
        self.seed = 41

    def load(self, fn=EVAL_DATA_FN):
        features_and_dataset = torch.load(fn)
        self._features, self._dataset, self._examples = features_and_dataset["features"], features_and_dataset["dataset"], features_and_dataset["examples"]
        self.features, self.dataset, self.examples = self._features, self._dataset, self._examples
        self.feature_dict = defaultdict(list)
        for feat in self.features: self.feature_dict[feat.example_index].append(feat)
    
    
    def train_ir(self, fn=TRAIN_DATA_FN):
        self.load(fn)
        sampler = RandomSampler(self.dataset)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size)
        t_total = len(dataloader) // self.num_train_epochs
        
        scheduler = get_linear_schedule_with_warmup(self.ir_optimizer, num_warmup_steps=0, num_training_steps=t_total)
        
        tb_writer = SummaryWriter()
        global_step = 1
        tr_loss, logging_loss = 0.0, 0.0
    
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        train_iterator = trange(0, int(self.num_train_epochs), desc="Epoch", disable=False)

        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':batch[0], 'attention_mask':batch[1], 'token_type_ids':batch[2]}
                feature_indices = batch[3]
                #cls_index, p_mask = batch[4], batch[5]    
                unanswerable = torch.tensor([int(self.features[feature_index.item()].is_impossible) for feature_index in feature_indices], dtype=torch.float)
                start_indices = torch.tensor([int(self.features[feature_index.item()].start_position) for feature_index in feature_indices], dtype=torch.long)
                end_indices = torch.tensor([int(self.features[feature_index.item()].end_position) for feature_index in feature_indices], dtype=torch.long)
                
                self.ir_model.zero_grad()
                self.ir_model.train()
                ifv_loss, span_loss = self.ir_model(inputs, start_indices, end_indices, unanswerable)
                
                rv_loss = self.alpha1 * span_loss + self.alpha2 * ifv_loss                
                rv_loss.backward()
                tr_loss += rv_loss.item()

                torch.nn.utils.clip_grad_norm_(self.ir_model.parameters(), self.max_grad_norm)
                self.ir_optimizer.step()
                scheduler.step()
                global_step += 1
                self.ir_optimizer.zero_grad()

                # Log metrics
                if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.logging_steps, global_step)
                    self.logging_loss = tr_loss

                if self.max_steps > 0 and global_step > self.max_steps:
                    epoch_iterator.close()
                    break

            if self.max_steps > 0 and global_step > self.max_steps:
                train_iterator.close()
                break
        tb_writer.close()
        self.ir_model.save_pretrained()
        return global_step, tr_loss / global_step

    def train_sr(self, fn=TRAIN_DATA_FN):
        self.load(fn)
        sampler = RandomSampler(self.dataset)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size)
        
        t_total = len(dataloader) // self.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(self.sr_optimizer, num_warmup_steps=0, num_training_steps=t_total)
        
        tb_writer = SummaryWriter()
        global_step = 1
        tr_loss, logging_loss = 0.0, 0.0

        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        train_iterator = trange(0, int(self.num_train_epochs), desc="Epoch", disable=False)

        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':batch[0], 'attention_mask':batch[1], 'token_type_ids':batch[2], 'output_hidden_states':False}
                feature_indices = batch[3]
                #cls_index, p_mask = batch[4], batch[5]    
                unanswerable = torch.tensor([int(self.features[feature_index.item()].is_impossible) for feature_index in feature_indices], dtype=torch.float)
                
                self.sr_model.zero_grad()
                self.sr_model.train()
                efv_loss = self.sr_model(inputs, unanswerable)
                
                efv_loss.backward()
                
                tr_loss += efv_loss.item()

                torch.nn.utils.clip_grad_norm_(self.sr_model.parameters(), self.max_grad_norm)
                self.sr_optimizer.step()
                scheduler.step()  # Update learning rate schedule
                global_step += 1
                self.sr_optimizer.zero_grad()

                # Log metrics
                if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.logging_steps, global_step)
                    self.logging_loss = tr_loss

                if self.max_steps > 0 and global_step > self.max_steps:
                    epoch_iterator.close()
                    break

            if self.max_steps > 0 and global_step > self.max_steps:
                train_iterator.close()
                break
        tb_writer.close()
        self.sr_model.save_pretrained()

        return global_step, tr_loss / global_step
    
    def add_results(self, feature_indices, start_logits_, end_logits_, rv_logits_=None):
        for i, feature_index in enumerate(feature_indices):
            feature = self.features[feature_index.item()]
            unique_id = int(feature.unique_id)
            start_logits, end_logits = start_logits_[i], end_logits_[i]
            if rv_logits_ is not None:
                rv_noans = rv_logits_[i].item()
                """
                print(rv_noans)
                start_min, start_max = start_logits[1:].min().item(), start_logits[1:].max().item()
                end_min, end_max = end_logits[1:].min().item(), end_logits[1:].max().item()
                print(start_logits[0].item(), end_logits[0].item())
                print(start_max, end_max)
                print('-' *20)
                """"""
                weights = softmax(torch.cat((start_logits[1:].unsqueeze(dim=1), end_logits[1:].unsqueeze(dim=1)), dim=1))
                start_logits[1:] = minmax(start_logits[1:], start_min, start_max, start_min, ans_score * weights[:, 0])
                end_logits[1:] = minmax(end_logits[1:], end_min, end_max, end_min, ans_score * weights[:, 1])
                """
                start_logits[0] = self.lambda1 * (start_logits[0] + end_logits[0])
                end_logits[0] = self.lambda2 * rv_noans
                #start_logits = softmax(start_logits)
                #end_logits = softmax(end_logits)
            start_logits, end_logits = to_list(start_logits), to_list(end_logits) 
            result = SquadResult(unique_id, start_logits, end_logits)
            self.result_dict[result.unique_id] = result
            self.results.append(result)
    
    def evaluate(self, fn=EVAL_DATA_FN):
        self.load(fn)
        sampler = SequentialSampler(self.dataset)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size)        

        train_iterator = trange(0, int(self.num_train_epochs), desc="Epoch", disable=False)
        self.results = []
        self.result_dict = {}
        use_baseline_logits = False

        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':batch[0], 'attention_mask':batch[1], 'token_type_ids':batch[2]}
                feature_indices = batch[3]
                #cls_index, p_mask = batch[4], batch[5]
                self.sr_model.eval()
                self.ir_model.eval()
                with torch.no_grad():
                    if use_baseline_logits:   
                        start_logits, end_logits = self.ir_model.baseline(inputs)
                        rv_logits = None
                    else:
                        efv_logits = self.sr_model(inputs)
                        ifv_logits, start_logits, end_logits = self.ir_model(inputs)
                        rv_logits =  self.beta1 * efv_logits + self.beta2 * ifv_logits
                    self.add_results(feature_indices, start_logits, end_logits, rv_logits)

        self.predictions = compute_predictions_logits(
            self.examples,
            self.features,
            self.results,
            self.n_best_size,
            self.max_answer_length,
            self.do_lower_case,
            self.output_prediction_file,
            self.output_nbest_file,
            self.output_null_log_odds_file,
            self.verbose_logging,
            self.version_2_with_negative,
            self.null_score_diff_threshold,
            self.tokenizer)

        results = squad_evaluate(self.examples, self.predictions)
        print('After Retrospective Reader:')
        names = ''
        values = ''
        for name, v in results.items():
            names += name + '  '
            values += str(v) + '  '
        print(names)
        print(values)



    def slice_(self, end):
        self.features = self._features[:end]
        self.dataset = TensorDataset(*[t[:end] for t in self._dataset.tensors])
        self.examples = self._examples[:self.features[-1].example_index+1]
       

    def print_pred(self):
        predictions = []
        for example in self.examples:
            print(self.get_sample_str(example))
            pred = self.predictions[example.qas_id][0]
            print('Prediction: {}\n'.format(pred if pred else 'No answer'))
            predictions.append(pred)
        #results = squad_evaluate(self.examples, predictions)
        #print(results)
    
    def get_sample_str(self, example):
        sample_str = ['Question: ' + example.question_text]
        answers = [answer['text'] for answer in example.answers]
        if answers: sample_str.append('Answers: ' + ', '.join(set(answers)))
        else: sample_str.append('Answers: No answer')
        return '\n'.join(sample_str)

    def get_context(self, example):
        return example.context_text

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def softmax(tensor):
    exp_tensor = tensor.exp()
    tensor_sum = exp_tensor.sum(dim=-1, keepdim=True)
    return exp_tensor / tensor_sum

def minmax(x, vmin, vmax, ymin=0, ymax=1):
    return ymin + ((ymax - ymin) * (x - vmin)) / (vmax - vmin)

def main():

    data = SQuAD2Data()
    #data.slice_(64)
    #data.load()
    
    #step, total_loss = data.train_sr()
    #print('step: {}, total loss:{:.2f}'.format(step, total_loss))
    data.evaluate()
    """
    
    print([0 if x not in ['[CLS]', '[SEP]'] else 1 for x in data.features[0].tokens])
    print('input_ids\n', [int(bool(x)) for x in data.features[0].input_ids])
    print('attn_mask\n', [int(bool(x)) for x in data.features[0].attention_mask])
    print('token_type_ids\n', [int(bool(x)) for x in data.features[0].token_type_ids])
    print('p_mask\n', [int(bool(x)) for x in data.features[0].p_mask])
    """
    
    
    """
    #SquadExample
    qas_id, question_text, context_text, answers, doc_tokens, title, is_impossible, answer_text, start_position, end_position 
    #SquadFeatures
    qas_id, unique_id, example_index, input_ids, attention_mask, token_type_ids, tokens, token_to_orig_map, cls_index, p_mask, paragraph_len, token_is_max_context, 
    start_position, end_position, is_impossible 
    #SquadResult
    unique_id, start_logits, end_logits 
    """


if __name__ == '__main__':
    main()