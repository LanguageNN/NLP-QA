import torch
import torch.nn as nn
import json
import os.path
import itertools
from collections import defaultdict
from nltk.tokenize import word_tokenize

NUM_LAYERS = 4
TRAIN_FN = os.path.join(os.path.curdir, 'SQuAD2', 'train-v2.0.json')
DEV_FN = os.path.join(os.path.curdir, 'SQuAD2', 'dev-v2.0.json')

#{version:str, data:[{title:str, paragraphs:[{qas:[Question...], context:str}, ...]}, ...]}
#Question = {plausible_answers:Answers, question:str, id:str, answers:Answers,  is_impossible:bool}
#Answers = [{text:str, answer_start:int}, ...]



class Vocab:
    def __init__(self, special_tokens=None):
        self.w2idx = {}
        self.idx2w = {}
        self.w2cnt = defaultdict(int)
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            self.add_tokens(special_tokens)

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
            self.w2cnt[token] += 1

    def add_token(self, token):
        if token not in self.w2idx:
            cur_len = len(self)
            self.w2idx[token] = cur_len
            self.idx2w[cur_len] = token

    def prune(self, min_cnt=2):
        to_remove = set([token for token in self.w2idx if self.w2cnt[token] < min_cnt])
        to_remove ^= set(self.special_tokens)

        for token in to_remove:
            self.w2cnt.pop(token)

        self.w2idx = {token: idx for idx, token in enumerate(self.w2cnt.keys())}
        self.idx2w = {idx: token for token, idx in self.w2idx.items()}

    def __contains__(self, item):
        return item in self.w2idx

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.w2idx[item]
        elif isinstance(item , int):
            return self.idx2w[item]
        else:
            raise TypeError("Supported indices are int and str")

    def __len__(self):
        return(len(self.w2idx))

class BERTInput:
    def __init__(self, data):
        """
        data: a sequence of samples
        """
        self.src = []
        self.tgt = [] 
        self.vocab = data.vocab
        for s in data.as_tuple():
            self.src.append((s[0], s[1]))
            answers = []
            for a in s[2]:
                answers.append((s[0], [ai * ci for ai, ci in zip(a, s[1])]))
            self.tgt.append(answers)

        self.cls = self.vocab['<CLS>']
        self.sep = self.vocab['<SEP>']
        self.pad = self.vocab['<PAD>']
        self.num_segments = len(self.src[0]) # should be 2
        self.max_lengths = []
        for i in range(self.num_segments):
            self.max_lengths.append(max((len(x[i]) for x in self.src)))
        self.sample_length = sum(self.max_lengths) + self.num_segments + 1 # sum(tokens) + <SEG> * num_seg + <CLS>  
        self.positions = tuple([p for p in range(self.sample_length)])
        self.segments = tuple([0] + [seg for _ in range(self.max_lengths[i] + 1) for seg in range(self.num_segments)])

    def pad_seg(self, tokens, seg):
        padded = []
        for t in range(self.max_lengths[seg]):
            if t < len(tokens[seg]): padded.append(tokens[seg][t])
            else: padded.append(self.pad)
        return padded
    
    def get_answers(self, index):
        targets = []
        for tgt in self.tgt[index]:
            target = [self.cls]
            for seg in range(self.num_segments):
                target.extend(self.pad_seg(tgt, seg))
                target.append(self.sep)
            targets.append(target)
        return targets

    def get_answers_str(self, index):
        targets = self.get_answers(index)
        texts = [' '.join([self.vocab[x] for x in tgt if x != self.vocab['<PAD>']]) for tgt in targets]
        return texts

    def get_input_str(self, index):
        source = self[index]
        text = ' '.join([self.vocab[x] for x in source if x != self.vocab['<PAD>']])
        return text

    def __getitem__(self, index):
        source = [self.cls]
        for seg in range(self.num_segments):
            source.extend(self.pad_seg(self.src[index], seg))
            source.append(self.sep)
        return source
    
    def __len__(self):
        return len(self.src)


class SQuADataset:
    def __init__(self, json_fn, lowercase=True, num_articles=None):
        with open(json_fn) as fp:
            data = json.load(fp)
        self.data = data['data']
        self.lowercase = lowercase

        if num_articles is None: self.num_articles = len(self.data)
        else: self.num_articles = num_articles

        self.qas = []
        for i in range(self.num_articles):
            for paragraph in self.data[i]['paragraphs']:
                qa = {}
                qa['ids'], qa['questions'], qa['answers'] = [], [], []
                qa['context'] = paragraph['context']                                                                       # str
                for q in paragraph['qas']:
                    qa['ids'].append(q['id'])                                                                              # List[str]
                    qa['questions'].append(q['question'])                                                                  # List[str]
                    qa['answers'].append([(a['answer_start'], a['answer_start'] + len(a['text'])) for a in q['answers']])  # List[List[Tuple[int, int]]]
                self.qas.append(qa)
        self.preprocess()

        self.vocab = Vocab(['<PAD>', '<UNK>', '<SEP>', '<CLS>'])
        self.vocab.add_tokens(itertools.chain.from_iterable(itertools.chain.from_iterable((qa['questions'] for qa in self.qas))))
        self.vocab.add_tokens(itertools.chain.from_iterable((qa['context'] for qa in self.qas)))

    def tokenize(self, text):
        if self.lowercase: text = text.lower()
        return word_tokenize(text)
    
    def preprocess(self):
        for qa in self.qas:
            spans = list(itertools.chain.from_iterable(qa['answers']))
            convert_ans, qa['context'] = self.preprocess_context(qa['context'], spans)
            for k in range(len(qa['questions'])):
                qa['answers'][k] = [(convert_ans[s1], convert_ans[s2]) for s1, s2 in qa['answers'][k]]
                qa['questions'][k] = self.tokenize(qa['questions'][k])

    def preprocess_context(self, text, spans):
        si = sorted(set(itertools.chain.from_iterable(spans)) | {0, len(text)})
        tokens = [self.tokenize(text[si[i-1]:si[i]]) for i in range(1, len(si))]
        new_si = [0] + [n for n in itertools.accumulate(map(len, tokens))]
        convert_si = {k:v for k,v in zip(si, new_si)}
        tokens = [tok for tok in itertools.chain.from_iterable(tokens)]
        return convert_si, tokens

    def as_tuple(self):
        samples = []
        for qa in self.qas:
            context = []
            for tok in qa['context']:
                if tok in self.vocab: context.append(self.vocab[tok])
                else: context.append(self.vocab['<UNK>'])
            for q in range(len(qa['questions'])):
                answers = [[i >= s1 and i < s2 for i in range(len(qa['context']))] for s1, s2 in qa['answers'][q]]
                if not answers: answers = [[0 for _ in range(len(context))]]
                question = []
                for tok in qa['questions'][q]:
                    if tok in self.vocab: question.append(self.vocab[tok])
                    else: question.append(self.vocab['<UNK>'])
                samples.append((question, context, answers))
        return tuple(samples)

class BERTInputEmbedding(nn.Module):
    def __init__(self, embed_size=128):
        self.input_dataset = SQuADataset(DEV_FN)
        self.input = BERTInput(self.input_dataset)
        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(self.input.sample_length, embed_size)
        self.segment_embedding = nn.Embedding(self.input.num_segments, embed_size)
        self.token_embedding = nn.Embedding(len(input_dataset.vocab), embed_size)
            

    def forward(self, tokens):
        pos = torch.tensor(self.input.get_positions(), dtype=torch.long, device=self.device)
        seg = torch.tensor(self.input.get_segments(), dtype=torch.long, device=self.device)
        tok = torch.tensor(tokens, dtype=torch.long, device=self.device)
        return self.position_embedding(pos) + self.segment_embedding(seg) + self.token_embedding(tok)

dataset = SQuADataset(DEV_FN)
input_ = BERTInput(dataset)


print('Answerable:')
sample = 3
question = input_.get_input_str(sample)
answers = input_.get_answers_str(sample)
print('input\n {}\n'.format(question))
print('answers\n {}\n'.format(answers))


print('Unanswerable:')
sample = 5
question = input_.get_input_str(sample)
answers = input_.get_answers_str(sample)
print('input\n {}\n'.format(question))
print('answers\n {}\n'.format(answers))
