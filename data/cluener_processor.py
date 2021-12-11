import json
from .vocabulary import Vocabulary
import jieba
import jieba.posseg as pseg
import paddle
paddle.enable_static()

seg_type = {
    'nz':  0,
    'n':   1,
    'nr':  2,
    'm':   3,
    'i':   4,
    'l':   5,
    'd':   6,
    's':   7,
    't':   8,
    'mq':  9,
    'j':   10,
    'a':   11,
    'r':   12,
    'b':   13,
    'f':   14,
    'nrt': 15,
    'v':   16,
    'z':   17,
    'ns':  18,
    'q':   19,
    'vn':  20,
    'c':   21,
    'nt':  22,
    'u':   23,
    'o':   24,
    'zg':  25,
    'nrfg':    26,
    'df':  27,
    'p':   28,
    'g':   29,
    'y':   30,
    'ad':  31,
    'vg':  32,
    'ng':  33,
    'x':   34,
    'ul':  35,
    'k':   36,
    'ag':  37,
    'dg':  38,
    'rr':  39,
    'rg':  40,
    'an':  41,
    'vq':  42,
    'e':   43,
    'uv':  44,
    'tg':  45,
    'mg':  46,
    'ud':  47,
    'vi':  48,
    'vd':  49,
    'uj':  50,
    'uz':  51,
    'h':   52,
    'ug':  53,
    'rz':  54,
    'eng': 55,
    'yg': 56,
    # 'PER': 24, # 人名 
    # 'LOC': 25, # 地名 
    # 'ORG': 26, # 机构名 
    # 'TIME': 27, # 时间
}
class CluenerProcessor:
    """Processor for the chinese ner data set."""
    def __init__(self,data_dir):
        self.vocab = Vocabulary()
        self.data_dir = data_dir

    def build_vocab(self):
        vocab_path = self.data_dir / 'vocab.pkl'
        if vocab_path.exists():
            self.vocab.load_from_file(str(vocab_path))
        else:
            files = ["train.json", "dev.json", "test.json"]
            for file in files:
                with open(str(self.data_dir / file), 'r') as fr:
                    for line in fr:
                        line = json.loads(line.strip())
                        text = line['text']
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "train.json"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "dev.json"), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "test.json"), "test")

    def _create_examples(self,input_path,mode):
        examples = []
        with open(input_path, 'r') as f:
            idx = 0
            for line in f:
                json_d = {}
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                seg_list = pseg.lcut(text, use_paddle=True)
                group_list = []
                for k, v in seg_list:
                    group_list.extend([seg_type[v]] * len(k))
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = words
                json_d['tag'] = labels
                json_d['raw_context'] = "".join(words)
                json_d['group'] = group_list
                idx += 1
                examples.append(json_d)
        return examples


