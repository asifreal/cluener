import json
from .vocabulary import Vocabulary

class ClassifierProcessor:
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
                labels = ["O"] * len(words)
                exists = ["O"] * len(words) # only I or O
                entities = []
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                entities.append([key, sub_name, start_index, end_index])
                                if start_index == end_index:
                                    labels[start_index] = 'I-' + key
                                    exists[start_index] = 'I'
                                else:
                                    labels[start_index:end_index + 1] = ['I-' + key] * len(sub_name) 
                                    exists[start_index:end_index + 1] = ['I'] * len(sub_name) 
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = words
                json_d['tag'] = labels
                json_d['exists'] = exists
                json_d['entities'] = entities
                json_d['raw_context'] = "".join(words)
                idx += 1
                examples.append(json_d)
        return examples


