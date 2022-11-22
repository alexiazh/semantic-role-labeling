import argparse
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
# import nltk.data
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


def format_data():
    test_path = "dataset/test.wp_target"

    with open(test_path, 'r') as f:
        stories = [" ".join(l.replace("<newline>","").split()) for l in f.readlines()]

    with open(test_path + "_formatted", 'w') as o:
        # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        for story in stories:
            # o.write("\n".join(sent_detector.tokenize(story)))
            o.write("\n".join(sent_tokenize(story)))


def load_data():
    test_path = "dataset/test.wp_target_formatted"
    with open(test_path, 'r') as f:
        test_data = [l.strip() for l in f.readlines()]
    return test_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="dataset/test.wp_target_formatted", type=Path, 
                        help="Path to input file.")
    parser.add_argument("--output-file", default="outputs/test.wp_target", type=Path, 
                        help="Path to output file.")
    parser.add_argument('--print', type=bool, default=False,
                        help="Will print to console if set to True")
    args = parser.parse_args()

    if not args.input_file.is_file():
        parser.error("You need to provide a valid input file.")

    return args


class SemanticRoleLabeller():

    def __init__(self, input, output):
        # with open(input, 'r') as f:
        #     self.data = f.readlines()
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    def predict_sentence(self, sent):
        preds = self.predictor.predict(sent)
        preds["pos"] = pos_tag(preds["words"])
        # verb_dict = {}
        # for pred in preds:
        #     verb_dict[pred["verb"]] = [(word, tag) for word, tag in zip(preds["words"], pred["tags"])]
        return preds

    def _predictions_to_labeled_instances(self, preds):
        
        pos_tags = preds["pos"]
        instances = []

        for pred in preds["verbs"]:
            pred_tags = pred["tags"]
            arg_dict = {}
            i = 0
            while i < len(pred_tags):
                tag = pred_tags[i]
                if tag[0] == "B":
                    curr_arg = tag[2:]
                    begin_idx = i
                    i += 1
                    tag = pred_tags[i]
                    while tag[0] == "I":
                        i += 1
                        tag = pred_tags[i]
                    end_idx = i
                    arg_dict[curr_arg] = [pos_tags[i] for i in range (begin_idx,end_idx)]
                    # arg_dict[curr_arg] = " ".join(words[begin_idx:end_idx])
                    # arg_lst.append(tuple((curr_arg, " ".join(words[begin_idx:end_idx]))))
                else:
                    i += 1
            instances.append(arg_dict)

        return instances


    def _get_main_word(self, tagged_lst):
        # for tok in reversed(tagged_lst):
        for tok in tagged_lst:
            if tok[1][:2] == "NN" or tok[1] == "PRP":
                return tok[0]
        return None


    def instances_to_tagged_trigrams(self, instances):
        trigrams = []
        for instance in instances:
            if len(instance) <= 1:
                continue
            v_args = []
            arg_before_v = False
            for tag, tokens in instance.items():
                if len(v_args) < 3:
                    if tag == "V":
                        if not arg_before_v: 
                            v_args.append('')
                        v_args.append(tokens[0][0])
                    if tag[:3] == "ARG":
                        for i in range(4):
                            if tag == f"ARG{i}":
                                main_word = self._get_main_word(tokens)
                                if main_word is not None:
                                    v_args.append(main_word)
                                    arg_before_v = True
                        if tag[3] == 'M':
                            main_word = self._get_main_word(tokens)
                            if main_word is not None:
                                v_args.append(main_word)
                                arg_before_v = True
            if len(v_args) == 2:
                v_args.append('')
            trigrams.append(tuple(v_args))
        return trigrams

    
    def predict_sentences(self, sents) -> list:
        preds = []
        for sent in sents:
            print(sent)
            pred = self.predict_sentence(sent)
            instances = self._predictions_to_labeled_instances(pred)
            trigrams = self.instances_to_tagged_trigrams(instances)
            print(trigrams)
            preds.append(trigrams)
        return preds


def main():
    args = parse_args()
    srl = SemanticRoleLabeller(args.input_file, args.output_file)
    # sent = "The keys, which were needed to access the building, were locked in the car."
    # print(sent)
    # pred = srl.predict_sentence(sent)
    # instances = srl._predictions_to_labeled_instances(pred)
    # tuple_lst = srl.instances_to_tagged_trigrams(instances)
    # print(tuple_lst)

    test_data = load_data()[:10]
    preds = srl.predict_sentences(test_data)


if __name__ == '__main__':
  main()