import torch
import re
import argparse
import numpy as np
import onnxruntime as ort
from vncorenlp import VnCoreNLP
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE


class PhoBertSentiment():
    def __init__(self, model_path):
        self.args = argparse.Namespace()
        self.args.dict_path = "dict.txt"
        self.args.config_path = "config.json"
        self.max_sequence_length = 256
        self.args.bpe_codes = "bpe.codes"

        self.rdrsegmenter = VnCoreNLP("D:/sentiment_fpt/train_model/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

        self.vocab = Dictionary()
        self.vocab.add_from_file(self.args.dict_path)
        self.bpe = fastBPE(self.args)

        self.ort_session = ort.InferenceSession(model_path)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def convert_lines(self, inputs, vocab, bpe, max_sequence_length):
        outputs = np.zeros((len(inputs), max_sequence_length))

        eos_id = 2
        pad_id = 1

        for index, text in enumerate(inputs):
            subwords = bpe.encode('<s> ' + text + ' </s>')
            input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
            if len(input_ids) > self.max_sequence_length:
                input_ids = input_ids[:self.max_sequence_length]
                input_ids[-1] = eos_id
            else:
                input_ids = input_ids + [pad_id, ] * (self.max_sequence_length - len(input_ids))

            outputs[index, :] = np.array(input_ids)

        return outputs


    def process_data(self, texts):
        new_texts = []
        for text in texts:
            text = text.lower()

            tmp = ""
            try:
                tmp += " ".join(self.rdrsegmenter.tokenize(text)[0]) + " "
            except:
                tmp += text + " "

            new_texts.append(tmp[:-1])

        return new_texts


    def predict(self, text):
        try:
            text = self.process_data([text])
            X_test = self.convert_lines(text, self.vocab, self.bpe, self.max_sequence_length)

            preds_fold = []

            test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

            pbar = enumerate(test_loader)
            for i, (x_batch,) in pbar:
                x_batch = x_batch.cpu().detach().numpy()

                outputs = self.ort_session.run(
                    None,
                    {"input_ids": x_batch},
                )
                preds_fold = self.sigmoid(outputs[0][0])
                labels = (preds_fold > 0.5).astype(np.int)

            return 'negative' if labels[0] == 1 else 'neutral', preds_fold[0] if preds_fold[0] > 0.5 else 1 - preds_fold[0]
        except:
            return 'neutral', 0


if __name__ == "__main__":
    sentiment_model = PhoBertSentiment('phoBert_sentiment.onnx')
    label, prob = sentiment_model.predict("huấn luyện nhân viên kỹ hơn một chút nữa để nhân viên có những cô rất tốt có cô rất")
    print (label, prob)