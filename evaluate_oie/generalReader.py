# -*- coding: utf-8 -*-


from evaluate_oie.oieReader import OieReader
from evaluate_oie.extraction import Extraction
import traceback
import logging

logger = logging.getLogger('root')


class GeneralReader(OieReader):

    def __init__(self):
        self.name = 'General'

    def read(self, fn, path_error=None):
        d = {}
        file_error = open(path_error, 'a', encoding='utf-8')
        with open(fn) as fin:
            for line in fin:
                try:
                    data = line.strip('\n').split('\t')
                    if len(data) > 0:
                        text = data[0]
                    if len(data) > 1:
                        confidence = data[1]
                    if len(data) > 2:
                        rel = data[2]
                    if len(data) > 3:
                        arg1 = data[3]
                    if len(data) > 4:
                        arg_else = data[4:]

                    curExtraction = Extraction(pred=rel, head_pred_index=-1, sent=text, confidence=float(confidence))
                    curExtraction.addArg(arg1)
                    for arg in arg_else:
                        curExtraction.addArg(arg)
                    d[text] = d.get(text, []) + [curExtraction]
                except:
                    file_error.write('====================')
                    file_error.write('file: {}'.format(fn))
                    file_error.write('line: {}'.format(line))
                    file_error.write('data: {}'.format(data))
                    file_error.write('====================')
                    file_error.write(str(traceback.format_exc()))
                    continue
        file_error.close()
        self.oie = d


if __name__ == "__main__":
    fn = "../data/other_systems/openie4_test.txt"
    reader = GeneralReader()
    reader.read(fn)
    for key in reader.oie:
        print(key)
        print(reader.oie[key][0].pred)
        print(reader.oie[key][0].args)
        print(reader.oie[key][0].confidence)
