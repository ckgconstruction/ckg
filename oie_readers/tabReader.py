""" Usage:
    tabReader --in=INPUT_FILE

Read a tab-formatted file.
Each line consists of:
sent, prob, pred, arg1, arg2, ...

"""

from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction
from docopt import docopt
import traceback
import sys
import logging

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('root')


class TabReader(OieReader):

    def __init__(self):
        self.name = 'TabReader'

    def read(self, fn, path_error=None):
        """
        Read a tabbed format line
        Each line consists of:
        sent, prob, pred, arg1, arg2, ...
        """
        d = {}
        ex_index = 0
        file_error = open(path_error, 'a', encoding='utf-8')
        with open(fn) as fin:
            for line in fin:
                if not line.strip():  # 该行为空时，跳过
                # if not line.strip('\n'):  # 该行为空时，跳过
                    continue
                data = line.strip().split('\t')
                # data = line.strip('\n').split('\t')
                try:
                    text, confidence, rel = data[:3]
                except Exception as e:  # 抽取的结果不对的
                    # print('====================')
                    # print('file: {}'.format(fn))
                    # print('line: {}'.format(line))
                    # print('data: {}'.format(data))
                    # print('====================')
                    # traceback.print_exc()
                    file_error.write('====================')
                    file_error.write('file: {}'.format(fn))
                    file_error.write('line: {}'.format(line))
                    file_error.write('data: {}'.format(data))
                    file_error.write('====================')
                    file_error.write(str(traceback.format_exc()))
                    continue
                    # sys.exit()
                curExtraction = Extraction(pred=rel,
                                           head_pred_index=None,
                                           sent=text,
                                           confidence=float(confidence),
                                           question_dist="./question_distributions/dist_wh_sbj_obj1.json",
                                           index=ex_index)
                ex_index += 1

                for arg in data[3:]:
                    curExtraction.addArg(arg)

                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d


if __name__ == "__main__":
    args = docopt(__doc__)
    input_fn = args["--in"]
    tr = TabReader()
    tr.read(input_fn)
