import re
import argparse
import sys
import numpy as np
import pandas as pd


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


parser = argparse.ArgumentParser(description="Log Process")
parser.add_argument('--log_file', type=str, default='logs\TESTING VDCNN_vdcnn49-2018-12-06 19-29-24.log')

if is_interactive():
    params = []
else:
    params = sys.argv[1:]
args = vars(parser.parse_args(params))

pattern_matrics = '\d+\.\d+'
pattern_epoch = '\d+\.ckpt'

test_info = []
all_info = []
with open(args.get('log_file'), 'r') as f:
    for line in f:
        result_epoch = re.findall(pattern_epoch, line)
        result_matrics = re.findall(pattern_matrics, line)

        if result_epoch != []:
            test_info.append(int(result_epoch[0].split('.')[0]))
        elif result_matrics != []:
            test_info.extend([float(v) for v in result_matrics])
            all_info.append(np.array(test_info))
            test_info = []
info_df = pd.DataFrame(data=all_info,columns=['epoch','test_loss','accuracy'])
info_df = info_df.set_index('epoch')
info_df = info_df.sort_index()
save_name = args.get('log_file').split('.')[0]
info_df.to_csv(save_name+'.csv')
