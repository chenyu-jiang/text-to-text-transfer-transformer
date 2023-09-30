from random import shuffle
import t5
import t5.data.mixtures
from t5 import seqio

import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Load dataset and save as json')
parser.add_argument('--task-or-mixture', type=str, default="supervised_proportional", help='dataset or mixture to load')
parser.add_argument('--output-path', type=str, help='output directory')
parser.add_argument('--num-examples', type=int, default=1000000, help='number of examples to load')

args = parser.parse_args()

if args.output_path is None:
    args.output_path = "./" + args.task_or_mixture + ".jsonl"

dataset = seqio.get_mixture_or_task(args.task_or_mixture).get_dataset(
        sequence_length={'inputs':40960,'targets':40960}, # Extranous length to capture all data
        num_epochs=1,
        copy_pretokenized=True,
        shuffle=True
)
with open(args.output_path, "w") as f:
    for i, example in tqdm(enumerate(dataset.as_numpy_iterator()), total=args.num_examples):
        raw_input_sequence = example['inputs_pretokenized'].decode()
        raw_target_sequence = example['targets_pretokenized'].decode()
        json_string = json.dumps(
            {"inputs": raw_input_sequence,
             "targets": raw_target_sequence,
             "input_seq_len": len(example['inputs']),
             "target_seq_len": len(example['targets'])})
        f.write(json_string)
        f.write("\n")
        if i >= args.num_examples - 1:
            break
