import jsonlines

filtered_objs = []
with jsonlines.open("./supervised_proportional.jsonl") as reader:
    for obj in reader:
        input_seqlen = obj["input_seq_len"]
        target_seqlen = obj["target_seq_len"]
        # the translation datasets can contain errorneous data
        # characterized by very long target sequences and 
        # very short input sequences. We filter these out.
        if target_seqlen > 2000 and input_seqlen < 1000:
            continue
        if target_seqlen / input_seqlen > 4 and target_seqlen > 200:
            continue
        filtered_objs.append(obj)

with jsonlines.open("./cleaned_supervised_proportional.jsonl", "w") as writer:
    for obj in filtered_objs:
        writer.write(obj)