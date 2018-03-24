from dataset import DataEntry, DataSet, Vocab, Action
import json
from io_utils import deserialize_from_file

def write_to_json_file(path,data):
    l = []
    g = data.grammar
    v = data.terminal_vocab
    for i in range(len(data.examples)):
        t = data.examples[i].parse_tree
        q = data.examples[i].query
        print(i,q)
        d = t.to_dict(q,g,v)
        l.append(d)
    print(l[0])
    with open(path,'w') as f:
        json.dump(l,f, encoding='latin1')

def main():
    flag = "django"
    # flag = "hs"
    train_data, dev_data, test_data = deserialize_from_file("../../"+flag+".cleaned.dataset.freq5.par_info.refact.space_only.bin")

    #uncomment below for Hearthstone data set
    #train_data, dev_data, test_data = deserialize_from_file("hs.freq3.pre_suf.unary_closure.bin")

    print("----- TRAIN -----")
    write_to_json_file("../data/"+flag+"_dataset/"+flag+".train.json", train_data)


    print("----- DEV -----")
    write_to_json_file("../data/"+flag+"_dataset/"+flag+".dev.json", dev_data)

    print("----- TEST -----")
    write_to_json_file("../data/"+flag+"_dataset/"+flag+".test.json", test_data)

    #print(train_data.get_examples(0).query)
    #print(train_data.get_examples(0).parse_tree)

if __name__ == '__main__':
	main()
