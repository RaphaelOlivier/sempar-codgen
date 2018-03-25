import grammar
import io_tools

class TargetDataset:
    def __init__(self,mode):
        self.mode=mode
        if mode == "django":
            print("Using Django dataset to load the target trees and indexers...")
            target_train_path = "../../data/django_dataset/django.train.json"
            target_dev_path = "../../data/django_dataset/django.dev.json"
            target_test_path = "../../data/django_dataset/django.test.json"
        else:
            print("Using HS dataset to load the target trees and indexers...")
            target_train_path = "../../data/hs_dataset/hs.train.json"
            target_dev_path = "../../data/hs_dataset/hs.dev.json"
            target_test_path = "../../data/hs_dataset/hs.test.json"

        self.indexer = grammar.Indexer(self.mode)
        self.target_train_dataset = io_tools.json_to_tree_dataset(target_train_path,self.indexer)
        self.target_dev_dataset = io_tools.json_to_tree_dataset(target_dev_path,self.indexer)
        self.target_test_dataset = io_tools.json_to_tree_dataset(target_test_path,self.indexer)

    def export(self,tree_list, suffix):
        path = "../../data/exp/results/test_"+self.mode+"_" + suffix + ".json"
        json_list = io_tools.tree_to_json_dataset(path,self.indexer,tree_list)
