import numpy as np
import random

from cpp_base import CPPBase
import cf_c

class Dataset(CPPBase):
    def __init__(self):
        super().__init__()


class ClickDataset(Dataset):
    def __init__(self, file_path, separator=' ', config=None):
        super().__init__()
        self.c_class = cf_c.modules.datasets.ClickDataset

        self.file_path = file_path
        self.user_items_dic = {}
        self.user_item_ids = []
        self.num_negs = config.num_negs
        self.en_his = config.en_his
        self.max_his = config.max_his
        self.his_items = None
        self.masks = None

        self.user_ids_dic = {}
        self.item_ids_dic = {}
        self.num_users = 0
        self.num_items = 0

        with open(file_path, mode='r') as in_file:
            lines = in_file.readlines()
            self.his_items = np.zeros((len(lines), self.max_his), dtype=np.uint64)
            self.masks = np.zeros((len(lines), 1), dtype=np.uint64)

            for line in lines:
                splits = line.strip().split(separator)
                user_id = int(splits[0])
                items = splits[1:]
                items = list(map(int, items))

                if user_id not in self.user_ids_dic:
                    self.user_ids_dic[user_id] = user_id

                # self.user_items_dic[user_id] = set(items)
                self.user_items_dic[user_id] = items

                if len(items) >= self.max_his:
                    user_his = random.sample(items, self.max_his)
                    self.his_items[user_id] = user_his
                    self.masks[user_id] = [self.max_his]
                elif len(items) > 0:
                    user_his = items.copy()
                    user_his.extend([user_his[-1]] * (self.max_his - len(items)))
                    self.his_items[user_id] = user_his
                    self.masks[user_id] = [len(items)]
                else:
                    user_his = []
                    user_his.extend([0] * self.max_his)
                    self.his_items[user_id] = user_his
                    self.masks[user_id] = [0]
                    print(f"Warning {user_id} has 0 items !!! ")

                for item in items:
                    if item not in self.item_ids_dic:
                        self.item_ids_dic[item] = item
                                
                    self.user_item_ids.append([user_id, item])

        # self.item_corpus = set(self.item_ids_dic.keys())
        self.gen_dataset_info()

        if 'train' in file_path:
            print('update config, init c_instance !!! ')
            config.num_users = self.num_users
            config.num_items = self.num_items
            config.train_size = len(self.user_item_ids)

            self.click_dataset = np.array(self.user_item_ids, dtype=np.uint64)
            self.init_c_instance(click_dataset=self.click_dataset, historical_items=self.his_items, masks=self.masks)
            self.c_instance.max_his = self.max_his

        print('\n')


    def gen_dataset_info(self):
        print(f'gen dataset info of {self.file_path} ')
        self.num_users = len(self.user_ids_dic)
        self.num_items = len(self.item_ids_dic)
        user_ids = list(self.user_ids_dic.keys())
        item_ids = list(self.item_ids_dic.keys())
        # user_ids.sort()
        # item_ids.sort()
        max_user_id = max(user_ids)
        min_user_id = min(user_ids)
        max_item_id = max(item_ids)
        min_item_id = min(item_ids)
        if max_user_id - min_user_id + 1 != self.num_users:
            print('Warning user_id is not continuous! ')
        
        if max_item_id - min_item_id + 1 != self.num_items:
            print('Warning item_id is not continuous! ')

        print(f'number of users: {self.num_users}; min_user_id: {min_user_id}; max_user_id: {max_user_id}')
        print(f'number of items: {self.num_items}; min_item_id: {min_item_id}; max_item_id: {max_item_id}')
        print(f'total samples: {len(self.user_item_ids)} ')
        # print('\n')


if __name__ == "__main__":
    pass
