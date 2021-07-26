# """
# Replace text in python files.
# """
# import glob
# import os.path as osp
# import argparse
# import fileinput
#
# EXTENSION = '.py'
#
#
# def is_python_file(filename):
#     ext = osp.splitext(filename)[1]
#     return ext == EXTENSION
#
#
# def update_file(filename, text_to_search, replacement_text):
#     print('Processing {}'.format(filename))
#     with fileinput.FileInput(filename, inplace=True, backup='') as file:
#         for line in file:
#             print(line.replace(text_to_search, replacement_text), end='')
#
#
# def recursive_update(directory, text_to_search, replacement_text):
#     filenames = glob.glob(osp.join(directory, '*'))
#
#     for filename in filenames:
#         if osp.isfile(filename):
#             if not is_python_file(filename):
#                 continue
#             update_file(filename, text_to_search, replacement_text)
#         elif osp.isdir(filename):
#             recursive_update(filename, text_to_search, replacement_text)
#         else:
#             raise NotImplementedError
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         'file_or_dir', type=str, help='path to file or directory'
#     )
#     parser.add_argument('text_to_search', type=str, help='name to be replaced')
#     parser.add_argument('replacement_text', type=str, help='new name')
#     parser.add_argument(
#         '--ext', type=str, default='.py', help='file extension'
#     )
#     args = parser.parse_args()
#
#     file_or_dir = args.file_or_dir
#     text_to_search = args.text_to_search
#     replacement_text = args.replacement_text
#     extension = args.ext
#
#     global EXTENSION
#     EXTENSION = extension
#
#     if osp.isfile(file_or_dir):
#         if not is_python_file(file_or_dir):
#             return
#         update_file(file_or_dir, text_to_search, replacement_text)
#     elif osp.isdir(file_or_dir):
#         recursive_update(file_or_dir, text_to_search, replacement_text)
#     else:
#         raise NotImplementedError
#
#
# if __name__ == '__main__':
#     main()
from torch.utils.data import TensorDataset,ConcatDataset, DataLoader,Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler, WeightedRandomSampler

import torch
class RandomDomainSampler(Sampler):
    """Random domain sampler.

    This sampler randomly samples N domains each with K
    images to form a minibatch.
    """
    def __init__(self, data_source, single_batch_size = -1,fix_batch_sie=128):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())
        self.n_domain = len(self.domains)

        # Make sure each domain has equal number of images

        if single_batch_size == -1:
            single_batch_size = fix_batch_sie//self.n_domain
        self.batch_per_domain = single_batch_size
        self.batch_size = fix_batch_sie
        # n_domain denotes number of domains sampled in a minibatch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.batch_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.batch_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length
class CusConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        output = []
        print("index i : ",i)
        for d in self.datasets:
            # print("current dataset : ",d)
            cycle = d[i %len(d)]
            # print("cycle : ",cycle)
            output.append(cycle)
        print("output : ",output)
        return tuple(output)

        # return tuple(d[i %len(d)] for d in self.datasets)

    def __len__(self):
        # return max(len(d) for d in self.datasets)
        return min(len(d) for d in self.datasets)

datasets = []
n = 0
for i in range(3):
    tensor = torch.arange(i*10, (i+1)*10+n)
    dataset = TensorDataset(tensor)
    datasets.append(dataset)
    print(tensor)
    # print("dataset element : ",dataset[0])
    n+=2
import pytorch_lightning as pl
pl.seed_everything(42)
final_dataset = CusConcatDataset(datasets)
sampler = RandomSampler(final_dataset)
s_iter = iter(sampler)
print("next : ",next(s_iter))
print("next : ",next(s_iter))


# print(final_dataset[0])
print("sampler iter: ",iter(sampler))
loader = DataLoader(
    final_dataset,
    sampler=sampler,
    # shuffle=True,
    num_workers=0,
    batch_size=4,
    drop_last=True

)

loader_size = len(loader)
print("loader size : ",loader_size)
idx = 0
for data in loader:

    print("batch - - ")
    print("idx : ",idx)
    print(data)
    print(len(data))
    print(len(data[0][0]))
    idx+=1

print("concat datasets")
final_dataset = ConcatDataset(datasets)

# print(final_dataset[0])
loader = DataLoader(
    final_dataset,
    shuffle=True,
    num_workers=0,
    batch_size=4,
    drop_last=True
)

loader_size = len(loader)
print("loader size : ",loader_size)
idx = 0
for data in loader:

    print("batch - - ")
    print("idx : ",idx)
    print(data)

    idx+=1

# test_list = [2,4,8,1,3]
# print(test_list[[1,0,2]])