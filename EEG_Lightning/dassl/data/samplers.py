import copy
import random
# import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch


class RandomDomainSampler(Sampler):
    """Random domain sampler.

    This sampler randomly samples N domains each with K
    images to form a minibatch.
    """

    def __init__(self, data_source, single_batch_size=-1, fix_batch_sie=128):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)

        self.domains = sorted(list(self.domain_dict.keys()))
        self.n_domain = len(self.domains)

        # Make sure each domain has equal number of images

        if single_batch_size == -1:
            single_batch_size = fix_batch_sie // self.n_domain
        self.batch_per_domain = single_batch_size
        self.batch_size = fix_batch_sie
        # n_domain denotes number of domains sampled in a minibatch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            # selected_domains = random.sample(self.domains, self.n_domain)
            selected_domains = self.domains
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
class GroupLabelSampler(Sampler):
    """Random domain sampler.

    This sampler randomly samples N domains each with K
    images to form a minibatch.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.label_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.label_dict[item.label].append(i)
        self.labels = list(self.label_dict.keys())

        #unique label
        n_label = len(self.labels)

        # Make sure each domain has equal number of images
        # if n_domain is None or n_domain <= 0:
        #     n_domain = len(self.domains)
        assert batch_size % n_label == 0
        self.n_img_per_label = batch_size // n_label

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_label = n_label
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        label_dict = copy.deepcopy(self.label_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.labels, self.n_label)
            # selected_domains = [i for i in range(self.n_label)]
            for domain in selected_domains:
                idxs = label_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_label)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    label_dict[domain].remove(idx)

                remaining = len(label_dict[domain])
                if remaining < self.n_img_per_label:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length

class SequentialDomainSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        # self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False
        selected_idxs = []
        for domain in self.domains:
            idxs = domain_dict[domain]
            total_domain_idx = len(idxs)
            if total_domain_idx == 0:
                pass
            elif total_domain_idx > self.batch_size:
                selected_idxs = random.sample(idxs, self.batch_size)
                final_idxs.extend(selected_idxs)
            # else total_domain_idx <= self.batch_size:
            else:
                selected_idxs = idxs
            final_idxs.extend(selected_idxs)
            #     if




            for idx in selected_idxs:
                domain_dict[domain].remove(idx)

            remaining = len(domain_dict[domain])

        # while not stop_sampling:
        return iter(final_idxs)


def calculate_sampling_weight(data_source):
    """

    """
    import numpy as np
    # for item in data_source:
    #     print("current item : ",item)
    labels = np.array([item[1] for item in data_source])

    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])

    weight = 1. / class_sample_count
    print("total : ",len(labels))
    print("class count : ",class_sample_count)
    print("class sample weights for sampler : ",weight)

    samples_weight = np.array([weight[t] for t in labels])

    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight,len(samples_weight)

def build_sampler(
    sampler_type, cfg=None, data_source=None, batch_size=32, n_domain=0
):
    if sampler_type == 'RandomSampler':
        return RandomSampler(data_source)

    elif sampler_type == 'SequentialSampler':
        return SequentialSampler(data_source)
    elif sampler_type =='GroupLabelSampler':
        return GroupLabelSampler(data_source,batch_size)
    elif sampler_type == 'RandomDomainSampler':
        return RandomDomainSampler(data_source, fix_batch_sie=batch_size)
    elif sampler_type == 'UnderSampler':
        sample_weights, counts = calculate_sampling_weight(data_source)
        # print("sample_weight : ",sample_weights)
        # print("count : ",counts )
        return WeightedRandomSampler(sample_weights, len(counts) * int(counts.min()), replacement=False)
    elif sampler_type == 'OverSampler':
        sample_weights, counts = calculate_sampling_weight(data_source)
        return WeightedRandomSampler(sample_weights, len(counts) * int(counts.max()), replacement=True)
    elif sampler_type == 'WeightRandomSampler':
        sample_weights, counts = calculate_sampling_weight(data_source)
        return WeightedRandomSampler(sample_weights,counts)
    else:
        raise ValueError('Unknown sampler type: {}'.format(sampler_type))
