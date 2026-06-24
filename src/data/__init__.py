# -*- coding: UTF-8 -*-
import torch
from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import json
import  os


class MyConcatDataset(ConcatDataset):#
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets

data_folder_path  = os.path.dirname(__file__)
#this concatdataset will return the data of datset and input the dataset list they can cat many dataset
dataset_json = open(os.path.join(data_folder_path, "dataset.json"), "r")
dataset_dict = json.load(dataset_json)

cv2denoisedataset_list = dataset_dict["cv2denoisedataset_list"]
threeinputdenoisedataset_list = dataset_dict["threeinputdenoisedataset_list"]
croltroldataset_list = dataset_dict["croltroldataset_list"]
sitkdenoisedataset_list = dataset_dict["sitkdenoisedataset_list"]
npydenoisedataset_list = dataset_dict["npydenoisedataset_list"]
cv2classdataset_list = dataset_dict["cv2classdataset_list"]
cv2segmentdataset_list = dataset_dict["cv2segmentdataset_list"]
cv2segmentdataset_edge_list = dataset_dict["cv2segmentdataset_edge_list"]
cv2segment_cop_class_dataset_list = dataset_dict["cv2segment_cop_class_dataset_list"]
cv2segmentdataset_lastframe_list = dataset_dict["cv2segmentdataset_lastframe_list"]
cv2segmentdataset_multitask_list = dataset_dict["cv2segmentdataset_multitask_list"]
cv2segmentdataset_multitask_multiframe_list = dataset_dict["cv2segmentdataset_multitask_multiframe_list"]
cv2npysegmentdataset_list = dataset_dict["cv2npysegmentdataset_list"]
cv2srdataset_list = dataset_dict["cv2srdataset_list"]


dataset_json.close()

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d  
                # define the dataset function   
                if module_name in sitkdenoisedataset_list:
                    dataset_class_name = "SitkDenoiseDataset"
                elif module_name in cv2denoisedataset_list:
                    dataset_class_name = "Cv2DenoiseDataset"
                elif module_name in threeinputdenoisedataset_list:
                    dataset_class_name = "ThreeinputDataset"
                elif module_name in croltroldataset_list:
                    dataset_class_name = "Controldata"
                elif module_name in npydenoisedataset_list:
                    dataset_class_name = "NPYDenoiseDataset"
                elif module_name in cv2classdataset_list:
                    dataset_class_name = "Cv2ClassDataset"
                elif module_name in cv2segmentdataset_list:
                    dataset_class_name = "Cv2SegmentDataset"
                elif module_name in cv2npysegmentdataset_list:
                    dataset_class_name = "Cv2NPYSegmentDataset"
                elif module_name in cv2segment_cop_class_dataset_list:
                    dataset_class_name = "Cv2SegmentDataset_Cop_Class"
                elif module_name in cv2segmentdataset_lastframe_list:
                    dataset_class_name = "Cv2SegmentDataset_lastframe"
                elif module_name in cv2segmentdataset_multitask_list:
                    dataset_class_name = "Cv2SegmentDataset_multitask"
                elif module_name in cv2segmentdataset_multitask_multiframe_list:
                    dataset_class_name = "Cv2SegmentDataset_multitask_multiframe"

                elif module_name in cv2srdataset_list:
                    dataset_class_name = "Cv2SRDataset"


                # print(module_name)
                m = import_module('data.' + dataset_class_name.lower())
                datasets.append(getattr(m, dataset_class_name)(args,train_dataset_name=d))
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=args.n_threads,
                prefetch_factor=8
            )
        self.loader_test = []
        for d in args.data_test:
            module_name = d
            if d.find("ISTD")>=0:
                m = import_module('data.threeinputbenchmark')
                testset = getattr(m, 'ThreeinputBenchmark')(args, test_dataset_name=d)
            # 250704 add imageclass test dataset
            elif module_name in cv2classdataset_list:
                m = import_module('data.classbenchmark')
                testset = getattr(m, 'ClassBenchmark')(args, test_dataset_name=d)

            elif module_name in cv2segmentdataset_list:
                m = import_module('data.segmentbenchmark')
                testset = getattr(m, 'SegmentBenchmark')(args, test_dataset_name=d)

            elif module_name in cv2npysegmentdataset_list:
                m = import_module('data.cv2npysegmentbenchmark')
                testset = getattr(m, 'Cv2NPYSegmentBenchmark')(args, test_dataset_name=d)
            # SegmentBenchmark_Cop_Class
            elif module_name in cv2segment_cop_class_dataset_list:
                m = import_module('data.segmentbenchmark_cop_class')
                testset = getattr(m, 'SegmentBenchmark_Cop_Class')(args, test_dataset_name=d)

            elif module_name in cv2segmentdataset_lastframe_list:
                m = import_module('data.segmentbenchmark_lastframe')
                testset = getattr(m, 'SegmentBenchmark_lastframe')(args, test_dataset_name=d)

            elif module_name in cv2segmentdataset_multitask_list:
                m = import_module('data.segmentbenchmark_multitask')
                testset = getattr(m, 'SegmentBenchmark_multitask')(args, test_dataset_name=d)

            elif module_name in cv2segmentdataset_multitask_multiframe_list:
                m = import_module('data.segmentbenchmark_multitask_multiframe')
                testset = getattr(m, 'SegmentBenchmark_multitask_multiframe')(args, test_dataset_name=d)

            elif module_name in cv2srdataset_list:
                m = import_module('data.srbenchmark')
                testset = getattr(m, 'SRBenchmark')(args, test_dataset_name=d)

            else:
                m = import_module('data.benchmark')
                testset = getattr(m, 'DnBenchmark')(args, test_dataset_name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=False,
                    num_workers=args.n_threads,
                )
            )


class DataM:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d
                # define the dataset function
                if module_name in sitkdenoisedataset_list:
                    dataset_class_name = "SitkDenoiseDataset"
                elif module_name in cv2denoisedataset_list:
                    dataset_class_name = "Cv2DenoiseDataset"
                elif module_name in threeinputdenoisedataset_list:
                    dataset_class_name = "ThreeinputDataset"
                elif module_name in croltroldataset_list:
                    dataset_class_name = "Controldata"
                elif module_name in npydenoisedataset_list:
                    dataset_class_name = "NPYDenoiseDataset"

                # print(module_name)
                m = import_module('data.' + dataset_class_name.lower())
                datasets.append(getattr(m, dataset_class_name)(args, train_dataset_name=d))
            self.loader_train =  MultiEpochsDataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=args.n_threads,

            )
        self.loader_test = []
        for d in args.data_test:
            if d.find("ISTD") >= 0:
                m = import_module('data.threeinputbenchmark')
                testset = getattr(m, 'ThreeinputBenchmark')(args, test_dataset_name=d)
            else:
                m = import_module('data.benchmark')
                testset = getattr(m, 'DnBenchmark')(args, test_dataset_name=d)
            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=False,
                    num_workers=args.n_threads,
                )
            )


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
