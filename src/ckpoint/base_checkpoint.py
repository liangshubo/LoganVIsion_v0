import os
import torch
import time
import datetime
from abc import abstractmethod
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Basecheckpoint():
    def __init__(self, args) -> None:
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.get_save_dir(args, now)

        if args.reset:
            print('\033[1;31m[ =======> Will reset the respath! <======= ]\033[0m ')
            os.system('rm -rf ' + self.dir)
            args.load = ''
        self.make_ckpdir(args)
        # ----- ------save the log.txt and load some information with now time information and the params in args -----
        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        # set the process number
        self.n_processes = 8

    def get_save_dir(self, args, now):
        # ---------------------- save_load_path and test_psnr_log init or load -----------------------------------------------------
        # 如果没有加载，就按照设定的ressave_path 创建路径后保存,判断是不是只测试，会将测试结果单独保存在test
        #

        if args.resume == -2:
            # New training mode, only for new training
            if args.ressave_path != '':
                save_dir = "[" + args.ressave_path + "]-[" + args.data_train[0] + "]" + "-[" + now + "]"
                self.dir = os.path.join(args.project_path, 'experiment', save_dir)
            else:
                save_dir = "[" + args.model + "]-[" + args.data_train[0] + "]" + "-[" + now + "]"
                self.dir = os.path.join(args.project_path, 'experiment', save_dir)
        elif args.resume == -1:
            # 断点继续模式， 训练数据保存路径应与加载路径一致,
            # 也可以用来测试最后一轮的模型结果
            if not args.test_only:
                # 断点训练
                if args.resload_path != '':
                    self.dir = os.path.join(args.project_path, 'experiment', args.resload_path)

                    if os.path.exists(self.dir):
                        self.log = torch.load(self.get_path('train-test-log.pt'))
                        print(
                            "\033[1;32m[ =======> CheckPoint Result Will Continue Save in {:} <======= ]\033[0m".format(
                                self.dir))
                        print('\033[1;32m[ =======> Continue from epoch {}... <======= ]\033[0m '.format(len(self.log)))
                    else:
                        args.load = ''
                        print('\033[1;32m[ =======> Resload_path is wrong No file <======= ]\033[0m ')
                else:
                    args.load = ''
                    print('\033[1;32m[ =======> Resload_path is wrong No file <======= ]\033[0m ')
            else:
                self.dir = os.path.join(args.project_path, 'experiment', args.resload_path, 'Evluation')

            # print("\033[1;32m[ =======>New CheckPoint Result Will Save in {:}<======= ]\033[0m".format(self.dir))
        elif args.resume == 0 or args.resume == -3:
            # 加载预训练模型模式，可以训练也用来进行模型的测试过程
            # 要是训练 应该放在新的文件夹下面
            # 测试的话，保存的结果应该放在其加载文件下方的测试文件夹中
            if args.test_only:
                self.dir = os.path.join('..', 'experiment', args.pre_train, 'Evluation')
            else:
                if args.ressave_path != '':
                    save_dir = "[" + args.ressave_path + "]-[" + args.data_train[0] + "]-[" + now + "]"
                else:
                    save_dir = "[" + args.model + "]-[" + args.data_train[0] + "]-[" + now + "]"
                self.dir = os.path.join('..', 'experiment', save_dir)
        else:
            if args.test_only:
                self.dir = os.path.join('..', 'experiment', args.pre_train, 'Evluation')

    def make_ckpdir(self, args):
        # ------------------makedir ------------------------------------------
        os.makedirs(self.dir,
                    exist_ok=True)  # exist_ok = True :if selfxdir is already exist will do nothing ;if exist_ok is False it will
        # raise a OSError in that condition
        if not args.test_only:
            os.makedirs(self.get_path('model'), exist_ok=True)
        # the data_test and data_eval not need diffierent
        # test dataset
        if args.test_only:
            for d in args.data_test:
                os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

    # ----function1 : input a subdir name will makedir a new dir that selfxdir / subdir -------------------------
    def get_path(self, *subdir):

        return os.path.join(self.dir, *subdir)

    # ----function2 : save trainer.model and train.loss by the way in model and loss  --------------------------
    @abstractmethod
    def save(self):
        """
        this will save trainer all include model ,loss ,logdraw ,optimizer
        """
        raise NotImplementedError

    # ----function3 : input a log then cat with self.log
    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    # ----function4 : load some information which called log  in the txt self.log_file the refresh will reopen the file
    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    # ----function5 : close the txt file
    def done(self):
        self.log_file.close()

    # ----function6 : draw the  iqa
    # ------ should be rewrite we not use psnr
    def plot_metric(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):  # for every data_test
            label = 'PSNR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            # for every scale 没有scale,这里要注意，log的维度问题
            plt.plot(axis, self.log[:, idx_data].numpy(),  # the log tensor will have three dim (psnr,test_data,scale)
                     label='Test PSNR')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Test PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.png'.format(d)))
            plt.close(fig)

    # ----function7 :mutil_prcess ------
    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):  # read and save all image and filename in queue one by one
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        # create a process pool and run the target_function by mutil process
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()  # mutil_process start

    # ----function8 :mutil_prcess ------
    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

