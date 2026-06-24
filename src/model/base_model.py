import os
from importlib import import_module
import torch
import torch.nn as nn
from torch.autograd import Variable
global test_only
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self, args, ckp):
        super(BaseModel,self).__init__()
        self.model_savename = self.set_model_savename()
        self.model_define = args.model
        
        #print('\033[1;32m[ =======> Making {:} {:}<======= ]\033[0m '.format(self.model_savename,self.model_define))
        self.cpu = args.cpu # 
        self.device = torch.device('cpu' if args.cpu else "cuda:0" )

       #  print(self.device)
        self.n_GPUs = args.n_GPUs 
        self.save_models = args.save_models 
        self.args = args
        
        
        #import model 
        module = import_module('model.' + self.model_define.lower()) 
        self.model = module.make_model(args).to(self.device)
        self.model.apply(self._init_weights)
        
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
            
        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file) # save in ckp log 
        
    @abstractmethod
    def set_model_savename(self):
        pass
    
    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform(m.weight)
            nn.init.constant(m.bias, 0)
        if isinstance(m,nn.LayerNorm):
            nn.init.constant(m.bias, 0)
            nn.init.constant(m.weight, 1.0)
    def get_model(self):
        return self.model
    
    def forward(self,x):
        out = self.model(x)
        return out

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
        
    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, '{:}_latest.pt'.format(self.model_savename))
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, '{:}_best.pt'.format(self.model_savename))
            )
        
        if self.save_models and epoch %2 == 0 :
            torch.save(
                target.state_dict(),
                os.path.join(apath, '{:}_{}.pt'.format(self.model_savename,epoch))
            )
            
    
    def load(self, ckpdir, pre_train='.', resume=-1, cpu=False):
        #这里的可控选项的包括如下 : resume  = -1 ，不管pretrain的加载路径，这里将会直接在之前保存的最近的模型
        #进行加载的过程 ，
        #如果这里的resume= 0 这里将会进行加载与训练的模型的过程
        # resume = -1 是继续训练模式，将会在当前文件夹找到最近的模型，为断点继续模式,要输入resload_path，结果会保存在上述路径中
        # resume = -1 且test_only 是最后模型评估模式，同样需要pre_train ，结果在pre_train中覆盖
        # resume = 0 是加载预训练模式，将会加载指定路径下的权重文件后，重新开始训练,要输入pre_train，结果会保存在ressave_path中
        # resume = 0 且test_only 是评估模式，如果指定了pre_train 的路径 将会找出指标最高模型文件测试，结果覆盖在Evation，如果指定了resload_path 这里将会 找出最后的模型文件 
        # resume = -2 是不加载模式，全新的训练
        # resume = 其他，是任意加载模式，将会加载指定epoch的相关文件
        #这里的ckpdir就是checkpoint里面的dir就是../experiment/train/[Unet]-[2023-7-18-14:23]
        # strict = True为严格模式 加载预训练的键值对如对不上就会报错，反之则不会 
        
        #print(self.model_savename,"self.model_savename")
        
        
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        if resume == -1:

            print(
                "\033[1;32m[ =======> New Train With Trained Model From The Lastest Epoch. Now Will Continue Train ..<====== ] \033[0m")
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(ckpdir+'/model/', '{:}_latest.pt'.format(self.model_savename)),   # ../experiment/train/resload_path/model/
                    **kwargs
                ),
                strict=True
            )

        elif resume == 0:
            if pre_train != '.':
                
                print("\033[1;32m[ =======> Loading model from Pretrain {} <====== ] \033[0m".format(pre_train))
                pre_train_model_path = os.path.join('..', 'experiment', pre_train, "model")
                self.get_model().load_state_dict(
                    torch.load(pre_train_model_path+"/{:}_best.pt".format(self.model_savename), **kwargs),
                    strict=True
                )
            
            else :
                print("\033[1;31m[ =======> The PreTrain Path Is Wrong ! Will Not Load Any Model <====== ] \033[0m")
        
        elif resume == -2:
            print("\033[1;32m[ =======> New Train Without Any Trained Model <====== ] \033[0m")
        elif resume == -3:
            if pre_train != '.':
                
                print("\033[1;32m[ =======> Loading model from Pretrain {} <====== ] \033[0m".format(pre_train))
                pre_train_model_path = os.path.join('..', 'experiment', pre_train, "model")
                self.get_model().load_state_dict(
                    torch.load(pre_train_model_path+"/{:}_latest.pt".format(self.model_savename), **kwargs),
                    strict=True
                )
            
            else :
                print("\033[1;31m[ =======> The PreTrain Path Is Wrong ! Will Not Load Any Model <====== ] \033[0m")
            
        else :
            print("\033[1;32m[ =======> New Train With Trained Model From Epoch :{}. Now Will ReTrain ..<====== ] \033[0m".format(resume))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(ckpdir, 'model', '{:}_{}.pt'.format(self.model_savename,resume)),
                    **kwargs
                ),
                strict=True
            )


class BaseModel2(nn.Module):
    def __init__(self, args, ckp):
        super(BaseModel2, self).__init__()
        self.model_savename = self.set_model_savename()
        self.model_define = None

        # print('\033[1;32m[ =======> Making {:} {:}<======= ]\033[0m '.format(self.model_savename,self.model_define))
        self.args = args
        self.ckp = ckp
        self.cpu = args.cpu  #
        self.device = torch.device('cpu' if args.cpu else 'cuda:0')   #   260530  cuda --> cuda0

        print(self.device)
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
	#self.model.apply(self._init_weights)

    def load_model(self):
        # import model
        module = import_module('model.' + self.model_define.lower())

        # print("device ",self.device)
        self.model = module.make_model(self.args).to(self.device)
        self.model.apply(self._init_weights)

        if not self.args.cpu and self.args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(self.args.n_GPUs))

        self.load(
            self.ckp.dir,
            pre_train=self.args.pre_train,
            resume=self.args.resume,
            cpu=self.args.cpu
        )
        print(self.model, file=self.ckp.log_file)  # save in ckp log

    @abstractmethod
    def set_model_savename(self):
        pass

    def _init_weights(self, m):
        #print(m)
        # 这里的初始化， 如果是bias 为false ,将会报错
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            #nn.init.constant(m.weight, 1.0)
            # nn.init.kaiming_normal_(
            #     m.weight,
            #     mode="fan_in",
            #     nonlinearity="relu"
            # )
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            #nn.init.constant(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant(m.bias, 0)
            nn.init.constant(m.weight, 1.0)

    def get_model(self):
        return self.model

    def forward(self, x,**kwargs):

        out = self.model(x,**kwargs)
        return out

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, '{:}_latest.pt'.format(self.model_savename))
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, '{:}_best.pt'.format(self.model_savename))
            )

        if self.save_models and epoch % self.args.save_models_every ==0 :
            torch.save(
                target.state_dict(),
                os.path.join(apath, '{:}_{}.pt'.format(self.model_savename, epoch))
            )

    def load(self, ckpdir, pre_train='.', resume=-1, cpu=False):
        # 这里的可控选项的包括如下 : resume  = -1 ，不管pretrain的加载路径，这里将会直接在之前保存的最近的模型
        # 进行加载的过程 ，
        # 如果这里的resume= 0 这里将会进行加载与训练的模型的过程
        # resume = -1 是继续训练模式，将会在当前文件夹找到最近的模型，为断点继续模式,要输入resload_path，结果会保存在上述路径中
        # resume = -1 且test_only 是最后模型评估模式，同样需要pre_train ，结果在pre_train中覆盖
        # resume = 0 是加载预训练模式，将会加载指定路径下的权重文件后，重新开始训练,要输入pre_train，结果会保存在ressave_path中
        # resume = 0 且test_only 是评估模式，将会找出指标最高模型文件测试，结果覆盖在Evation
        # resume = -2 是不加载模式，全新的训练
        # resume = -3 仅评估最后保存的模型 
        # resume = 其他，是任意加载模式，将会加载指定epoch的相关文件
        # 这里的ckpdir就是checkpoint里面的dir就是../experiment/train/[Unet]-[2023-7-18-14:23]
        # strict = True为严格模式 加载预训练的键值对如对不上就会报错，反之则不会

        # print(self.model_savename,"self.model_savename")

        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        if resume == -1:

            print(
                "\033[1;32m[ =======> New Train With Trained Model From The Lastest Epoch. Now Will Continue Train ..<====== ] \033[0m")
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(ckpdir + '/model/', '{:}_latest.pt'.format(self.model_savename)),
                    # ../experiment/train/resload_path/model/
                    **kwargs
                ),
                strict=True
            )

        elif resume == 0:
            if pre_train != '.':

                pre_train_model_path = os.path.join(self.args.project_path, 'experiment', pre_train, "model")
                self.get_model().load_state_dict(
                    torch.load(pre_train_model_path + "/{:}_best.pt".format(self.model_savename), **kwargs),
                    strict=True
                )
                print("\033[1;32m[ =======> Loading model from Pretrain {} <====== ] \033[0m".format(pre_train))
            else:
                print("\033[1;31m[ =======> The PreTrain Path Is Wrong ! Will Not Load Any Model <====== ] \033[0m")

        elif resume == -3:
            if pre_train != '.':

                print("\033[1;32m[ =======> Loading model from Pretrain {} <====== ] \033[0m".format(pre_train))

                pre_train_model_path = os.path.join(self.args.project_path, 'experiment', pre_train, "model")
                self.get_model().load_state_dict(
                    torch.load(pre_train_model_path + "/{:}_latest.pt".format(self.model_savename), **kwargs),
                    strict=True
                )
            else:
                print("\033[1;31m[ =======> The PreTrain Path Is Wrong ! Will Not Load Any Model <====== ] \033[0m")

        elif resume == -2:
            print("\033[1;32m[ =======> New Train Without Any Trained Model <====== ] \033[0m")
        else:
            print(
                "\033[1;32m[ =======> Test  With Trained Model From Epoch :{}...<====== ] \033[0m".format(
                    resume))
            if pre_train != '.':
                print("\033[1;32m[ =======> Loading model from Pretrain {} <====== ] \033[0m".format(pre_train))
                pre_train_model_path = os.path.join(self.args.project_path, 'experiment', pre_train, "model")
                self.get_model().load_state_dict(
                    torch.load(pre_train_model_path + "/{:}_{}.pt".format(self.model_savename,resume), **kwargs),
                    strict=True
                )




