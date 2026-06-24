import torch
import ckpoint as ckp
import data
import loss
import model
from option import args
from train.trainer_denoise_2loss_shallow import Trainer_2loss

import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(args.seed)
checkpoint = ckp.checkpoint_1model_2loss(args)


#\033[1;34m 字体颜色：蓝色\033[0m
def main():
    global model
    if checkpoint.ok: 
        loader = data.Data(args) 
        _model = model.Model(args, checkpoint) 
        print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (sum(p.numel() for p in _model.parameters())/1000000.0)) 
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        _loss2 = loss.Loss2(args, checkpoint)  if not args.test_only else None
        
        model_list = [_model]
        loss_list = [_loss,_loss2]
        
        t = Trainer_2loss(args, loader, model_list, loss_list, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()


if __name__ == '__main__':
    
    main()
