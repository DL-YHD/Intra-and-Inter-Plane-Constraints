from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" # 放在import torch后无法指定gpu id号。
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset

from trains.car_pose import CarPoseTrainer
from torchstat import stat
from thop import profile
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)
  if opt.distribute:
      local_rank = os.environ['SLURM_LOCALID']
      torch.cuda.set_device(int(float(local_rank)))
  #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  print('model parameters')
  getModelSize(model)

  #dummy_input = torch.randn(1, 4, 128, 128,128)
  #flops, params = profile(model, (dummy_input,))
  #print('flops: ', flops, 'params: ', params)
  #print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

  # dummy_input = torch.randn(2, 3, 384, 1280)
  # flops, params = profile(model.cpu(), (dummy_input,))
  # print('flops: ', flops, 'params: ', params)
  # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
  # flops, params, _, _ , _ = profile(model, (2, 3, 384, 1280))
  # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
  # print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))

  # inputs = torch.rand([3, 384, 1280])
  # tensor = (torch.rand(3, 384, 1280).to(opt.device),)
  # print('tensor',len(tensor))
  # inputs = inputs.to(opt.device)
  # model.to(opt.device)
  # stat(model.to(opt.device), (3, 384, 1280),logger=None)

  # 创建输入网络的tensor
  # tensor = (torch.rand(1, 3, 224, 224),)
  # 分析FLOPs
  # flops = FlopCountAnalysis(model, tensor)
  # print("FLOPs: ", flops.total())

  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    

  Trainer = CarPoseTrainer
  trainer = Trainer(opt, model, optimizer)
  
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device, opt.distribute)


  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val', opt.planes_n_kps), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train', opt.planes_n_kps), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  ) # 加载 y label

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      # with torch.no_grad():
      #   log_dict_val, preds = trainer.val(epoch, val_loader)
      # for k, v in log_dict_val.items():
      #   logger.scalar_summary('val_{}'.format(k), v, epoch)
      #   logger.write('{} {:8f} | '.format(k, v))
      # if log_dict_val[opt.metric] < best:
      #   best = log_dict_val[opt.metric]
      #   save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
      #              epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
        if opt.distribute:
            local_rank = os.environ['SLURM_LOCALID']
            if int(float(local_rank)) == 0:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                           epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
        lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
        print('Drop LR to', lr)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
