import argparse
import ast
import torch

from attacks.attack_backdoor_kf import FinetuneTest
from utils.general import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune and test the WF model')
    # parser.add_argument('--model', choices=['df', 'inception', 'tmwf', 'tf', 'rf', 'rf2', 'rf3', 'rf4'],
    #                     default='rf4',
    #                     help='choose the model')
    parser.add_argument('--model', choices=['df', 'inception', 'tmwf', 'ares', 'varcnn', 'tf', 'rf', 'kfingerprinting'],
                        default='rf',)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--unanimous', action='store_true')  # 若你想默认 True，可以改成 default=True 的 bool 参数

    parser.add_argument('--backdoor-ratio', type=float, default=0.1, help='backdoor ratio')
    parser.add_argument('--backdoor-num', type=int, default=4, help='backdoor ratio')
    parser.add_argument('--backdoor-type', type=str, default='None', help='backdoor type')
    parser.add_argument('--backdoor-amp', action='store_true', default=False, help='not use mixed precision training')
    parser.add_argument('--return-backdoored', action='store_true', default=False, help='return backdoored sample')
    parser.add_argument('--adversarial-state', action='store_true', default=False, help='not use backdoor training and test with triggers')
    parser.add_argument('--feature-type', choices=['tam', 'tam+', 'burst', 'df', 'tiktok', 'fusion', 'patch'],
                        default='fusion', help='feature type')

    parser.add_argument('--fusion-granularity', '--fg', type=int, default=9, help='fusion granularity')

    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='train or test')

    # paths and file config
    parser.add_argument('--data-path', type=str, help="data path")
    parser.add_argument('--model-path', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--exist-ok', action='store_true', default=False, help='increment the path if exists')
    parser.add_argument('--nosave', action='store_true', default=False, help='do not save the model')
    parser.add_argument('--asr', action='store_true', default=False, help='evaluate asr')

    parser.add_argument('--backdoor-lable', default=0, type=int, help='the backdoor lable')
    parser.add_argument('--backdoor-label-type', type=str, default='poi', help='backdoor label type is poisoned or clean')
    parser.add_argument('--backdoor-length', default=20, type=int, help='the backdoor lable')
    parser.add_argument('--eval-nums', default=20, type=int, help='sample nums for eval and test dataset')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--SHAP-trigger-pth', type=str, default=None, help='pretrained model path')
    parser.add_argument('--trigger-pth', type=str, default=None, help='pretrained model path')
    parser.add_argument('--trigger-idx', type=str, default="[]", help='List of trigger indices')
    parser.add_argument('--trigger-value', type=str, default="[]", help='List of trigger values')


    parser.add_argument('--suffix', type=str, default='.cell', help='suffix of the output file')
    parser.add_argument('--mon-classes', default=100, type=int, help='Number of monitored classes')
    parser.add_argument('--mon-inst', default=100, type=int,
                        help='Number of monitored instances per class')
    parser.add_argument('--mon-inst-train', default=-1, type=int,
                        help='Number of monitored instances per class for training (-1 to use all)')
    parser.add_argument('--unmon-inst', default=10000, type=int,
                        help='Number of unmonitored instances')
    parser.add_argument('--unmon-inst-train', default=-1, type=int, help='Number of unmonitored instances for training')
    parser.add_argument('--open-world', default=False, action="store_true", help='Open world or not')
    parser.add_argument('--seq-length', default=1800, type=int, help='The input trace length')

    # website or webpage
    parser.add_argument('--page-per-class', default=1, type=int,
                        help='For good enough dataset, each class has a few pages')

    # augmentation
    parser.add_argument('--num-kernels', type=int, default=4, help='for rf4')
    parser.add_argument('--n-tam', default=2, type=int, help='The number of tam channels')
    parser.add_argument('--aug-times', '-a', default=0, type=int, help='The number of augmentation times')
    parser.add_argument('--averaging-times', default=4, type=int, help='The number of averaging times')

    # optimization
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial optimizer learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--label-smoothing', '--ls', type=float, default=0., help='label smoothing')

    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')

    # GPU
    parser.add_argument('--use-gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use-multi-gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multiple gpus')
    parser.add_argument('--not-amp', action='store_true', default=False, help='not use mixed precision training')

    # LOG
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed performance')

    # seed
    parser.add_argument('--seed', type=int, default=2024, help='seed')

    # one fold or not
    parser.add_argument('--one-fold', action='store_true', default=False, help='run only one fold')

    _args = parser.parse_args()
    return _args 


if __name__ == '__main__':
    args = parse_arguments()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    seed_everything(2024)

    args.trigger_idx = ast.literal_eval(args.trigger_idx)
    args.trigger_value = ast.literal_eval(args.trigger_value)

    if args.verbose:
        print(args)

    attack = FinetuneTest(args)
    attack.run(one_fold_only=args.one_fold)
