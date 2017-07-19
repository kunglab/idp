import mlp_multi
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MLP.
    '''
    args = util.default_parser().parse_args()
    args.opt = 'momentum'
    args.epoch = 30
    args.batchsize = 128
    args.dataset = 'mnist'
    args.ext = 'pdf'
    mlp_multi.run(args)
