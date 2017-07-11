import mlp
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MLP.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 10
    args.batchsize = 64
    args.dataset = 'mnist'
    mlp.run(args)
