import mlp
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MLP.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 20
    args.batchsize = 128
    args.dataset = 'mnist'
    args.ext = 'png'
    mlp.run(args)
