import mlp_coef_comparison
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MLP.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 50
    args.batchsize = 128
    args.dataset = 'mnist'
    args.ext = 'pdf'
    mlp_coef_comparison.run(args)
