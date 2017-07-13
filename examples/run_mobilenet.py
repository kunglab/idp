import mobilenet
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MobileNet.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 100
    args.batchsize = 128
    args.dataset = 'cifar10'
    mobilenet.run(args)
