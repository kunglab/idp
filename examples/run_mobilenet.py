import mobilenet
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MobileNet.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 1
    args.batchsize = 1024
    args.dataset = 'cifar10'
    mobilenet.run(args)
