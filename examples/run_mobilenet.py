import mobilenet
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MobileNet.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 70
    args.batchsize = 256
    args.dataset = 'cifar10'
    mobilenet.run(args)
