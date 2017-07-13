import binarynet
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MobileNet.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 10 
    args.clip = 1
    args.batchsize = 64
    args.learnrate = 1e-4
    args.dataset = 'mnist'
    binarynet.run(args)
