import binarynet
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for MobileNet.
    '''
    args = util.default_parser().parse_args()
    args.epoch = 4 
    args.batchsize = 64
    args.dataset = 'mnist'
    binarynet.run(args)
