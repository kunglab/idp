import binarynet
import util

if __name__ == '__main__':
    '''
    Settings used to generate figures in paper for BinaryNet.
    '''
    args = util.default_parser().parse_args()
    args.clip = 1
    args.epoch = 50
    args.batchsize = 128
    args.opt = 'adam'
    args.learnrate = 0.003
    args.dataset = 'mnist'
    args.ext = 'pdf'
    binarynet.run(args)
