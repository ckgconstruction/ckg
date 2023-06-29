
class EMFHCfgs():
    def __init__(self, args):

        self.HIGH_ORDER = args.high_order
        self.HIDDEN_SIZE = args.hidden_size
        self.MFB_K = args.mfb_k
        self.MFB_O = args.mfb_o
        self.LSTM_OUT_SIZE = args.lstm_out_size
        self.FRCN_FEAT_SIZE = args.frcn_feat_size
        self.DROPOUT_R = args.dropout_r
        self.I_GLIMPSES = args.i_glimpses
        self.Q_GLIMPSES = args.q_glimpses
        self.FEAT_SIZE = args.feat_size
        self.n_layers = args.n_layers
        self.BATCH_SIZE = args.batch_size


def get_emfh_args(args):
    emfh_args = EMFHCfgs(args)
    return emfh_args
