"""Running MixHop or N-GCN."""

import torch
from param_parser import parameter_parser
from trainer import Trainer
from utils import tab_printer

def main():
    """
    Parsing command line parameters, reading data.
    Fitting an NGCN and scoring the model.
    """
    args = parameter_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    tab_printer(args)
    trainer = Trainer(args, True)
    trainer.fit()

    trainer.evaluate_architecture()
    # args = trainer.reset_architecture()
    # trainer = Trainer(args, False)
    # trainer.fit()

if __name__ == "__main__":
    main()

