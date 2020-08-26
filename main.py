import torch
from param_parser import parameter_parser
from trainer import Trainer
from utils import tab_printer


def main():
    # parse command line arguments
    args = parameter_parser()

    # set arguments
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    tab_printer(args)

    # define a trainer and train HOGCN
    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
