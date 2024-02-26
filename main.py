import argparse
from pathlib import Path
from fit_data import get_args_parser, train_model
from render import renderObjects, threeDObject
import torch

def parse_args():
    parser = get_args_parser()
    parser.add_argument('--q', type=float, default=1.1, help='Question number for the assignment')
    parser.add_argument("--output_path", type=str, default="data/shreyasj")
    return parser.parse_args()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    print(args)

    output_path = Path(args.output_path) / str(args.q)
    output_path.mkdir(parents=True, exist_ok=True)

    q, subq = str(args.q).split('.')

    if q == '1':
        if subq == '1':
            args.type = "vox"
            train_model(args)
        elif subq == '2':
            args.type = "point"
            train_model(args)
        elif subq == '3':
            args.type = "mesh"
            train_model(args)
            
    elif q == '2':
        pass
    elif q == '3':
        raise('Not implemented')
    else:
        raise('Invalid question value')