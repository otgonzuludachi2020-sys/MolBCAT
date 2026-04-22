"""
MolBCAT — Primary CLI Entry Point
===================================
Usage:

  # Step 1: Pretrain GRU encoder on ZINC250k
  python main.py pretrain

  # Step 2a: Train classification experiments (BBBP, HIV, ClinTox, Tox21)
  python main.py train_cls

  # Step 2b: Train regression experiments (ESOL, Lipophilicity)
  python main.py train_reg

  # Step 3a: Data efficiency experiment (BBBP + Lipophilicity)
  python main.py train_dataeff

  # Step 3b: Concatenation ablation study (BBBP, ClinTox, Lipophilicity)
  python main.py train_ablation

  # Predict on new molecules
  python main.py predict --dataset BBBP --smiles "CC(=O)Oc1ccccc1C(=O)O"
  python main.py predict --dataset ESOL --input molecules.csv --output results.csv

  # Run a specific dataset only
  python main.py train_cls --dataset BBBP
  python main.py train_reg --dataset ESOL
"""
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='molbcat',
        description='MolBCAT — Bidirectional Cross-Modal Attention for Molecular Property Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest='command')

    # pretrain
    p_pre = subparsers.add_parser('pretrain', help='Pretrain SMILES encoder on ZINC250k')
    p_pre.add_argument('--config', type=str, default='configs/pretrain.yaml')

    # train_cls
    p_cls = subparsers.add_parser('train_cls', help='Run classification experiments')
    p_cls.add_argument('--config',  type=str, default='configs/classification.yaml')
    p_cls.add_argument('--dataset', type=str, default='all',
                       help='Dataset name or "all" (default: all)')

    # train_reg
    p_reg = subparsers.add_parser('train_reg', help='Run regression experiments')
    p_reg.add_argument('--config',  type=str, default='configs/regression.yaml')
    p_reg.add_argument('--dataset', type=str, default='all',
                       help='Dataset name or "all" (default: all)')

    # train_dataeff
    p_de = subparsers.add_parser('train_dataeff', help='Run data efficiency experiments')
    p_de.add_argument('--cls_config', type=str, default='configs/classification.yaml')
    p_de.add_argument('--reg_config', type=str, default='configs/regression.yaml')
    p_de.add_argument('--out_dir',    type=str, default='./outputs/data_efficiency')

    # train_ablation
    p_ab = subparsers.add_parser('train_ablation', help='Run concatenation ablation study')
    p_ab.add_argument('--cls_config', type=str, default='configs/classification.yaml')
    p_ab.add_argument('--reg_config', type=str, default='configs/regression.yaml')
    p_ab.add_argument('--out_dir',    type=str, default='./outputs/ablation')

    # predict
    p_pr = subparsers.add_parser('predict', help='Predict molecular properties')
    p_pr.add_argument('--weights_dir', type=str, default='./weights')
    p_pr.add_argument('--dataset',     type=str, required=True,
                      choices=['BBBP', 'HIV', 'ClinTox', 'Tox21_NR_AR',
                                'ESOL', 'Lipophilicity'])
    p_pr.add_argument('--model',       type=str, default='MolBCAT',
                      choices=['MolBCAT', 'GRU_Finetune', 'GRU_Frozen',
                                'GRU_Random', 'GIN'])
    p_pr.add_argument('--seed',        type=int, default=1)
    p_pr.add_argument('--smiles',      type=str, default=None)
    p_pr.add_argument('--input',       type=str, default=None)
    p_pr.add_argument('--output',      type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'pretrain':
        from scripts.pretrain import main as run
        run(args)

    elif args.command == 'train_cls':
        from scripts.train_cls import main as run
        run(args)

    elif args.command == 'train_reg':
        from scripts.train_reg import main as run
        run(args)

    elif args.command == 'train_dataeff':
        from scripts.train_dataeff import main as run
        run(args)

    elif args.command == 'train_ablation':
        from scripts.train_ablation import main as run
        run(args)

    elif args.command == 'predict':
        if args.smiles is None and args.input is None:
            raise ValueError("Either --smiles or --input must be provided")

        from scripts.predict import main as run
        run(args)


if __name__ == '__main__':
    main()
