"""
MolBCAT — Primary CLI Entry Point
===================================

Usage:

  # Step 1: Pretrain GRU encoder on ZINC250k
  python main.py pretrain

  # Step 2a: Train classification experiments
  python main.py train_cls

  # Resume interrupted classification experiment
  python main.py train_cls --resume

  # Run a specific classification dataset
  python main.py train_cls --dataset BBBP

  # Step 2b: Train regression experiments
  python main.py train_reg

  # Resume interrupted regression experiment
  python main.py train_reg --resume

  # Run a specific regression dataset
  python main.py train_reg --dataset ESOL

  # Step 3: Ablation study
  python main.py train_ablation

  # Predict on new molecules
  python main.py predict --dataset BBBP \
      --smiles "CC(=O)Oc1ccccc1C(=O)O"

  python main.py predict \
      --dataset ESOL \
      --input molecules.csv \
      --output results.csv
"""

import argparse


def main():

    parser = argparse.ArgumentParser(
        prog="molbcat",
        description=(
            "MolBCAT — Bidirectional Cross-Modal Attention "
            "for Molecular Property Prediction"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )


    p_pre = subparsers.add_parser(
        "pretrain",
        help="Pretrain SMILES encoder on ZINC250k",
    )

    p_pre.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
    )



    p_cls = subparsers.add_parser(
        "train_cls",
        help="Run classification experiments",
    )

    p_cls.add_argument(
        "--config",
        type=str,
        default="configs/classification.yaml",
    )

    p_cls.add_argument(
        "--dataset",
        type=str,
        default="all",
        help='Dataset name or "all" (default: all)',
    )

    p_cls.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed fold checkpoints",
    )

    

    p_reg = subparsers.add_parser(
        "train_reg",
        help="Run regression experiments",
    )

    p_reg.add_argument(
        "--config",
        type=str,
        default="configs/regression.yaml",
    )

    p_reg.add_argument(
        "--dataset",
        type=str,
        default="all",
        help='Dataset name or "all" (default: all)',
    )

    p_reg.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed fold checkpoints",
    )



    p_ab = subparsers.add_parser(
        "train_ablation",
        help="Run concatenation ablation study",
    )

    p_ab.add_argument(
        "--cls_config",
        type=str,
        default="configs/classification.yaml",
    )

    p_ab.add_argument(
        "--reg_config",
        type=str,
        default="configs/regression.yaml",
    )

    p_ab.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/ablation",
    )

   

    p_pr = subparsers.add_parser(
        "predict",
        help="Predict molecular properties",
    )

    p_pr.add_argument(
        "--weights_dir",
        type=str,
        default="./weights",
    )

    p_pr.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "BBBP",
            "HIV",
            "ClinTox",
            "Tox21_NR_AR",
            "ESOL",
            "Lipophilicity",
        ],
    )

    p_pr.add_argument(
        "--model",
        type=str,
        default="MolBCAT",
        choices=[
            "MolBCAT",
            "GRU_Finetune",
            "GRU_Frozen",
            "GRU_Random",
            "GIN",
            "ChemBERTa",
        ],
    )

    p_pr.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    p_pr.add_argument(
        "--smiles",
        type=str,
        default=None,
    )

    p_pr.add_argument(
        "--input",
        type=str,
        default=None,
    )

    p_pr.add_argument(
        "--output",
        type=str,
        default=None,
    )


    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return


    if args.command == "pretrain":

        from scripts.pretrain import main as run

        run(args)

    elif args.command == "train_cls":

        from scripts.train_cls import main as run

        run(args)

    elif args.command == "train_reg":

        from scripts.train_reg import main as run

        run(args)

    
    elif args.command == "train_ablation":

        from scripts.train_ablation import main as run

        run(args)

    elif args.command == "predict":

        if args.smiles is None and args.input is None:
            raise ValueError(
                "Either --smiles or --input must be provided"
            )

        from scripts.predict import main as run

        run(args)


if __name__ == "__main__":
    main()