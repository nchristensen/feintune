from feintune.fused_autotuning import main
import argparse
# FIXME: Module invocation doesn't currently support Charm
# use python fused_autotuning for that for now (with use_charm=True set)

parser = argparse.ArgumentParser("feintune")
parser.add_argument("--infile", dest="infile", type=str,
                    help="File containing pickled (macro)kernels to tune.", default="./pickled_programs")
parser.add_argument("--outfile", dest="--outfile", type=str,
                    help="File while where tuning data will be saved.", default="./autotuning_files")
parser.add_argument("--benchmark", dest="benchmark", action="store_true",
                    help="Whether or not to gather roofline data before tuning.")
args = parser.parse_args()

main(args)
