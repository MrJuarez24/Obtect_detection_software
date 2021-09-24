import argparse
from coco_evaluation_utils import evaluate

parser = argparse.ArgumentParser(description='Script to execute the coco evaluation for object detection')
parser.add_argument('-gt', '--ground_truth', help="Path to groundtruth .json annotations file.")
parser.add_argument('-r', '--results', help="Path to results .json annotations file.")
args = parser.parse_args()

evaluate(args.ground_truth, args.results)
