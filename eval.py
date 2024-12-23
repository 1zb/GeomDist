
import trimesh
from scipy.spatial import cKDTree as KDTree
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ply', required=True, type=str)
parser.add_argument('--reference', required=True, type=str)
parser.add_argument('--scale', required=True, type=str)
args = parser.parse_args()

scale = np.load(args.scale)

prediction = trimesh.load(args.ply).vertices * scale
reference = trimesh.load(args.reference).vertices * scale

tree = KDTree(prediction)
dist, _ = tree.query(reference)
d1 = dist
gt_to_gen_chamfer = np.mean(dist)
gt_to_gen_chamfer_sq = np.mean(np.square(dist))

tree = KDTree(reference)
dist, _ = tree.query(prediction)
d2 = dist
gen_to_gt_chamfer = np.mean(dist)
gen_to_gt_chamfer_sq = np.mean(np.square(dist))

cd = gt_to_gen_chamfer + gen_to_gt_chamfer
print(cd)