# Geometry Distributions

This repoitory is a refractory of the code of GeometryDistributions https://1zb.github.io/GeomDist/. Credit to Biao Zangh for the original code and the wonderful work.

### :bullettrain_front: Training 

```
torchrun --nproc_per_node=1 main.py --blr 5e-7 --output_dir output/loong --log_dir output/loong --target ../Matching-Flowing-Distributions/data/horse_04.ply --inference --epochs 501 --num_points_inference 10000 --num-step 64
```

### :balloon: Inference

```
torchrun --nproc_per_node=1 main.py --blr 5e-7 --output_dir output/loong --log_dir output/loong --target ../Matching-Flowing-Distributions/data/horse_04.ply --train --epochs 501
```

### :floppy_disk: Datasets
https://huggingface.co/datasets/Zbalpha/shapes

### :briefcase: Checkpoints
https://huggingface.co/Zbalpha/geom_dist_ckpt

## :e-mail: Contact

Contact [Biao Zhang](mailto:biao.zhang@kaust.edu.sa) ([@1zb](https://github.com/1zb)) if you have any further questions. This repository is for academic research use only.

## :blue_book: Citation

```bibtex
@article{zhang2024geometry,
  title={Geometry Distributions},
  author={Zhang, Biao and Ren, Jing and Wonka, Peter},
  journal={arXiv preprint arXiv:2411.16076},
  year={2024}
}
```