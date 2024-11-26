# Geometry Distributions

### [Project Page](https://1zb.github.io/GeomDist/) | [Paper (arXiv)](https://arxiv.org/abs/2411.16076)

### :bullettrain_front: Training

```
torchrun --nproc_per_node=4 main.py --blr 5e-7 --output_dir output/loong --log_dir output/loong --data_path shapes/loong.obj
```

### :balloon: Inference

```
python infer.py --pth output/loong/checkpoint-999.pth --target Gaussian --num-steps 64 --output samples/loong --N 10000000
```

### :floppy_disk: Datasets
https://huggingface.co/datasets/Zbalpha/shapes

### :briefcase: Checkpoints
https://huggingface.co/Zbalpha/geom_dist_ckpt

## :e-mail: Contact

Contact [Biao Zhang](mailto:biao.zhang@kaust.edu.sa) ([@1zb](https://github.com/1zb)) if you have any further questions. This repository is for academic research use only.

## :blue_book: Citation

```bibtex
@misc{zhang2024geometrydistributions,
      title={Geometry Distributions}, 
      author={Biao Zhang and Jing Ren and Peter Wonka},
      year={2024},
      eprint={2411.16076},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.16076}, 
}
```