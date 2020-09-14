# Superpixels Generator

The superpixel generator for the support image format.

The fast way to generate superpixel with target images. It default run 
multiprocessing for the single node. Also, it supports distributed multiprocessing 
for multi nodes.

## Introduction

As explained by David Stutz (2017), "Superpixel group perceptually similar pixels to create visually meaningful entities while heavily reducing the number of primitives for subsequent processing steps. As of these properties, superpixel algorithms have received much attention since their naming in 2003. By today, publicly available superpixel algorithms have turned into standard tools in low-level vision." [1](https://arxiv.org/abs/1612.01601)

## Requirements

```
pip install -r requirements.txt
```

## Configuration

By using factory design pattern, you can configure your method in file [config_superpixel.json](./config_superpixel.json).

## Usage

```
usage: superpixel.py [-h] -i INPUT -o OUTPUT -m METHOD [--config]
                     [--multiprocessing-distributed]

Superpixels generator

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        please input the path of image folder e.g.
                        "image_folder/train"
  -o OUTPUT, --output OUTPUT
                        please output the path of image folder e.g.
                        "image_folder/train_sp"
  -m METHOD, --method METHOD
                        please select superpixel method: "watershed", "quick",
                        "fz", "slic"
  --config              use config in file ./config_superpixel.json to
                        generate superpixel based on select method
  --multiprocessing-distributed
                        use multi-processing distributed to generate
```

## License

[MIT License](./LICENSE)