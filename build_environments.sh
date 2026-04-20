#!/usr/bin/env bash

name="cross_math"

while [[ $# -gt 0 ]]; do
    case ${1} in
        --name) name="${2}"; shift 2 ;;
        *) echo "Unknown argument: ${1}"; shift ;;
    esac
done


conda create -n "${name}" python=3.11 -y
conda activate "${name}"

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install triton==3.3.1

git clone https://github.com/fla-org/flash-linear-attention.git
cd flash-linear-attention
git checkout v0.4.2
pip install .
cd ..

pip install numpy==2.3.5 transformers==5.3.0 accelerate==1.13.0 datasets==4.7.0 einops ninja peft

pip install "setuptools<82"