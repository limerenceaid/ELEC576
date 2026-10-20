#!/bin/bash
set -e
case "$1" in
  a)
    python -u mnist_dcn.py --name baseline --lr 1e-3
    python -u mnist_dcn.py --name act_tanh --act tanh --lr 1e-3
    python -u mnist_dcn.py --name act_sigmoid --act sigmoid --lr 1e-3
    python -u mnist_dcn.py --name act_leaky_relu --act leaky_relu --lr 1e-3
    python -u mnist_dcn.py --name adam_lr1e-2 --lr 1e-2
    ;;
  b)
    python -u mnist_dcn.py --name sgd --opt sgd --lr 1e-2
    python -u mnist_dcn.py --name sgd_xavier --opt sgd --init xavier --lr 1e-2
    python -u mnist_dcn.py --name sgd_kaiming --opt sgd --init kaiming --lr 1e-2
    python -u mnist_dcn.py --name momentum --opt momentum --lr 1e-2
    python -u mnist_dcn.py --name adagrad --opt adagrad --lr 1e-2
    ;;
esac
