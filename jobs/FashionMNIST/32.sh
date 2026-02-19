
# It is highly recommended to run all these experiments in parallel on a cluster.

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=32 > checkpoint/log/FashionMNIST/log-32.txt
