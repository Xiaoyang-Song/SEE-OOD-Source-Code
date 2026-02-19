import os
import numpy as np

# Configuration
EXP_DSET = 'MNIST'
# EXP_DSET = 'MNIST-FashionMNIST'
# EXP_DSET = 'FashionMNIST'
# EXP_DSET = 'FashionMNIST-R2'
# EXP_DSET = 'CIFAR10-SVHN'
# EXP_DSET = 'SVHN'
# EXP_DSET = 'SVHN-R2'
# EXP_DSET = '3DPC-R1'
# EXP_DSET = '3DPC-R2'

# N = [100, 200, 500, 1000, 1500, 2000]
N = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

CMD_ONLY = False


for n in N:
    print(f"bash jobs/{EXP_DSET}/{n}.sh")

if not CMD_ONLY:
    print("Generating job files...")
    # Create logging directory
    log_path = os.path.join('checkpoint', 'log', EXP_DSET)
    os.makedirs(log_path, exist_ok=True)

    for n in N:
        # Create job directory
        job_path = os.path.join('jobs', EXP_DSET)
        os.makedirs(job_path, exist_ok=True)
        # Declare job name
        filename = os.path.join('jobs', EXP_DSET, f"{n}.sh")
        # Write files
        f = open(filename, 'w')
        f.write(f"python3 main/main_ood.py --config=config/GAN/OOD-GAN-{EXP_DSET}.yaml --n_ood={n} > checkpoint/log/{EXP_DSET}/log-{n}.txt\n")
        
        f.close()
