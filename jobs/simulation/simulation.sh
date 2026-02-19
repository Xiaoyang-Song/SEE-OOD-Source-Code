
# To re-train simulation models, run the following commands:

python3 simulation.py --mode=G --config=config/simulation/setting_1_config.yaml

python3 simulation.py --mode=R --config=config/simulation/run_config.yaml --JID=I-FINAL --n_ood=2 --h=128 --beta=1 \
    --w_ce=1 --w_ood=1 --w_z=0.001 --wood_lr=0.001 --d_lr=0.0001 --g_lr=0.0001 --bsz_tri=256 --bsz_val=256 --bsz_ood=2 --n_d=2 --n_g=1

python3 simulation.py --mode=R --config=config/simulation/run_config.yaml --JID=II-FINAL --n_ood=2 --h=128 --beta=1 --w_ce=1 --w_ood=1 --w_z=100 \
    --wood_lr=0.001 --d_lr=0.0001 --g_lr=0.001 --bsz_tri=256 --bsz_val=256 --bsz_ood=2 --n_d=1 --n_g=3


