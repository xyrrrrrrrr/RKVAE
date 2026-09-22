"""Print a reproducible command matrix; add --execute to run it sequentially."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',default='results/rkvae_paper')
p.add_argument('--systems',nargs='+',choices=['damping','franka'],default=['damping','franka'])
p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3,4,5])
p.add_argument('--device',default='cpu')
p.add_argument('--train-sigma',type=float,default=.002)
p.add_argument('--eval-sigmas',nargs='+',type=float,default=[0,.0005,.001,.002])
p.add_argument('--epochs',type=int,default=200)
p.add_argument('--patience-epochs',type=int,default=200)
p.add_argument('--activation',choices=['relu','tanh'],default='relu')
p.add_argument('--variant',choices=['joint','bilinear','lifted'],default='joint')
p.add_argument('--control-r',type=float,default=.1)
p.add_argument('--execute',action='store_true')
a=p.parse_args()
script=str(Path(__file__).with_name('rkvae.py').resolve())
for system in a.systems:
    for seed in a.seeds:
        folder=Path(a.root)/system/f'seed_{seed}'
        commands=[[sys.executable,script,'train','--system',system,'--seed',str(seed),'--device',a.device,'--sigma',str(a.train_sigma),'--epochs',str(a.epochs),'--patience-epochs',str(a.patience_epochs),'--activation',a.activation,'--variant',a.variant,'--output',str(folder/'train')]]
        if system == 'damping':
            commands.append([sys.executable,str(Path(__file__).with_name('select_control_checkpoint.py').resolve()),
                             '--train-dir',str(folder/'train'),'--output',str(folder/'control_validation')])
        for mode in ('lifted','bilinear'):
            checkpoint = folder/'control_validation'/f'deploy_{mode}.pt'
            for sigma in a.eval_sigmas:
                for task in (('eight','star') if system=='franka' else ('eight',)):
                    commands.append([sys.executable,script,'evaluate','--checkpoint',str(checkpoint),'--mode',mode,'--sigma',str(sigma),'--task',task,'--seed','10001','--horizon','3000' if system=='franka' else '200','--trials','20' if system=='franka' else '100','--control-r',str(a.control_r),'--output',str(folder/f'{mode}_{task}_sigma{sigma}')])
        for command in commands:
            print(shlex.join(command),flush=True)
            if a.execute:
                subprocess.run(command,check=True)
