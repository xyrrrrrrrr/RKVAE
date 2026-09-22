"""Evaluate the normalized DKUC Franka checkpoint on one noise-free task."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve_discrete_are
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'utility'),str(ROOT/'train')]
from rkvae import System, franka_reference
from Learn_Koopman_with_KlinearEig import Network

def main(a):
    torch.set_num_threads(1); d=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=d['config']; net=Network([14,128,128,128,20],34,7,device='cpu').double(); net.load_state_dict(d['model']); net.eval()
    system=System('franka'); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    input_scale=np.asarray(cfg['input_scale'], dtype=float)
    state_scale=np.r_[np.ones(7), np.full(7, 2.175)]
    Q=np.diag([1.]*10+[0.]*4); R=.1*np.eye(7)
    # DKUC is trained with normalized inputs u_norm=u_physical/input_scale.
    # Convert the physical penalty to the coordinates used by B before LQR.
    R_design=np.diag(input_scale) @ R @ np.diag(input_scale)
    Qz=np.zeros((34,34)); Qz[:14,:14]=np.diag(state_scale) @ Q @ np.diag(state_scale)
    A=net.lA.weight.detach().numpy(); B=net.lB.weight.detach().numpy()
    P=solve_discrete_are(A,B,Qz,R_design); K=np.linalg.solve(R_design+B.T@P@B,B.T@P@A)
    rows=[]
    try:
      for trial in range(a.trials):
        rng=np.random.default_rng(a.seed+trial); x=system.reset(rng,broad=trial%2==1)
        refs,targets=franka_reference(system,x,a.horizon,a.task)
        # Reference IK generation changes the simulator state; restore x before rollout.
        system.set(x)
        xs=[x.copy()]; us=[]; ee=[]; tg=[]; times=[]; cost=0.
        for k in range(a.horizon):
          ref=refs[k]; target=targets[k];
          start=time.perf_counter()
          with torch.no_grad():
            z=torch.cat((torch.as_tensor(system.norm(x)), net.encode_only(torch.as_tensor(system.norm(x))))).numpy()
            zr=torch.cat((torch.as_tensor(system.norm(ref)), net.encode_only(torch.as_tensor(system.norm(ref))))).numpy()
          u_normalized=-a.gain_scale*K@(z-zr)
          u=np.clip(input_scale*u_normalized,-system.limit,system.limit)
          ee.append(system.env.get_state()[:3]); tg.append(target); us.append(u.copy())
          err=x-ref; cost += float(err@Q@err+u@R@u)
          x,_=system.step(u,a.sigma,rng); xs.append(x.copy())
          times.append(time.perf_counter()-start)
        position=np.linalg.norm(np.asarray(ee)-np.asarray(tg),axis=1)
        rows.append({'trial':trial,'completed':True,'cost':cost,'position_rmse_m':float(np.sqrt(np.mean(position**2))),'position_max_error_m':float(position.max()),'position_tail_rmse_m':float(np.sqrt(np.mean(position[-20:]**2))), 'median_latency_ms':float(np.median(times)*1000), 'p95_latency_ms':float(np.quantile(times,.95)*1000), 'warm_median_latency_ms':float(np.median(times[10:])*1000 if len(times)>10 else np.median(times)*1000), 'warm_p95_latency_ms':float(np.quantile(times[10:],.95)*1000 if len(times)>10 else np.quantile(times,.95)*1000)})
        np.savez(out/f'trial_{trial:03}.npz',states=xs,controls=us,references=refs,ee=ee,targets=tg)
      summary={'config':vars(a),'training_config':cfg,'checkpoint_epoch':d['epoch'],'trials':rows,'completion_rate':1.,'success_rate':None,'completed_cost_mean':float(np.mean([r['cost'] for r in rows]))}
      (out/'metrics.json').write_text(json.dumps(summary,indent=2))
      with np.load(out/'trial_000.npz') as z:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(); ax.plot(z['targets'][:,1],z['targets'][:,2],label='Desired'); ax.plot(z['ee'][:,1],z['ee'][:,2],label='DKUC'); ax.set_xlabel('y (m)'); ax.set_ylabel('z (m)'); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/'control.png',dpi=160); plt.close(fig)
    finally: system.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--task',choices=['eight','star'],required=True); p.add_argument('--trials',type=int,default=20); p.add_argument('--horizon',type=int,default=3000); p.add_argument('--seed',type=int,default=2); p.add_argument('--sigma',type=float,default=0.0,help='Process noise during evaluation'); p.add_argument('--gain-scale',type=float,default=1.0); p.add_argument('--output',required=True); main(p.parse_args())
