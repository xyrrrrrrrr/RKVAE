import sys,pathlib,json,numpy as np,torch
from scipy.linalg import solve_discrete_are
sys.path[:0]=['train','utility']; from Learn_Koopman_with_KlinearEig import Network; from Utility import SinglePendulum
ROOT=pathlib.Path('results/e2_pendulum_norm_sampled_z_sigma002_theta2_eta001_200ep_gpu'); ckpt=ROOT/'dkuc_aligned_train/best.pt'; torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
d=torch.load(ckpt,map_location='cpu'); net=Network(d['layer'],22,1,device='cpu').double(); net.load_state_dict(d['model']); net.eval(); A=net.lA.weight.detach().numpy(); B=net.lB.weight.detach().numpy(); Q=np.zeros((22,22)); Q[0,0]=5*np.pi**2; Q[1,1]=.01*64; R=np.array([[.1*64]])
P=solve_discrete_are(A,B,Q,R); K=np.linalg.solve(R+B.T@P@B,B.T@P@A); env=SinglePendulum(); scale=np.array([np.pi,8.]); tau=.02
def psi(x):
 with torch.no_grad(): return net.encode(torch.as_tensor(np.array([(x[0]+np.pi)%(2*np.pi)-np.pi,x[1]])/scale)).numpy()
for sigma,tag in [(0.,'0'),(.001,'001'),(.002,'002'),(.005,'005')]:
 out=ROOT/f'control_dkuc_aligned_corrected_sigma{tag}'; out.mkdir(exist_ok=True); rows=[]
 for j in range(100):
  rng=np.random.default_rng(2+j); width=np.array([2.,2.]) if j%2 else np.ones(2); x=rng.uniform(-width,width); env.reset_state(x); xs=[x.copy()]; cost=0.; us=[]
  for k in range(200):
   z=psi(x); un=float((-K@(z-psi(np.zeros(2))).reshape(-1,1))[0,0]); u=np.clip(un*8.,-8,8); xn=env.step(np.array([u]))[0]+sigma*rng.standard_normal(2)*scale; env.reset_state(xn); th=((x[0]+np.pi)%(2*np.pi)-np.pi); cost+=5*th**2+.01*x[1]**2+.1*u*u; xs.append(xn.copy()); us.append(u); x=xn
  cost += 5*((x[0]+np.pi)%(2*np.pi)-np.pi)**2 + .01*x[1]**2
  tail=np.asarray(xs[-20:]); th=(tail[:,0]+np.pi)%(2*np.pi)-np.pi; m=float(np.mean(th**2+tail[:,1]**2)); rows.append({'trial':j,'cost':cost,'tail_m':m,'ms_success':bool(m<=tau)}); np.savez(out/f'trial_{j:03}.npz',states=np.asarray(xs),controls=np.asarray(us))
 met={'config':{'mode':'dkuc','sigma':sigma,'trials':100,'horizon':200,'seed':2,'control_r':.1,'initial_angle_limit':2.0,'checkpoint':str(ckpt)},'training_config':json.load(open(ROOT/'dkuc_aligned_train/config.json')),'checkpoint_epoch':d['epoch'],'controller_protocol':'discrete LQR on phi(x)-phi(0), physical input saturation, terminal state cost; no residual feedback','completion_rate':1.,'success_rate':float(np.mean([r['ms_success'] for r in rows])),'completed_cost_mean':float(np.mean([r['cost'] for r in rows])),'completed_cost_std':float(np.std([r['cost'] for r in rows])),'tail_m_mean':float(np.mean([r['tail_m'] for r in rows])),'tail_m_std':float(np.std([r['tail_m'] for r in rows])),'tau_m':tau,'trials':rows}; (out/'metrics.json').write_text(json.dumps(met,indent=2)); print(sigma,met['success_rate']*100,met['completed_cost_mean'],met['tail_m_mean'],flush=True)
