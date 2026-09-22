"""Reproducible process-noise training and control for the two paper systems."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'utility'), str(ROOT / 'train')]
from Learn_Kovae_with_KlinearEig import Network


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def tracking_error(x, reference, system):
    error = np.asarray(x, dtype=float) - np.asarray(reference, dtype=float)
    if system == 'damping':
        error = error.copy()
        error[..., 0] = wrap_angle(error[..., 0])
    return error


class System:
    def __init__(self, name):
        self.name = name
        if name == 'damping':
            from Utility import SinglePendulum
            self.env = SinglePendulum()
            self.scale = np.array([np.pi, 8.])
            self.center = np.zeros(2)
            self.limit = np.array([8.])
        else:
            from franka.franka_env import FrankaEnv
            self.env = FrankaEnv(render=False)
            # Model independent joint positions/velocities; EE is recomputed by FK.
            self.center = np.r_[self.env.reset_joint_state, np.zeros(7)]
            # Task-adapted scales: cover the notebook Eight/Star IK envelope
            # with margin while retaining useful normalized state excitation.
            self.scale = np.r_[np.array([0.60, 0.70, 0.70, 0.90, 0.40, 0.70, 0.40]),
                               np.ones(7)]
            self.limit = np.full(7, self.env.sat_val)
        self.n, self.m = len(self.scale), len(self.limit)

    def state(self):
        return self.env.s0.copy() if self.name == 'damping' else self.env.get_state()[7:]

    def set(self, x):
        if self.name == 'damping':
            self.env.reset_state(x.copy())
        else:
            import pybullet as pb
            for i in range(7):
                pb.resetJointState(self.env.robot, i, float(x[i]), float(x[7+i]), physicsClientId=self.env.client)
        return self.state()

    def reset(self, rng, broad=False, angle_limit=None):
        width = np.array([3., 2.]) if broad else np.ones(2)
        if self.name == 'damping' and angle_limit is not None:
            width[0] = min(width[0], angle_limit)
        if self.name == 'franka':
            width = np.r_[np.full(7, .15), np.zeros(7)]
        return self.set(self.center + rng.uniform(-width, width))

    def step(self, u, sigma, rng):
        self.env.step(np.clip(u, -self.limit, self.limit))
        v = sigma * rng.standard_normal(self.n) * self.scale
        return self.set(self.state() + v), v

    def norm(self, x):
        return (x - self.center) / self.scale

    def close(self):
        if self.name == 'franka':
            import pybullet as pb
            pb.disconnect(self.env.client)


def collect(system, count, steps, sigma, seed, normalize_states=True,
            franka_init_position_width=0.15, franka_init_velocity_width=0.0):
    rng = np.random.default_rng(seed)
    data = np.empty((steps+1, count, system.m+system.n))
    for j in range(count):
        if system.name == 'franka':
            init_width = np.r_[np.full(7, franka_init_position_width),
                               np.full(7, franka_init_velocity_width)]
            x = system.set(system.center + rng.uniform(-init_width, init_width))
        else:
            x = system.reset(rng, broad=True)
        for k in range(steps+1):
            u = rng.uniform(-system.limit, system.limit)
            state = system.norm(x) if normalize_states else x
            data[k, j] = np.r_[u, state]
            if k < steps:
                x, _ = system.step(u, sigma, rng)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite training data')
    return data


def data_coverage(data, system, normalized_states):
    """Return physical-coordinate coverage for the saved train/validation data."""
    values = np.asarray(data)
    states = values[..., system.m:]
    if normalized_states:
        states = states * system.scale + system.center
    controls = values[..., :system.m]
    return {
        'state_min': states.min(axis=(0, 1)).tolist(),
        'state_max': states.max(axis=(0, 1)).tolist(),
        'input_min': controls.min(axis=(0, 1)).tolist(),
        'input_max': controls.max(axis=(0, 1)).tolist(),
    }


def check_damping_coverage(coverage):
    """Reject a pendulum dataset that does not excite the intended operating box."""
    state_min = np.asarray(coverage['state_min'])
    state_max = np.asarray(coverage['state_max'])
    input_min = np.asarray(coverage['input_min'])
    input_max = np.asarray(coverage['input_max'])
    if state_min[0] > -2.9 or state_max[0] < 2.9:
        raise ValueError(f'Pendulum angle coverage is too narrow: [{state_min[0]}, {state_max[0]}]')
    if state_min[1] > -1.9 or state_max[1] < 1.9:
        raise ValueError(f'Pendulum velocity coverage is too narrow: [{state_min[1]}, {state_max[1]}]')
    if input_min[0] > -7.9 or input_max[0] < 7.9:
        raise ValueError(f'Pendulum input coverage is too narrow: [{input_min[0]}, {input_max[0]}]')


def losses(net, data, variant):
    x, u = data[..., net.u_dim:], data[:-1, ..., :net.u_dim]
    if getattr(net, 'normalize_inputs', True):
        input_scale = torch.as_tensor(net.input_scale, dtype=data.dtype, device=data.device) if hasattr(net, 'input_scale') else torch.ones(net.u_dim, dtype=data.dtype, device=data.device)
        u = u / input_scale
    mu_xz, sample, mu, lv, _ = net.encode(x)
    if variant == 'deterministic':
        sample = mu
    # Use the sampled latent state for both reconstruction and dynamics losses.
    z = torch.cat([x, sample], dim=-1)
    terms = {'rec': (net.decode(sample)-x).square().mean((-1, -2)).sum(),
             'kl': .5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()}
    b0_floor = getattr(net, 'b0_floor', 0.0)
    b0_floor_weight = getattr(net, 'b0_floor_weight', 0.0)
    if b0_floor > 0 and b0_floor_weight > 0:
        sigma_max = torch.linalg.matrix_norm(net.lB0.weight, ord=2)
        terms['b0_floor'] = b0_floor_weight * torch.relu(
            torch.as_tensor(b0_floor, dtype=data.dtype, device=data.device) - sigma_max
        ).square()
    else:
        terms['b0_floor'] = data.new_zeros(())
    if variant in ('deterministic', 'no_kl'):
        terms['kl'] = terms['kl'] * 0
    z_hat_bilinear_next = net.predict(z[:-1], u, 'bilinear')
    z_hat_lifted_next = net.predict(z[:-1], u, 'lifted')
    terms['bilinear'] = (z_hat_bilinear_next-z[1:]).square().mean((-1, -2)).sum()
    terms['lifted'] = (z_hat_lifted_next-z[1:]).square().mean((-1, -2)).sum()
    active = ['rec', 'kl'] + ([variant] if variant in ('bilinear', 'lifted') else ['bilinear', 'lifted'])
    return sum(terms[k] for k in active) + terms['b0_floor'], terms


def make_net(config, device):
    n, m, h = config['n'], config['m'], config['latent']
    net = Network([n,64,64,h], [h,64,64,n], n+h,m,n,device=device,
                  use_logvar=True, activation_name=config.get('activation', 'relu')).double()
    net.input_scale = torch.as_tensor(config.get('input_scale', [1.0] * m), dtype=torch.float64, device=net.device)
    net.normalize_inputs = bool(config.get('normalize_inputs', True))
    return net


def control_penalty(R, mode, input_scale=None):
    # The learned dynamics use normalized input u_norm=u_physical/input_scale.
    # Lifted control additionally divides by gx online; use the physical input
    # scale here instead of the previous hard-coded 0.1 lower bound.
    if mode == 'lifted':
        if input_scale is None:
            raise ValueError('input_scale is required for normalized lifted control')
        scale = np.diag(np.asarray(input_scale, dtype=float))
        return scale @ R @ scale
    if mode == 'bilinear' and input_scale is not None:
        scale = np.diag(np.asarray(input_scale, dtype=float))
        return scale @ R @ scale
    return R


def normalized_cost_matrices(system, cfg, mode, control_r):
    """Map physical Q/R into the normalized coordinates used by the model."""
    if system.name == 'damping':
        q_physical = np.diag([5., .01])
    else:
        q_physical = np.diag([10.] * 10 + [0.] * 4)
    r_physical = control_r * np.eye(system.m)
    state_scale = np.diag(system.scale) if cfg.get('normalize_states', True) else np.eye(system.n)
    input_scale = np.asarray(cfg.get('input_scale', system.limit), dtype=float)
    input_map = np.diag(input_scale) if cfg.get('normalize_inputs', True) else np.eye(system.m)
    q_model = state_scale @ q_physical @ state_scale
    r_model = input_map @ r_physical @ input_map
    return q_physical, r_physical, q_model, r_model, input_scale


def passive_capture(x, angle):
    """Damping-only local deployment: disable input outside its chosen region."""
    return angle is not None and abs(x[0]) >= angle


def capture_input(x, env, policy):
    """Physical-model assistance outside the local lifted region, not pure lifted."""
    if policy == 'passive':
        return np.zeros(1)
    # Natural-frequency PD gains; multiply by cos(theta), never divide at its zeros.
    omega = np.sqrt(env.g / env.l)
    acceleration = -omega**2 * x[0] - 2 * omega * x[1]
    return np.array([env.m * env.l * np.cos(x[0]) * acceleration])


def franka_reference(system, initial, horizon, task):
    """Generate the absolute Cartesian Eight/Star paths from the Franka notebook."""
    t = (1.6 + 0.02*np.linspace(0., 300., horizon + 1)
         if task == 'eight' else 0.02*np.linspace(0., 300., horizon + 1))
    if task == 'eight':
        a = 0.3
        x_path = np.full(horizon + 1, 0.3)
        y_path = a*np.cos(t)/(1. + np.sin(t)**2)
        z_path = 0.59 + 2.*a*np.sin(t)*np.cos(t)/(1. + np.sin(t)**2)
    else:
        center = np.array([0., 0.6]); radius = 0.3; theta0 = np.pi/10.
        eradius = np.tan(2.*theta0)*radius*np.cos(theta0)-radius*np.sin(theta0)
        points = np.zeros((11, 2))
        for i in range(5):
            theta = 2.*np.pi/5.*(i + 0.25)
            points[2*i] = [np.cos(theta)*radius, np.sin(theta)*radius] + center
            beta = 2.*np.pi/5.*(i + 0.75)
            points[2*i + 1] = [np.cos(beta)*eradius, np.sin(beta)*eradius] + center
        points[-1] = points[0]
        path = np.zeros((horizon + 1, 2)); each_num = int((horizon + 1 - 10)/9.5)
        for i in range(10):
            start = (each_num + 1)*i; path[start] = points[i]
            num = each_num if i != 9 else horizon + 1 - start - 1
            for j in range(num):
                alpha = (j + 1)/(each_num + 1)
                path[start + j + 1] = alpha*points[i + 1] + (1.-alpha)*points[i]
        x_path = np.full(horizon + 1, 0.3); y_path, z_path = path[:, 0], path[:, 1]
    references, targets = [], []
    for k in range(horizon + 1):
        target = np.array([x_path[k], y_path[k], z_path[k]])
        ref = system.center.copy(); ref[:7] = system.env.get_ik(target)
        system.set(ref); references.append(ref); targets.append(target)
    return np.asarray(references), np.asarray(targets)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    system = System(args.system)
    config = vars(args).copy() | {'n':system.n, 'm':system.m, 'scale':system.scale.tolist(), 'center':system.center.tolist(), 'input_scale':system.limit.tolist(), 'activation':args.activation}
    config.update(learning_rate=1e-3, lr_step_size=20, lr_gamma=0.9)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'best.pt').exists():
        raise FileExistsError('Use a new output directory; existing checkpoint is preserved')
    cache = Path(args.data_cache) if args.data_cache else out / 'dataset'
    cache.mkdir(parents=True, exist_ok=True)
    config['data_cache'] = str(cache)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    try:
        train_file, val_file = cache / 'train.npy', cache / 'validation.npy'
        expected = (args.sequence + 1, args.samples, system.m + system.n)
        expected_val = (args.sequence + 1, args.validation_samples, system.m + system.n)
        if train_file.exists() and val_file.exists():
            data = np.load(train_file, mmap_mode='r')
            val = np.load(val_file, mmap_mode='r')
            if data.shape != expected or val.shape != expected_val:
                raise ValueError(f'Cached dataset shape mismatch: {data.shape}, {val.shape}')
        else:
            data = collect(system,args.samples,args.sequence,args.sigma,args.seed,args.normalize_states,
                           args.franka_init_position_width, args.franka_init_velocity_width)
            val = collect(system,args.validation_samples,args.sequence,args.sigma,args.seed+100000,args.normalize_states,
                          args.franka_init_position_width, args.franka_init_velocity_width)
            np.save(train_file, data)
            np.save(val_file, val)
        config['train_coverage'] = data_coverage(data, system, args.normalize_states)
        config['validation_coverage'] = data_coverage(val, system, args.normalize_states)
        if system.name == 'damping':
            check_damping_coverage(config['train_coverage'])
            check_damping_coverage(config['validation_coverage'])
        (out/'config.json').write_text(json.dumps(config, indent=2))
    finally:
        system.close()
    net = make_net(config,args.device)
    net.b0_floor = float(config.get('b0_floor', 0.0))
    net.b0_floor_weight = float(config.get('b0_floor_weight', 0.0))
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    rng = np.random.default_rng(args.seed)
    best, best_bilinear, best_lifted = float('inf'), float('inf'), float('inf')
    patience_best, best_epoch = float('inf'), 0
    (out/'candidates').mkdir(exist_ok=True)
    with (out/'history.jsonl').open('w') as log:
        for epoch in range(1,args.epochs+1):
            net.train(); total = 0.; seen = 0
            order = rng.permutation(args.samples)
            for start in range(0,args.samples,args.batch_size):
                ids = order[start:start+args.batch_size]
                batch = torch.as_tensor(data[:,ids],dtype=torch.float64,device=args.device)
                loss,_ = losses(net,batch,args.variant)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item()*len(ids); seen += len(ids)
            assert seen == args.samples
            record = {'epoch':epoch,'train_loss':total/seen,'samples_seen':seen,
                      'learning_rate':optimizer.param_groups[0]['lr']}
            scheduler.step()
            if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
                net.eval(); scores = []
                with torch.no_grad():
                    for start in range(0,len(val[0]),args.batch_size):
                        batch = torch.as_tensor(val[:,start:start+args.batch_size],dtype=torch.float64,device=args.device)
                        _,parts = losses(net,batch,args.variant)
                        modes = [args.variant] if args.variant in ('bilinear','lifted') else ['bilinear','lifted']
                        scores.append((sum(parts[m] for m in modes).item(), batch.shape[1]))
                score = sum(s*n for s,n in scores)/sum(n for _,n in scores)
                bilinear_score = score if args.variant == 'bilinear' else 0.0
                lifted_score = score if args.variant == 'lifted' else 0.0
                # For joint training, accumulate branch-specific validation losses directly.
                if args.variant == 'joint':
                    branch_totals = {'bilinear': 0.0, 'lifted': 0.0}; branch_count = 0
                    with torch.no_grad():
                        for start in range(0,len(val[0]),args.batch_size):
                            batch = torch.as_tensor(val[:,start:start+args.batch_size],dtype=torch.float64,device=args.device)
                            _, parts = losses(net, batch, args.variant); n_batch = batch.shape[1]
                            branch_totals['bilinear'] += parts['bilinear'].item() * n_batch
                            branch_totals['lifted'] += parts['lifted'].item() * n_batch; branch_count += n_batch
                    bilinear_score = branch_totals['bilinear'] / branch_count
                    lifted_score = branch_totals['lifted'] / branch_count
                record['validation_prediction_loss'] = score
                record['validation_bilinear_loss'] = bilinear_score
                record['validation_lifted_loss'] = lifted_score
                saved = {'model':net.state_dict(),'config':config,'epoch':epoch,'score':score,
                         'validation_bilinear_loss':bilinear_score,'validation_lifted_loss':lifted_score}
                if score < best:
                    best = score
                    torch.save(saved, out/'best.pt')
                if bilinear_score < best_bilinear:
                    best_bilinear = bilinear_score
                    torch.save(saved, out/'best_bilinear.pt')
                if lifted_score < best_lifted:
                    best_lifted = lifted_score
                    torch.save(saved, out/'best_lifted.pt')
                if score < patience_best-args.min_delta:
                    patience_best, best_epoch = score, epoch
                torch.save(saved, out/'last.pt')
                torch.save(saved, out/'candidates'/f'epoch_{epoch:04d}.pt')
            log.write(json.dumps(record)+'\n'); log.flush(); print(record,flush=True)
            if epoch % args.eval_every_epochs == 0 and epoch-best_epoch >= args.patience_epochs:
                break
    plot_loss(out)


def evaluate(args):
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg = saved['config']; net = make_net(cfg,'cpu'); net.load_state_dict(saved['model']); net.eval()
    capture_angle = args.lifted_capture_angle
    if capture_angle is not None and (cfg['system'] != 'damping' or args.mode != 'lifted' or args.controller != 'rkvae'):
        raise ValueError('Passive capture is only supported for the damping lifted controller')
    if cfg['variant'] in ('bilinear','lifted') and args.mode != cfg['variant']:
        raise ValueError('Requested untrained branch')
    system = System(cfg['system']); out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows = []
    if args.resume and (out/'metrics.json').exists():
        previous = json.loads((out/'metrics.json').read_text())
        for key, value in vars(args).items():
            # Output paths may be relative/absolute across a resumed invocation;
            # all physical and controller settings remain strictly identical.
            if key not in ('resume', 'trials', 'output') and previous['config'].get(key) != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        rows = previous['trials']
        if len(rows) > args.trials or [r['trial'] for r in rows] != list(range(len(rows))):
            raise ValueError('Invalid trial sequence for resume')
        if not all((out/f'trial_{i:03}.npz').exists() for i in range(len(rows))):
            raise ValueError('Missing saved trial')
    elif list(out.glob('trial_*.npz')) or (out/'metrics.json').exists():
        system.close()
        raise FileExistsError('Use a new evaluation output directory')
    system.scale = np.array(cfg['scale']); system.center = np.array(cfg['center'])
    A = net.lA.weight.detach().numpy()
    B = net.lB.weight.detach().numpy()
    Q, R, Q_model, R_model, input_scale = normalized_cost_matrices(
        system, cfg, args.mode, args.control_r)
    Qz = np.zeros_like(A)
    Qz[:system.n,:system.n] = Q_model
    R_design = R_model
    def controller_state(x):
        return tracking_error(x, np.zeros(system.n), 'damping') if system.name == 'damping' and args.angle_coordinates == 'periodic' else x
    def lift(x):
        with torch.no_grad():
            state = system.norm(controller_state(x)) if cfg.get('normalize_states', True) else controller_state(x)
            xt = torch.as_tensor(state,dtype=torch.float64)
            mu,_,gx = net.encode_only(xt)
            return np.r_[xt.numpy(),mu.numpy()],gx.numpy()
    def gain(G):
        P = solve_discrete_are(A,G,Qz,R_design)
        return -np.linalg.solve(R_design+G.T@P@G,G.T@P@A)
    gain_start = time.perf_counter()
    fixed_gain = gain(B) if args.mode == 'lifted' and args.controller == 'rkvae' else None
    gain_setup_ms = (time.perf_counter() - gain_start) * 1000
    try:
        for trial in range(len(rows), args.trials):
            rng = np.random.default_rng(args.seed+trial)
            x = system.reset(rng, broad=trial % 2 == 1, angle_limit=args.initial_angle_limit); xs=[x.copy()]; us=[]; refs=[]; noises=[]; residuals=[]; times=[]; xis=[]; ee=[]; targets=[]
            residual = np.zeros(len(A)); cost=0.; failure=None
            if system.name == 'franka':
                reference_path, target_path = franka_reference(system, x, args.horizon, args.task)
                # Match the notebook protocol: initialize at the first IK
                # reference, then evolve from the same state used by the model.
                x = reference_path[0].copy()
                system.set(x)
            for k in range(args.horizon):
                ref = system.center.copy()
                control_ref = ref
                if system.name == 'franka':
                    ref = reference_path[k]; target = target_path[k]
                    # The learned model predicts x_{k+1}; match the notebook
                    # controller by feeding the next reference to feedback.
                    control_ref = reference_path[min(k + 1, args.horizon - 1)]
                    ee.append(system.env.get_state()[:3]); targets.append(target)
                try:
                    start=time.perf_counter(); z,gx=lift(x); zr,_=lift(control_ref)
                    G=B if args.mode=='lifted' else net.lB0.weight.detach().numpy()+net.lBbilinear.weight.detach().numpy().reshape(len(A),system.m,len(A))@z
                    if args.controller=='zero':
                        u=np.zeros(system.m)
                    elif passive_capture(controller_state(x), capture_angle):
                        u=capture_input(controller_state(x), system.env, args.lifted_capture_policy)
                    else:
                        K=fixed_gain if fixed_gain is not None else gain(G)
                        xi=args.gain_scale*(K@(z-zr))-args.eta*np.linalg.pinv(G)@residual
                        u=xi/gx if args.mode=='lifted' else xi
                    u=np.clip(input_scale*u,-system.limit,system.limit) if cfg.get('normalize_inputs', True) else np.clip(u,-system.limit,system.limit)
                    xi=(u/input_scale)*gx if args.mode=='lifted' and cfg.get('normalize_inputs', True) else (u*gx if args.mode=='lifted' else u)
                    control_elapsed = time.perf_counter()-start
                    xn,v=system.step(u,args.sigma,rng)
                    residual_start = time.perf_counter()
                    zn,_=lift(xn)
                    residual=zn-(A@z+G@xi)
                    if system.name == 'damping':
                        residual[0] = wrap_angle(residual[0] * system.scale[0]) / system.scale[0]
                    times.append(control_elapsed + time.perf_counter() - residual_start)
                    if not np.isfinite(xn).all(): raise FloatingPointError('Nonfinite state')
                    error = tracking_error(x, ref, system.name)
                    cost+=float(error@Q@error+u@R@u)
                    us.append(u); xis.append(xi); refs.append(ref); noises.append(v); residuals.append(residual); xs.append(xn.copy()); x=xn
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    failure=str(exc); break
            completed=len(us)==args.horizon and failure is None
            if completed:
                error = tracking_error(x, ref, system.name)
                cost+=float(error@Q@error)
            success=bool(completed and np.all(np.abs(wrap_angle(np.array(xs)[-20:,0]))<=.1)) if system.name=='damping' else None
            row={'trial':trial,'completed':completed,'failure':failure,'cost':cost,'success':success,'final_norm':float(np.linalg.norm(tracking_error(x, ref, system.name))) if refs else None,'median_latency_ms':float(np.median(times)*1000) if times else None,'p95_latency_ms':float(np.quantile(times,.95)*1000) if times else None}
            if ee: row['position_rmse_m']=float(np.sqrt(np.mean(np.sum((np.array(ee)-targets)**2,axis=1))))
            if ee:
                position_error = np.linalg.norm(np.array(ee)-targets, axis=1)
                row['position_max_error_m'] = float(position_error.max())
                row['position_tail_rmse_m'] = float(np.sqrt(np.mean(position_error[-20:]**2)))
            if system.name == 'damping':
                row['initial_group'] = 'broad' if trial % 2 else 'narrow'
                row['tail_velocity_rmse'] = float(np.sqrt(np.mean(np.array(xs)[-20:, 1]**2)))
                bad = np.flatnonzero(abs(wrap_angle(np.array(xs)[:, 0])) > .1)
                row['settling_time_s'] = float((bad[-1]+1)*system.env.dt) if len(bad) and bad[-1] < len(xs)-1 else (0. if not len(bad) else None)
            warmed_times = np.array(times[10:])
            row['warm_median_latency_ms'] = float(np.median(warmed_times)*1000) if len(warmed_times) else None
            row['warm_p95_latency_ms'] = float(np.quantile(warmed_times,.95)*1000) if len(warmed_times) else None
            rows.append(row)
            np.savez(out/f'trial_{trial:03}.npz',states=xs,controls=us,xi=xis,references=refs,noise=noises,residual=residuals,latency=times,ee=ee,targets=targets)
            summary={'config':vars(args),'training_config':cfg,'checkpoint_epoch':saved['epoch'],'trials':rows,'completion_rate':float(np.mean([r['completed'] for r in rows]))}
            summary['physical_input_penalty'] = R.tolist()
            summary['angle_metric'] = 'shortest_arc' if system.name == 'damping' else 'joint_coordinates'
            summary['design_input_penalty'] = R_design.tolist()
            summary['gain_setup_ms'] = gain_setup_ms
            summary['reference_protocol'] = 'precomputed_path_ik' if system.name == 'franka' else 'zero_angle_modulo_2pi'
            completed_cost=[r['cost'] for r in rows if r['completed']]
            summary['completed_cost_mean']=float(np.mean(completed_cost)) if completed_cost else None
            summary['completed_cost_std']=float(np.std(completed_cost)) if completed_cost else None
            summary['success_rate']=float(np.mean([r['success'] for r in rows])) if system.name=='damping' else None
            pending = out/'metrics.json.tmp'
            pending.write_text(json.dumps(summary,indent=2,allow_nan=False))
            pending.replace(out/'metrics.json')
            print(f'{out.name}: {trial+1}/{args.trials}', flush=True)
        plot(out,system.env.dt,system.name, args.task if system.name == 'franka' else None)
    finally:
        system.close()


def plot(out, dt, system, task=None, trial=0):
    """Match Control_RKVAE cell 7 and evaluate_Kovae_Franka1 cells 10/15.

    Each figure uses one trial and its own reference, as in the notebooks.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    with np.load(out / f'trial_{trial:03}.npz') as d:
        fig, ax = plt.subplots()
        if system == 'damping':
            observations = tracking_error(d['states'], np.zeros(2), 'damping').T
            time_history = np.arange(observations.shape[1]) * dt
            for i in range(len(observations)):
                ax.plot(time_history, observations[i], label=f'x{i}')
            ax.grid(True)
            ax.set_title('LQR Regulator')
        else:
            ax.plot(d['targets'][:, 1], d['targets'][:, 2], label='Desired')
            ax.plot(d['ee'][:, 1], d['ee'][:, 2], label='KP')
        ax.legend()
        fig.savefig(out / 'control.png', dpi=160)
        plt.close(fig)
    plot_error(out, dt, system)


def plot_error(out, dt, system):
    """Pointwise mean +/- one population SD across paired trial errors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    errors = []
    for path in sorted(out.glob('trial_*.npz')):
        with np.load(path) as d:
            if system == 'damping':
                # References are recorded before each transition (no final reference).
                error = tracking_error(d['states'][:len(d['references'])], d['references'], 'damping')
            else:
                error = np.linalg.norm(d['ee'] - d['targets'], axis=1)[:, None] if len(d['ee']) else np.empty((0, 1))
            if len(error):
                errors.append(error)
    if not errors:
        raise ValueError('No paired state/reference samples for error visualization')
    # Failed/short trials contribute only where observations actually exist.
    values = np.full((len(errors), max(map(len, errors)), errors[0].shape[1]), np.nan)
    for i, error in enumerate(errors):
        values[i, :len(error)] = error
    mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    count = np.sum(np.isfinite(values), axis=0)
    times = np.arange(len(mean)) * dt
    np.savez(out / 'error_statistics.npz', time=times, mean=mean, std=std, count=count)
    labels = ['Angle error (rad)', 'Angular velocity error (rad/s)'] if system == 'damping' else ['EE position error (m)']
    fig, axes = plt.subplots(len(labels), 1, figsize=(7, 3.5 * len(labels)), sharex=True, squeeze=False)
    for i, (ax, label) in enumerate(zip(axes[:, 0], labels)):
        ax.plot(times, mean[:, i], color='tab:blue', label='Mean')
        ax.fill_between(times, mean[:, i] - std[:, i], mean[:, i] + std[:, i],
                        color='tab:blue', alpha=.2, label='Mean +/- 1 SD')
        ax.set_ylabel(label)
        ax.grid(alpha=.3)
        ax.legend()
    n_min, n_max = int(count.min()), int(count.max())
    sample_label = str(n_min) if n_min == n_max else f'{n_min}-{n_max}'
    axes[0, 0].set_title(f'Tracking error (n={sample_label} trials)')
    axes[-1, 0].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out / 'error.png', dpi=160)
    plt.close(fig)


def plot_loss(out):
    """Save epoch-indexed training and validation losses from history.jsonl."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = []
    with (Path(out) / 'history.jsonl').open() as log:
        for line in log:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return
    epochs = np.array([r['epoch'] for r in records])
    train = np.array([r['train_loss'] for r in records])
    valid = [(r['epoch'], r['validation_prediction_loss'])
             for r in records if 'validation_prediction_loss' in r]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(epochs, train, color='tab:blue', linewidth=1.5, label='train total')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Training loss')
    axes[0].legend(loc='best')
    if valid:
        val_epochs, val_loss = zip(*valid)
        axes[1].plot(val_epochs, val_loss, color='tab:orange', linewidth=1.5,
                     marker='o', markersize=2, label='validation prediction')
        axes[1].legend(loc='best')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(out) / 'loss.png', dpi=160)
    plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--system',choices=['damping','franka'],default='damping')
    t.add_argument('--samples',type=int,default=50000); t.add_argument('--validation-samples',type=int,default=20000)
    t.add_argument('--sequence',type=int,default=15); t.add_argument('--epochs',type=int,default=100)
    t.add_argument('--eval-every-epochs',type=int,default=1); t.add_argument('--patience-epochs',type=int,default=40)
    t.add_argument('--min-delta',type=float,default=1e-4); t.add_argument('--batch-size',type=int,default=512)
    t.add_argument('--latent',type=int,default=20); t.add_argument('--activation',choices=['relu','tanh'],default='relu'); t.add_argument('--device',default='cpu')
    t.add_argument('--normalize-inputs',action='store_true',help='Normalize physical controls by input limits')
    t.add_argument('--data-cache',default=None,help='Directory for reusable train.npy and validation.npy datasets')
    t.add_argument('--franka-init-position-width',type=float,default=0.35)
    t.add_argument('--franka-init-velocity-width',type=float,default=0.5)
    t.add_argument('--no-normalize-states',dest='normalize_states',action='store_false',default=True,
                   help='Keep physical state coordinates instead of scaling by system.scale')
    t.add_argument('--b0-floor',type=float,default=0.1, help='Minimum sigma_max(B0) during training')
    t.add_argument('--b0-floor-weight',type=float,default=1.0, help='Weight for the B0 floor penalty')
    t.add_argument('--variant',choices=['joint','deterministic','no_kl','bilinear','lifted'],default='joint')
    e=sub.add_parser('evaluate'); e.add_argument('--checkpoint',required=True)
    e.add_argument('--mode',choices=['bilinear','lifted'],default='lifted'); e.add_argument('--controller',choices=['rkvae','zero'],default='rkvae')
    e.add_argument('--eta',type=float,default=0.01); e.add_argument('--task',choices=['eight','star'],default='eight')
    e.add_argument('--gain-scale',type=float,default=10., help='Multiply nominal LQR state feedback gain')
    e.add_argument('--control-r',type=float,default=.1, help='Input penalty R used for LQR design and reported cost')
    e.add_argument('--angle-coordinates',choices=['periodic','raw'],default='periodic',
                   help='Damping controller input chart; evaluation always uses shortest-arc errors')
    e.add_argument('--lifted-capture-angle',type=float,default=None,
                   help='Damping only: use capture control outside this local angle (radians)')
    e.add_argument('--lifted-capture-policy',choices=['passive','pd'],default='passive',
                   help='Outside-region control: zero input or physical-model PD assistance')
    e.add_argument('--initial-angle-limit',type=float,default=None,
                   help='Damping validation: cap |theta_0| for broad initial states')
    e.add_argument('--resume', action='store_true', help='Extend matching saved evaluation to --trials total')
    e.add_argument('--trials',type=int,default=100); e.add_argument('--horizon',type=int,default=3000)
    for command in (t,e):
        command.add_argument('--seed',type=int,default=1); command.add_argument('--sigma',type=float,default=.002 if command is t else 0.)
        command.add_argument('--threads',type=int,default=1); command.add_argument('--output',required=True)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    for key in ('samples','validation_samples','sequence','epochs','batch_size','eval_every_epochs','patience_epochs','trials','horizon','threads'):
        if hasattr(args,key) and getattr(args,key)<1: raise ValueError(f'{key} must be positive')
    if args.sigma<0: raise ValueError('sigma must be nonnegative')
    if hasattr(args,'eta') and args.eta<0: raise ValueError('eta must be nonnegative')
    if hasattr(args,'control_r') and args.control_r <= 0: raise ValueError('control-r must be positive')
    if getattr(args,'lifted_capture_angle',None) is not None and not 0 < args.lifted_capture_angle <= np.pi / 2:
        raise ValueError('lifted-capture-angle must be in (0, pi/2]')
    (train if args.command=='train' else evaluate)(args)
