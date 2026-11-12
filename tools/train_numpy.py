"""Pure NumPy end-to-end trainer for MLP and TCN architectures."""
import json
import numpy as np
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]

LABELS = ["ai cho", "bực mình", "ghen tị", "không cần", "xin lỗi"]
PARTS = [(0, 65, 65), (66, 131, 131), (132, 158, 158)]
STEPS = 32
FEATURES = 159

def resample_masked(seq, times, target_times=None):
    seq = np.asarray(seq, np.float32)
    times = np.asarray(times, np.float64)
    target = np.linspace(times[0], times[-1], STEPS) if target_times is None else np.asarray(target_times)
    hi = np.searchsorted(times, target, side='left').clip(0, len(times) - 1)
    lo = np.maximum(hi - 1, 0)
    exact = np.abs(times[hi] - target) < 1e-6
    lo = np.where(exact, hi, lo)
    nearest = np.where(np.abs(times[lo] - target) <= np.abs(times[hi] - target), lo, hi)
    gap = times[hi] - times[lo]
    ratio = np.divide(target - times[lo], gap, out=np.zeros_like(target), where=gap > 0)
    out = np.zeros((len(target), FEATURES), np.float32)
    for begin, end, mask in PARTS:
        both = (seq[lo, mask] > .5) & (seq[hi, mask] > .5) & (gap <= .26)
        out[both, begin:end] = seq[lo[both], begin:end] * (1 - ratio[both, None]) + seq[hi[both], begin:end] * ratio[both, None]
        out[both, mask] = 1
        close = (~both) & (seq[nearest, mask] > .5) & (np.abs(times[nearest] - target) <= .045)
        out[close, begin:end] = seq[nearest[close], begin:end]
        out[close, mask] = 1
    for a, b, mask in [(63, 65, 65), (129, 131, 131)]:
        out[:, a:b] *= out[:, 158, None]
    return out

def augmentation(seq, times, rng):
    increments = rng.uniform(.85, 1.15, 31)
    grid = np.r_[0, np.cumsum(increments)]
    grid = times[0] + grid / grid[-1] * (times[-1] - times[0])
    x = resample_masked(seq, times, grid)
    for begin, end, mask in PARTS:
        seen = x[:, mask] > .5
        x[seen, begin:end] += rng.normal(0, .006, (int(seen.sum()), end - begin))
    for begin in [0, 66]:
        x[:, begin:begin + 3] = 0
    x[:, [63, 64, 129, 130]] *= rng.uniform(.96, 1.04)
    return x

def normalize(x, mean, std):
    x = np.asarray(x, np.float32)
    out = (x - mean) / std
    for begin, end, mask in PARTS:
        out[..., begin:end] *= x[..., mask, None]
        out[..., mask] = x[..., mask]
    for begin, end in [(63, 65), (129, 131)]:
        out[..., begin:end] *= x[..., 158, None]
    return np.clip(out, -8, 8).astype(np.float32)

def prepare_augmented_dataset(manifest, seed=20261103):
    rng = np.random.default_rng(seed)
    raw = []
    y = []
    for r in manifest:
        z = np.load(ROOT / f"data/features/{r['id']}.npz")
        x_raw = z["features"]
        t_raw = z["times"]
        original = resample_masked(x_raw, t_raw)
        raw.append(original)
        y.append(LABELS.index(r["label"]))
        for _ in range(20):
            raw.append(augmentation(x_raw, t_raw, rng))
            y.append(LABELS.index(r["label"]))
            
    originals = np.array([raw[i * 21] for i in range(len(manifest))])
    mean = originals.mean(axis=(0, 1))
    std = np.maximum(originals.std(axis=(0, 1)), 0.2)
    for _, _, mask in PARTS:
        mean[mask] = 0
        std[mask] = 1
        
    x_all = normalize(np.array(raw), mean, std)
    y_all = np.asarray(y, np.int32)
    return x_all, y_all, mean, std

# -------------------------------------------------------------
# Pure NumPy MLP Training
# -------------------------------------------------------------
def train_mlp_numpy(x_all, y_all, num_classes=5, seed=20261103, epochs=500, lr=0.001):
    rng = np.random.default_rng(seed)
    N = len(x_all)
    x_flat = x_all.reshape((N, -1)) # (N, 5088)
    
    # Initialize params
    w0 = (rng.normal(0, np.sqrt(2 / 5088), (5088, 96))).astype(np.float32)
    b0 = np.zeros(96, dtype=np.float32)
    w1 = (rng.normal(0, np.sqrt(2 / 96), (96, 48))).astype(np.float32)
    b1 = np.zeros(48, dtype=np.float32)
    w2 = (rng.normal(0, np.sqrt(2 / 48), (48, num_classes))).astype(np.float32)
    b2 = np.zeros(num_classes, dtype=np.float32)
    
    params = {'w0': w0, 'b0': b0, 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    
    batch_size = 64
    for k in range(1, epochs + 1):
        idx = rng.integers(0, N, batch_size)
        bx = x_flat[idx]
        by = y_all[idx]
        
        # Forward
        h0 = np.maximum(bx @ params['w0'] + params['b0'], 0)
        h1 = np.maximum(h0 @ params['w1'] + params['b1'], 0)
        logits = h1 @ params['w2'] + params['b2']
        
        # Softmax & loss
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        
        # Label smoothing 0.02
        target = np.full_like(probs, 0.02 / num_classes)
        target[np.arange(batch_size), by] += 0.98
        
        dlogits = (probs - target) / batch_size
        
        # Backward
        grads = {}
        grads['w2'] = h1.T @ dlogits + 2e-5 * params['w2']
        grads['b2'] = dlogits.sum(axis=0)
        
        dh1 = (dlogits @ params['w2'].T) * (h1 > 0)
        grads['w1'] = h0.T @ dh1 + 2e-5 * params['w1']
        grads['b1'] = dh1.sum(axis=0)
        
        dh0 = (dh1 @ params['w1'].T) * (h0 > 0)
        grads['w0'] = bx.T @ dh0 + 2e-5 * params['w0']
        grads['b0'] = dh0.sum(axis=0)
        
        # Adam step
        for key in params:
            g = grads[key]
            m[key] = 0.9 * m[key] + 0.1 * g
            v[key] = 0.999 * v[key] + 0.001 * (g * g)
            m_hat = m[key] / (1.0 - 0.9 ** k)
            v_hat = v[key] / (1.0 - 0.999 ** k)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
    return params

# -------------------------------------------------------------
# Pure NumPy TCN Training (Conv Layers + Feature Pooling + MLP Head)
# -------------------------------------------------------------
def temporal_conv(x, w, b, dilation):
    result = np.zeros((len(x), w.shape[-1]), np.float32)
    size = w.shape[0]
    pad = dilation * (size - 1) // 2
    padded = np.pad(x, ((pad, pad), (0, 0)))
    for k in range(size):
        result += padded[k * dilation:k * dilation + len(x)] @ w[k]
    return result + b

def forward_tcn_sample(x_seq, params):
    # x_seq: (32, 159)
    h0 = np.maximum(x_seq @ params['w0'] + params['b0'], 0) # (32, 48)
    h1 = np.maximum(temporal_conv(h0, params['w1'], params['b1'], 1) + h0, 0) # (32, 48)
    h2 = np.maximum(temporal_conv(h1, params['w2'], params['b2'], 2) + h1, 0) # (32, 48)
    
    mask = (x_seq[:, 65] + x_seq[:, 131]) > 0
    if mask.any():
        pooled = np.r_[h2[mask].mean(axis=0), h2[mask].max(axis=0)]
    else:
        pooled = np.zeros(h2.shape[1] * 2, np.float32)
        
    h3 = np.maximum(pooled @ params['w3'] + params['b3'], 0)
    logits = h3 @ params['w4'] + params['b4']
    return h0, h1, h2, mask, pooled, h3, logits

def train_tcn_numpy(x_all, y_all, base_params=None, num_classes=5, seed=20261103, epochs=500, lr=0.001):
    rng = np.random.default_rng(seed)
    N = len(x_all)
    
    if base_params is None:
        w0 = rng.normal(0, np.sqrt(2 / 159), (159, 48)).astype(np.float32)
        b0 = np.zeros(48, dtype=np.float32)
        w1 = rng.normal(0, np.sqrt(2 / (3 * 48)), (3, 48, 48)).astype(np.float32)
        b1 = np.zeros(48, dtype=np.float32)
        w2 = rng.normal(0, np.sqrt(2 / (3 * 48)), (3, 48, 48)).astype(np.float32)
        b2 = np.zeros(48, dtype=np.float32)
        w3 = rng.normal(0, np.sqrt(2 / 96), (96, 48)).astype(np.float32)
        b3 = np.zeros(48, dtype=np.float32)
        w4 = rng.normal(0, np.sqrt(2 / 48), (48, num_classes)).astype(np.float32)
        b4 = np.zeros(num_classes, dtype=np.float32)
        params = {'w0': w0, 'b0': b0, 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3, 'w4': w4, 'b4': b4}
    else:
        params = {k: v.copy() for k, v in base_params.items()}
        
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    
    batch_size = 64
    for step in range(1, epochs + 1):
        idx = rng.integers(0, N, batch_size)
        
        # Compute pooled representations
        batch_pooled = []
        batch_y = y_all[idx]
        
        # We optimize the classification layers (w3, b3, w4, b4) using full gradient
        for i in idx:
            _, _, _, _, pooled, _, _ = forward_tcn_sample(x_all[i], params)
            batch_pooled.append(pooled)
            
        bx_pooled = np.array(batch_pooled) # (64, 96)
        
        h3 = np.maximum(bx_pooled @ params['w3'] + params['b3'], 0) # (64, 48)
        logits = h3 @ params['w4'] + params['b4'] # (64, 5)
        
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        
        target = np.full_like(probs, 0.02 / num_classes)
        target[np.arange(batch_size), batch_y] += 0.98
        
        dlogits = (probs - target) / batch_size
        
        gw4 = h3.T @ dlogits + 2e-5 * params['w4']
        gb4 = dlogits.sum(axis=0)
        
        dh3 = (dlogits @ params['w4'].T) * (h3 > 0)
        gw3 = bx_pooled.T @ dh3 + 2e-5 * params['w3']
        gb3 = dh3.sum(axis=0)
        
        # Adam step on w3, b3, w4, b4
        for k_name, grad in [('w4', gw4), ('b4', gb4), ('w3', gw3), ('b3', gb3)]:
            m[k_name] = 0.9 * m[k_name] + 0.1 * grad
            v[k_name] = 0.999 * v[k_name] + 0.001 * (grad * grad)
            m_hat = m[k_name] / (1.0 - 0.9 ** step)
            v_hat = v[k_name] / (1.0 - 0.999 ** step)
            params[k_name] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
    return params

if __name__ == "__main__":
    print("Testing pure NumPy training modules...")
    with open(ROOT / "data/manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    print(f"Loaded {len(m)} manifest samples.")
    x_all, y_all, mean, std = prepare_augmented_dataset(m)
    print(f"Generated {len(x_all)} augmented sequences.")
    
    # Train MLP
    t0 = time.time()
    mlp_p = train_mlp_numpy(x_all, y_all)
    print(f"Trained MLP in {time.time()-t0:.2f}s!")
    
    # Train TCN
    t0 = time.time()
    base_tcn = dict(np.load(ROOT / "models/tcn.npz"))
    tcn_p = train_tcn_numpy(x_all, y_all, base_params=base_tcn)
    print(f"Trained TCN in {time.time()-t0:.2f}s!")
