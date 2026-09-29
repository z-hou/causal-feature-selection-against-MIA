"""Shared encoding, deterministic MLP training and RMIA numerical routines."""
import json
from decimal import Decimal
import numpy as np
import torch
from torch import nn

class Encoder:
    """Train-only encoding; an explicit column handles unseen categories."""
    def __init__(self, numeric):
        self.numeric = set(numeric)

    def fit(self, frame):
        self.columns, self.spec = list(frame.columns), {}
        for c in self.columns:
            if c in self.numeric:
                v = np.asarray(frame[c], dtype=float)
                valid = v[np.isfinite(v)]
                median = float(np.median(valid)) if len(valid) else 0.0
                v = np.where(np.isfinite(v), v, median)
                self.spec[c] = [median, float(v.mean()), float(v.std()) or 1.0]
            else:
                self.spec[c] = sorted(set(frame[c].fillna('__MISSING__').astype(str)))
        return self

    def transform(self, frame):
        parts = []
        for c in self.columns:
            if c in self.numeric:
                median, mean, std = self.spec[c]
                v = np.asarray(frame[c], dtype=float)
                parts.append(((np.where(np.isfinite(v), v, median) - mean) / std)[:, None])
            else:
                mapping = {v: i for i, v in enumerate(self.spec[c])}
                indices = [mapping.get(v, len(mapping)) for v in frame[c].fillna('__MISSING__').astype(str)]
                block = np.zeros((len(frame), len(mapping) + 1), dtype=np.float32)
                block[np.arange(len(frame)), indices] = 1
                parts.append(block)
        return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.ones((len(frame), 1), dtype=np.float32)

    def save(self, path):
        path.write_text(json.dumps(dict(columns=self.columns, numeric=sorted(self.numeric), spec=self.spec)))


def train_predict(train, labels, frames, classes, numeric, args, name, seed):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = Encoder(numeric).fit(train)
    x = torch.from_numpy(encoder.transform(train))
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    hidden = list(getattr(args, 'hidden', [512, 256, 128]))
    layers, previous = [], x.shape[1]
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, classes))
    model = nn.Sequential(*layers).to(args.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(args.device)), yb.to(args.device))
            loss.backward()
            optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f'{name}: epoch {epoch + 1}/{args.epochs}', flush=True)
    torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'input_dim': x.shape[1], 'classes': classes, 'hidden': hidden,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'encoder': dict(columns=encoder.columns, numeric=sorted(encoder.numeric), spec=encoder.spec),
                'training': dict(getattr(args, 'metadata', {}), epochs=args.epochs,
                                 batch_size=args.batch_size, seed=seed)}, args.output / f'{name}.pt')
    model.eval()
    predictions = []
    with torch.no_grad():
        for frame, truth in frames:
            chunks = []
            for start in range(0, len(frame), 2048):
                xb = torch.from_numpy(encoder.transform(frame.iloc[start:start + 2048])).to(args.device)
                # Convert on CPU for float64 support on MPS as well.
                logits = model(xb).cpu().double()
                logp = torch.log_softmax(logits, dim=1)
                yy = torch.as_tensor(np.asarray(truth[start:start + 2048]), dtype=torch.long)
                chunks.append(logp.gather(1, yy[:, None]).squeeze(1).numpy())
            predictions.append(np.concatenate(chunks))
    return predictions


def logmean(logp):
    return np.logaddexp.reduce(logp, axis=0) - np.log(logp.shape[0])


def score(target_x, reference_x, target_z, reference_z, a=0.3, gamma=2.0):
    """Finite log-probabilities in, fraction of strict LR > gamma out.
    x is OUT to all references; z is IN to half (no offline correction for z).
    """
    if not 0 <= a <= 1 or not np.isfinite(gamma) or gamma < 1:
        raise ValueError('Require 0 <= a <= 1 and finite gamma >= 1')
    target_x, reference_x, target_z, reference_z = [np.asarray(v, dtype=np.float64)
        for v in [target_x, reference_x, target_z, reference_z]]
    if (target_x.ndim != 1 or target_z.ndim != 1 or not len(target_z)
            or reference_x.ndim != 2 or reference_z.ndim != 2
            or reference_x.shape[1] != len(target_x) or reference_z.shape[1] != len(target_z)
            or reference_x.shape[0] != reference_z.shape[0] or reference_x.shape[0] == 0):
        raise ValueError('Invalid prediction shapes')
    for values in [target_x, target_z, reference_x, reference_z]:
        if not np.isfinite(values).all() or (values > 1e-10).any():
            raise ValueError('Expected finite log-probabilities <= 0')
    out = logmean(reference_x)
    corrected = out if a == 1 else np.logaddexp(np.log1p(a) + out, np.log1p(-a)) - np.log(2)
    ratio_z = np.sort(target_z - logmean(reference_z))
    return np.searchsorted(ratio_z, target_x - corrected - np.log(gamma), side='left') / len(ratio_z)


def auc(member, nonmember):
    """Tie-aware AUC used only to select a on auxiliary models."""
    ordered = np.sort(nonmember)
    return float(np.mean((np.searchsorted(ordered, member, 'left')
                          + np.searchsorted(ordered, member, 'right')) / (2 * len(ordered))))


def predict(directory, name, frame, labels):
    checkpoint = torch.load(directory / f'{name}.pt', map_location='cpu', weights_only=True)
    spec = checkpoint['encoder']
    encoder = Encoder(spec['numeric'])
    encoder.columns, encoder.spec = spec['columns'], spec['spec']
    layers, previous = [], checkpoint['input_dim']
    for width in checkpoint.get('hidden', [128, 64]):
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, checkpoint['classes']))
    model = nn.Sequential(*layers)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logps, correct = [], 0
    with torch.no_grad():
        for start in range(0, len(frame), 2048):
            logits = model(torch.from_numpy(encoder.transform(frame.iloc[start:start+2048]))).double()
            truth = torch.from_numpy(labels[start:start+2048])
            correct += int((logits.argmax(1) == truth).sum())
            logps.append(torch.log_softmax(logits,1).gather(1,truth[:,None]).squeeze(1).numpy())
    return np.concatenate(logps), correct/len(labels)


def best_attack_accuracy(member, nonmember):
    """Descriptive maximum over empirical thresholds, accepting entire ties."""
    thresholds = np.unique(np.concatenate([member, nonmember]))
    tp = len(member) - np.searchsorted(np.sort(member), thresholds, side='right')
    tn = np.searchsorted(np.sort(nonmember), thresholds, side='right')
    return float(max(np.max(tp + tn), len(member)) / (len(member) + len(nonmember)))


def tpr_at_fpr(member, nonmember, fpr):
    """Maximum empirical TPR under FP budget; whole ties, no interpolation."""
    member, nonmember = np.asarray(member), np.asarray(nonmember)
    if member.ndim != 1 or nonmember.ndim != 1 or not len(member) or not len(nonmember):
        raise ValueError('Expected nonempty one-dimensional score arrays')
    if not np.isfinite(member).all() or not np.isfinite(nonmember).all():
        raise ValueError('Nonfinite attack scores')
    if not 0 <= fpr < 1:
        raise ValueError('FPR must be in [0, 1)')
    budget = int(Decimal(str(fpr)) * len(nonmember))
    # The (budget+1)-th largest negative cannot be accepted, including all ties.
    boundary = np.sort(nonmember)[-(budget + 1)]
    return float(np.mean(member > boundary))


