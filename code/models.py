"""Model definitions for Diagnostics round 2 (target article 3135, MSCA-Mamba on PTB-XL).

Part A -- the reference architecture and every control the article reports.  The article
releases no code, so the layer layout is inferred from its text and pinned down by its
published parameter counts; `count_check()` asserts the five counts that can be matched
exactly (121,149 / 119,493 / 51,141 / 114,813 / 214,077 and the two SE placements, 121,133).

Part B -- our compact student and the large teachers.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _Mamba

    def make_mamba(d):
        return _Mamba(d_model=d, d_state=16, d_conv=4, expand=2)
    MAMBA_IMPL = "mamba_ssm (CUDA selective scan)"
except Exception:                                   # pure-PyTorch fallback, same parameters
    from mambapy.mamba import MambaBlock, MambaConfig

    def make_mamba(d):
        return MambaBlock(MambaConfig(d_model=d, n_layers=1, d_state=16, expand_factor=2,
                                      d_conv=4, pscan=True))
    MAMBA_IMPL = "mambapy (parallel scan)"


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- Part A
class SE(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        self.f1 = nn.Linear(c, hidden)
        self.f2 = nn.Linear(hidden, c)

    def forward(self, x):                           # (B, C, T)
        w = torch.sigmoid(self.f2(F.relu(self.f1(x.mean(-1)))))
        return x * w.unsqueeze(-1)


class SeqMamba(nn.Module):
    """x2 = LN(x1); H = Dropout(Mamba(x2)); Y = x1 + H   (article Eqs. 2-5)."""

    def __init__(self, d, p=0.1, bidirectional=False):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.f = make_mamba(d)
        self.b = make_mamba(d) if bidirectional else None
        self.do = nn.Dropout(p)
        self.out_dim = 2 * d if bidirectional else d

    def forward(self, x):                           # (B, T, C)
        z = self.ln(x)
        y = x + self.do(self.f(z))
        if self.b is None:
            return y
        yb = x + self.do(self.b(z.flip(1)).flip(1))
        return torch.cat([y, yb], -1)


class SeqConv(nn.Module):
    def __init__(self, d, c=50, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, c, 7, padding=3)
        self.bn = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, d, 7, padding=3)
        self.do = nn.Dropout(p)
        self.out_dim = d

    def forward(self, x):
        z = self.ln(x).transpose(1, 2)
        h = self.c2(F.relu(self.bn(self.c1(z)))).transpose(1, 2)
        return x + self.do(h)


class SeqRNN(nn.Module):
    def __init__(self, d, kind, hidden, p=0.1):
        super().__init__()
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(d, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d)
        self.do = nn.Dropout(p)
        self.out_dim = d

    def forward(self, x):
        h, _ = self.rnn(x)
        return x + self.do(self.proj(h))


class SeqTransformer(nn.Module):
    def __init__(self, d, ff=158, heads=4, p=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=p, batch_first=True,
                                                norm_first=True)
        self.out_dim = d

    def forward(self, x):
        T, d = x.shape[1], x.shape[2]
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=x.device) * (-math.log(10000.0) / d))
        pe = torch.zeros(T, d, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return self.layer(x + pe)


class MSCA(nn.Module):
    """Reference architecture.

    front : 3 x [Conv1d(12->32, k=7, dilation d) + BN + ReLU (+ SE, r=4)], d in {1,2,4}
            -> concat (96) -> 1x1 fusion conv (96->96) + ReLU
    seq   : residual sequence block (Mamba by default)
    head  : LayerNorm -> GAP || GMP (192) -> MLP 192-128-64-5
    """

    def __init__(self, seq="mamba", se="branch", multiscale=True, n_out=5, p_seq=0.1,
                 p_head=0.3):
        super().__init__()
        self.multiscale, self.se_mode = multiscale, se
        if multiscale:
            self.branches = nn.ModuleList(
                nn.Sequential(nn.Conv1d(12, 32, 7, padding=3 * d, dilation=d),
                              nn.BatchNorm1d(32), nn.ReLU()) for d in (1, 2, 4))
            self.bse = (nn.ModuleList(SE(32, 8) for _ in range(3)) if se == "branch" else None)
            self.fuse = nn.Conv1d(96, 96, 1)
        else:                                       # single receptive field, same width
            self.single = nn.Sequential(nn.Conv1d(12, 96, 7, padding=3, bias=False),
                                        nn.BatchNorm1d(96), nn.ReLU())
            self.bse = SE(96, 24) if se == "branch" else None
        self.pse = SE(96, 8) if se in ("postfusion", "postseq") else None
        d = 96
        self.seq = {"mamba": lambda: SeqMamba(d, p_seq),
                    "bimamba": lambda: SeqMamba(d, p_seq, bidirectional=True),
                    "conv": lambda: SeqConv(d, 50, p_seq),
                    "gru": lambda: SeqRNN(d, "gru", 99, p_seq),
                    "lstm": lambda: SeqRNN(d, "lstm", 83, p_seq),
                    "transformer": lambda: SeqTransformer(d, 160, 4, p_seq),
                    "none": lambda: None}[seq]()
        od = self.seq.out_dim if self.seq is not None else d
        self.ln_final = nn.LayerNorm(od)
        self.head = nn.Sequential(nn.Linear(2 * od, 128), nn.ReLU(), nn.Dropout(p_head),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_head),
                                  nn.Linear(64, n_out))

    def features(self, x):                          # x (B, 12, T)
        if self.multiscale:
            fs = [b(x) for b in self.branches]
            if self.bse is not None:
                fs = [s(f) for s, f in zip(self.bse, fs)]
            h = F.relu(self.fuse(torch.cat(fs, 1)))
        else:
            h = self.single(x)
            if self.bse is not None:
                h = self.bse(h)
        if self.se_mode == "postfusion":
            h = self.pse(h)
        h = h.transpose(1, 2)
        if self.seq is not None:
            h = self.seq(h)
        if self.se_mode == "postseq":
            h = self.pse(h.transpose(1, 2)).transpose(1, 2)
        return self.ln_final(h)

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(1), h.amax(1)], -1))


REF_CONFIGS = {                                     # name -> (kwargs, published parameter count)
    "msca_mamba": (dict(), 121149),
    "msca_noSE": (dict(se="none"), 119493),
    "ms_cnn": (dict(se="none", seq="none"), 51141),
    "single_scale": (dict(multiscale=False), 114813),
    "bimamba": (dict(seq="bimamba"), 214077),
    "se_postfusion": (dict(se="postfusion"), 121133),
    "se_postseq": (dict(se="postseq"), 121133),
    "m_conv": (dict(seq="conv"), 120653),
    "m_gru": (dict(seq="gru"), 120906),
    "m_lstm": (dict(seq="lstm"), 120953),
    "m_transformer": (dict(seq="transformer"), 121405),
}
EXACT = ["msca_mamba", "msca_noSE", "ms_cnn", "single_scale", "bimamba", "se_postfusion",
         "se_postseq", "m_gru", "m_lstm", "m_transformer"]     # m_conv is within 0.2%


def count_check(verbose=True):
    rows = []
    for k, (kw, pub) in REF_CONFIGS.items():
        n = n_params(MSCA(**kw))
        rows.append((k, n, pub, n - pub))
        if verbose:
            print(f"{k:16s} ours {n:7d}  published {pub:7d}  diff {n - pub:+d}")
        if k in EXACT:
            assert n == pub, (k, n, pub)
    return rows


# ----------------------------------------------------------------------------- Part B
class DSBlock(nn.Module):
    """Multi-scale depthwise-separable residual block: three dilated depthwise convolutions
    (d = 1, 2, 4) share one pointwise projection, so the multi-scale idea of the reference
    front-end costs 3*k*C + C*C weights instead of 3*k*C*C."""

    def __init__(self, c, k=7, dil=(1, 2, 4), p=0.1):
        super().__init__()
        self.dw = nn.ModuleList(nn.Conv1d(c, c, k, padding=(k // 2) * d, dilation=d, groups=c,
                                          bias=False) for d in dil)
        self.bn1 = nn.BatchNorm1d(c)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c)
        self.do = nn.Dropout(p)

    def forward(self, x):
        h = sum(d(x) for d in self.dw)
        h = F.silu(self.bn1(h))
        h = self.bn2(self.pw(h))
        return F.silu(x + self.do(h))


class Down(nn.Module):
    def __init__(self, ci, co, k=7, stride=2):
        super().__init__()
        self.c = nn.Conv1d(ci, co, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(co)

    def forward(self, x):
        return F.silu(self.bn(self.c(x)))


class NoisyOrHead(nn.Module):
    """Hierarchy-consistent output layer.

    The network scores the 23 diagnostic subclasses; a superclass is present when at least one
    of its subclasses is, so its probability is the noisy-OR of its children,
    p_S = 1 - prod_k (1 - p_k), i.e. log(1 - p_S) = -sum_k softplus(z_k).  The layer returns
    superclass *logits* so the rest of the pipeline is unchanged.  `mode='flat'` replaces it
    with an ordinary linear layer of the same input for the ablation.
    """

    def __init__(self, d_in, child_index, n_sub, mode="noisyor"):
        super().__init__()
        self.mode = mode
        self.sub = nn.Linear(d_in, n_sub)
        M = torch.zeros(len(child_index), n_sub)
        for s, kids in enumerate(child_index):
            M[s, kids] = 1
        self.register_buffer("M", M)
        if mode != "noisyor":
            self.sup = nn.Linear(d_in, len(child_index))

    def forward(self, h):
        zs = self.sub(h)                                        # (B, 23) subclass logits
        if self.mode != "noisyor":
            return self.sup(h), zs
        log_not = -(F.softplus(zs) @ self.M.t())                # log(1 - p_S) <= 0
        log_not = log_not.clamp(max=-1e-6)
        log_p = torch.log(-torch.expm1(log_not))                # log p_S, stable
        return log_p - log_not, zs                              # logit = log p - log(1-p)


class Student(nn.Module):
    """Compact student.  Stem (stride 2) -> multi-scale DS blocks -> stride 2 -> DS blocks ->
    bidirectional GRU on the 4x-shortened sequence -> GAP || GMP -> hierarchy head.
    Works on 100 Hz, 10 s, 12-lead input in millivolts (no per-record standardisation)."""

    def __init__(self, child_index, n_sub=23, c1=40, c2=64, hid=40, n1=1, n2=2, p=0.1,
                 head="noisyor", rnn=True):
        super().__init__()
        self.stem = Down(12, c1, 7, 2)
        self.s1 = nn.Sequential(*[DSBlock(c1, p=p) for _ in range(n1)])
        self.down = Down(c1, c2, 5, 2)
        self.s2 = nn.Sequential(*[DSBlock(c2, p=p) for _ in range(n2)])
        self.rnn = nn.GRU(c2, hid, batch_first=True, bidirectional=True) if rnn else None
        d = 2 * hid if rnn else c2
        self.ln = nn.LayerNorm(d)
        self.do = nn.Dropout(0.2)
        self.head = NoisyOrHead(2 * d, child_index, n_sub, head)

    def features(self, x):
        h = self.s2(self.down(self.s1(self.stem(x)))).transpose(1, 2)
        if self.rnn is not None:
            h, _ = self.rnn(h)
        h = self.ln(h)
        return torch.cat([h.mean(1), h.amax(1)], -1)

    def forward(self, x):
        return self.head(self.do(self.features(x)))


# ---- large from-scratch teacher (xresnet1d-style), any sampling rate
class ResBlock(nn.Module):
    def __init__(self, ci, co, k, stride):
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, k, stride=stride, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(co)
        self.c2 = nn.Conv1d(co, co, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(co)
        self.sc = None
        if stride != 1 or ci != co:
            self.sc = nn.Sequential(nn.AvgPool1d(stride, ceil_mode=True) if stride > 1
                                    else nn.Identity(), nn.Conv1d(ci, co, 1, bias=False),
                                    nn.BatchNorm1d(co))

    def forward(self, x):
        h = self.b2(self.c2(F.silu(self.b1(self.c1(x)))))
        return F.silu(h + (x if self.sc is None else self.sc(x)))


class BigResNet(nn.Module):
    def __init__(self, n_out, widths=(64, 128, 192, 256), blocks=(2, 2, 2, 2), k=9,
                 stem_stride=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, k, stride=stem_stride, padding=k // 2,
                                            bias=False), nn.BatchNorm1d(32), nn.SiLU(),
                                  nn.Conv1d(32, widths[0], k, padding=k // 2, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.SiLU())
        L, ci = [], widths[0]
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                L.append(ResBlock(ci, w, k, 2 if (j == 0 and i > 0) else 1))
                ci = w
        self.body = nn.Sequential(*L)
        self.do = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * ci, n_out)

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.fc(self.do(torch.cat([h.mean(-1), h.amax(-1)], 1)))


def ecgfounder(n_out, ckpt=None):
    """ECGFounder 12-lead Net1D (Li et al., pretrained on Harvard-Emory ECG Database; PTB-XL was
    not part of its pretraining data).  Architecture from the authors' repository."""
    import sys
    from project_paths import ROOT
    sys.path.insert(0, str(__import__("pathlib").Path(ROOT) / "data" / "weights" / "ecgfounder"))
    from net1d import Net1D
    m = Net1D(in_channels=12, base_filters=64, ratio=1,
              filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
              kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False,
              use_do=False, n_classes=n_out)
    info = "random init"
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k: v for k, v in sd.items() if not k.startswith("dense.")}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        info = f"loaded {len(sd)} tensors, missing {list(missing)}, unexpected {len(unexpected)}"
    return m, info


if __name__ == "__main__":
    print("Mamba implementation:", MAMBA_IMPL)
    count_check()
    kids = [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22]]
    for kw in (dict(), dict(head="flat"), dict(rnn=False)):
        s = Student(kids, **kw)
        print("student", kw, n_params(s), s(torch.randn(2, 12, 1000))[0].shape)
    print("BigResNet", n_params(BigResNet(28)))


# ----------------------------------------------------------------------------- seed-batched student
class StudentK(nn.Module):
    """K independent `Student` networks evaluated in one pass.

    Convolutions use `groups=K` (depthwise ones `groups=K*C`), batch-norm is per channel, and the
    GRU / LayerNorm / head are K separate modules, so the K networks share no weight and no
    statistic: gradients of network k's loss with respect to network j's weights are exactly
    zero (`selfcheck()` asserts this, and that `extract(k)` reproduces network k's outputs in a
    plain `Student`).  Input (B, K, 12, T): network k reads slice k, so each can have its own
    mini-batch.  This only removes the cost of K processes time-slicing one GPU.
    """

    def __init__(self, K, child_index, n_sub=23, c1=48, c2=96, hid=56, n1=1, n2=2, p=0.1,
                 head="noisyor", rnn=True):
        super().__init__()
        self.K, self.c2, self.kw = K, c2, dict(n_sub=n_sub, c1=c1, c2=c2, hid=hid, n1=n1, n2=n2,
                                               p=p, head=head, rnn=rnn)
        self.child_index = child_index

        def down(ci, co, k):
            return nn.ModuleDict(dict(c=nn.Conv1d(ci * K, co * K, k, stride=2, padding=k // 2,
                                                  bias=False, groups=K),
                                      bn=nn.BatchNorm1d(co * K)))

        def block(c):
            return nn.ModuleDict(dict(
                dw=nn.ModuleList(nn.Conv1d(c * K, c * K, 7, padding=3 * d, dilation=d,
                                           groups=c * K, bias=False) for d in (1, 2, 4)),
                bn1=nn.BatchNorm1d(c * K), pw=nn.Conv1d(c * K, c * K, 1, bias=False, groups=K),
                bn2=nn.BatchNorm1d(c * K)))
        self.stem = down(12, c1, 7)
        self.s1 = nn.ModuleList(block(c1) for _ in range(n1))
        self.down = down(c1, c2, 5)
        self.s2 = nn.ModuleList(block(c2) for _ in range(n2))
        self.p = p
        self.rnn = (nn.ModuleList(nn.GRU(c2, hid, batch_first=True, bidirectional=True)
                                  for _ in range(K)) if rnn else None)
        d = 2 * hid if rnn else c2
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(K))
        self.head = nn.ModuleList(NoisyOrHead(2 * d, child_index, n_sub, head) for _ in range(K))

    @staticmethod
    def _down(m, x):
        return F.silu(m["bn"](m["c"](x)))

    def _block(self, m, x):
        h = sum(d(x) for d in m["dw"])
        h = F.silu(m["bn1"](h))
        h = m["bn2"](m["pw"](h))
        return F.silu(x + F.dropout(h, self.p, self.training))

    def forward(self, x):                               # (B, K, 12, T)
        B, K = x.shape[0], self.K
        h = self._down(self.stem, x.reshape(B, K * 12, -1))
        for m in self.s1:
            h = self._block(m, h)
        h = self._down(self.down, h)
        for m in self.s2:
            h = self._block(m, h)
        h = h.view(B, K, self.c2, -1)
        sup, sub = [], []
        for k in range(K):
            g = h[:, k].transpose(1, 2)
            if self.rnn is not None:
                self.rnn[k].flatten_parameters()
                g, _ = self.rnn[k](g)
            g = self.ln[k](g)
            f = F.dropout(torch.cat([g.mean(1), g.amax(1)], -1), 0.2, self.training)
            a, b = self.head[k](f)
            sup.append(a)
            sub.append(b)
        return torch.stack(sup, 1), torch.stack(sub, 1)   # (B, K, 5), (B, K, 23)

    def extract(self, k):
        """Network k as a stand-alone `Student` (for deployment, timing and attribution)."""
        K = self.K
        s = Student(self.child_index, **self.kw)
        sd = {}

        def sl(t):                                      # slice a per-channel / per-filter tensor
            if t.dim() == 0:
                return t.clone()
            n = t.shape[0] // K
            return t[k * n:(k + 1) * n].clone()

        def cp_down(src, dst):
            sd[f"{dst}.c.weight"] = sl(src["c"].weight)
            for a in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                sd[f"{dst}.bn.{a}"] = sl(getattr(src["bn"], a))

        def cp_block(src, dst):
            for i, d in enumerate(src["dw"]):
                sd[f"{dst}.dw.{i}.weight"] = sl(d.weight)
            sd[f"{dst}.pw.weight"] = sl(src["pw"].weight)
            for bn in ("bn1", "bn2"):
                for a in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    sd[f"{dst}.{bn}.{a}"] = sl(getattr(src[bn], a))
        cp_down(self.stem, "stem")
        for i, m in enumerate(self.s1):
            cp_block(m, f"s1.{i}")
        cp_down(self.down, "down")
        for i, m in enumerate(self.s2):
            cp_block(m, f"s2.{i}")
        if self.rnn is not None:
            for n, v in self.rnn[k].state_dict().items():
                sd[f"rnn.{n}"] = v.clone()
        for n, v in self.ln[k].state_dict().items():
            sd[f"ln.{n}"] = v.clone()
        for n, v in self.head[k].state_dict().items():
            sd[f"head.{n}"] = v.clone()
        s.load_state_dict(sd)
        return s


def selfcheck():
    torch.manual_seed(0)
    kids = [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22]]
    for kw in (dict(), dict(head="flat"), dict(rnn=False)):
        m = StudentK(3, kids, **kw)
        assert n_params(m) == 3 * n_params(Student(kids, c1=48, c2=96, hid=56, **kw))
        m.train()
        for _ in range(3):                              # move the BN statistics off their init
            m(torch.randn(8, 3, 12, 1000))
        x = torch.randn(4, 3, 12, 1000)
        m.zero_grad()
        m(x)[0][:, 1].sum().backward()                  # loss of network 1 only
        for n, p in m.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            if n.split(".")[0] in ("rnn", "ln", "head"):
                own = n.split(".")[1] == "1"
                assert (g.abs().sum() > 0) == own or not own, n
                if not own:
                    assert float(g.abs().sum()) == 0, n
            else:
                c = g.shape[0] // 3
                assert float(g[:c].abs().sum()) == 0 and float(g[2 * c:].abs().sum()) == 0, n
                assert float(g[c:2 * c].abs().sum()) > 0, n
        m.eval()
        with torch.no_grad():
            a, b = m(x)
            for k in range(3):
                s = m.extract(k).eval()
                a1, b1 = s(x[:, k])
                assert torch.allclose(a[:, k], a1, atol=1e-4), (a[:, k] - a1).abs().max()
                assert torch.allclose(b[:, k], b1, atol=1e-4)
    print("StudentK selfcheck passed: networks are independent and extract() is exact")


if __name__ == "__main__":
    selfcheck()
