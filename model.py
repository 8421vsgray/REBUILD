import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ot_torch import Partial_ot, entropic_wasserstein

L2norm = nn.functional.normalize

class CorreGenBaselineModel(torch.nn.Module):
    def __init__(self, n_views, layer_dims, temperature, n_classes, drop_rate=0.5,
                    beta=0.15, reg=0.05, rho=0.2):
        super(CorreGenBaselineModel, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.beta = beta
        self.reg = reg
        self.rho = rho
        self.temperature = temperature

        self.online_encoder = nn.ModuleList(
            [FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)]
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cross_view_decoder = nn.ModuleList(
            [MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)]
        )

        self.cl = MaxLikelihoodLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(
                self.online_encoder[i].parameters(), self.target_encoder[i].parameters()
            ):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def realign(self,z_t,p):
        N = z_t[0].shape[0]
        z_t_n = [L2norm(z_t[i]) for i in range(self.n_views)]
        p_n = [L2norm(p[i]) for i in range(self.n_views)]
        res_idx = torch.zeros((self.n_views - 1, N), dtype=torch.long)
        res = [z_t_n[0].detach()]
        z0_t = z_t_n[0]
        for i in range(1,self.n_views):
            tmp = p_n[i]
            sim = z0_t @ tmp.T
            idx = sim.argmax(1)
            res_idx[i-1] = idx.cpu()
            select_matrix = F.one_hot(idx,num_classes=N).float()
            z_tmp = select_matrix @ z_t_n[i]
            res.append(z_tmp)
        return res, res_idx

    @torch.no_grad()
    def joint_dist_estimation(self, z_t, z, p, training_state, include_inter=True):
        dist = training_state["dist"]
        mp_inter = []
        mp_intra = []
        if dist is not None:
            dist = dist.cuda()
            for i in range(self.n_views):
                # Normalize the marginal distribution to ensure it sums to 1
                dist[i] /= dist[i].sum()
        eps = 1e-13
        z_t = [L2norm(z_t[i]) for i in range(self.n_views)]
        z = [L2norm(z[i]) for i in range(self.n_views)]
        p = [L2norm(p[i]) for i in range(self.n_views)]
        bs = z[0].shape[0]

        for i in range(self.n_views):
            for j in range(self.n_views):
                if i == j:
                    cost_intra_cross  = 1.0 - z[i] @ z_t[i].t()
                    if dist is not None:
                        mp = entropic_wasserstein(cost_intra_cross,p=dist[i],q=dist[i],reg=self.reg)
                        mp /= dist[i].unsqueeze(1)
                    else:
                        mp = entropic_wasserstein(cost_intra_cross, reg=self.reg)
                        mp *= bs
                    mp = torch.eye(mp.shape[0], device=mp.device)  * (1 - self.beta) + mp * self.beta
                    mp_intra.append(mp)
                elif include_inter:
                    cost_inter_cross = 1.0 - p[i] @ z_t[j].t()
                    if dist is not None:
                        mp = Partial_ot(cost_inter_cross,p=dist[i],q=dist[j],reg=self.reg,rho=self.rho)
                        mp = torch.div(mp, 1 - self.rho)
                        mp /= dist[i].unsqueeze(1)
                    else:
                        mp = Partial_ot(cost_inter_cross,reg=self.reg,rho=self.rho)
                        mp = torch.div(mp, 1 - self.rho)
                        mp *= bs
                    mp_inter.append(mp)
        return mp_inter, mp_intra

    @staticmethod
    def _validate_external_targets(external_targets, batch_size, device, dtype):
        """Validate externally supplied row/column-stochastic pseudo-labels."""
        if not isinstance(external_targets, dict):
            raise TypeError("external_targets must be a dict with P12 and P21")
        if set(external_targets) != {"P12", "P21"}:
            raise ValueError("external_targets must contain exactly P12 and P21")

        validated = {}
        for name in ("P12", "P21"):
            target = external_targets[name]
            if not torch.is_tensor(target):
                raise TypeError("{} must be a torch.Tensor".format(name))
            if target.shape != (batch_size, batch_size):
                raise ValueError(
                    "{} must have shape ({}, {}), got {}".format(
                        name, batch_size, batch_size, tuple(target.shape)
                    )
                )
            if target.device != device:
                raise ValueError(
                    "{} must be on {}, got {}".format(name, device, target.device)
                )
            if target.dtype != dtype:
                raise ValueError(
                    "{} must have dtype {}, got {}".format(
                        name, dtype, target.dtype
                    )
                )
            if not target.is_floating_point():
                raise TypeError("{} must use a floating-point dtype".format(name))
            if target.requires_grad:
                raise ValueError("{} must not require gradients".format(name))
            if not torch.isfinite(target).all():
                raise ValueError("{} contains NaN or Inf".format(name))
            if (target < 0).any():
                raise ValueError("{} must be non-negative".format(name))

            ones = torch.ones(batch_size, device=device, dtype=dtype)
            if not torch.allclose(target.sum(dim=1), ones, atol=1e-5, rtol=1e-3):
                raise ValueError("{} rows must sum to 1".format(name))
            if not torch.allclose(target.sum(dim=0), ones, atol=1e-5, rtol=1e-3):
                raise ValueError("{} columns must sum to 1".format(name))
            validated[name] = target.detach()
        return validated["P12"], validated["P21"]

    @torch.no_grad()
    def extract_feature(self, data):
        z = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        res, idx = self.realign(z,p)
        z = [L2norm(z[i]) for i in range(self.n_views)]
        return res, idx, z

    @torch.no_grad()
    def forward_features(self, data):
        z = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        p = [L2norm(p[i]) for i in range(self.n_views)]
        return p

    def compute_training_features(self, data, momentum=None, update_target=True):
        """Compute one training-state feature tuple for STRIDE supervision.

        The target branch stays in the caller's current training mode.  The
        no-grad context only prevents target features from entering the
        optimizer graph; it does not disable target Dropout or BatchNorm.
        """
        if update_target:
            if momentum is None:
                raise ValueError("momentum is required when update_target=True")
            self._update_target_branch(momentum)
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        with torch.no_grad():
            z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        return z, p, z_t

    def loss_from_training_features(self, z, p, z_t, inter_targets):
        """Consume cross-view labels using an already-computed feature tuple."""
        if len(z) != 2 or len(p) != 2 or len(z_t) != 2:
            raise ValueError("STRIDE training currently requires exactly two views")
        if len(inter_targets) != 2:
            raise ValueError("two cross-view targets are required")
        batch_size = z[0].shape[0]
        diagonal = torch.eye(batch_size, dtype=z[0].dtype, device=z[0].device)
        l_intra = (
            self.cl(z[0], z_t[0], diagonal)
            + self.cl(z[1], z_t[1], diagonal)
        ) / 2.0
        l_inter = self._cross_view_loss(p, z_t, inter_targets)
        return l_intra + l_inter

    def _cross_view_loss(self, predictions, target_embeddings, inter_targets):
        if len(inter_targets) != 2:
            raise ValueError("two cross-view targets are required")
        return (
            self.cl(predictions[0], target_embeddings[1], inter_targets[0])
            + self.cl(predictions[1], target_embeddings[0], inter_targets[1])
        ) / 2

    def forward(self, data, training_state, training_phase, external_targets=None, update_target=True):
        validated_external_targets = None
        if external_targets is not None:
            if self.n_views != 2:
                raise ValueError("external P12/P21 targets require exactly two views")
            validated_external_targets = self._validate_external_targets(
                external_targets,
                data[0].shape[0],
                data[0].device,
                data[0].dtype,
            )
        if validated_external_targets is not None:
            z, p, z_t = self.compute_training_features(
                data, momentum=training_state["mmt"], update_target=update_target
            )
            return self.loss_from_training_features(z, p, z_t, validated_external_targets)
        if update_target:
            self._update_target_branch(training_state["mmt"])
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        if training_phase["rectify"]:
            mp_intra = torch.eye(
                z[0].shape[0], device=z[0].device, dtype=z[0].dtype
            )
            mp_intra = [mp_intra, mp_intra]
            mp_inter = mp_intra
        else:
            mp_inter, mp_intra = self.joint_dist_estimation(
                z_t, z, p, training_state
            )
        l_intra = (self.cl(z[0], z_t[0], mp_intra[0]) + self.cl(z[1], z_t[1], mp_intra[1])) / 2
        l_inter = self._cross_view_loss(p, z_t, mp_inter)
        loss = l_inter + l_intra
        return loss

class FCN(nn.Module):
    def __init__(
        self,
        dim_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_out=0.0,
        norm_last_layer=True,
    ):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), act_layer(), nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class MaxLikelihoodLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(MaxLikelihoodLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, target=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if target is None:
            target = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T),self.temperature)
        similarity = similarity.view(-1)  # 展平为一维
        similarity = torch.softmax(similarity, dim=0) * N
        similarity = -torch.log(similarity.view(N, N))
        nll_loss = similarity * target
        loss = nll_loss.mean()
        return loss


# Compatibility symbol for checkpoints or external scripts written against
# the original CorreGen name. New REBUILD code must use CorreGenBaselineModel.
OursModel = CorreGenBaselineModel
