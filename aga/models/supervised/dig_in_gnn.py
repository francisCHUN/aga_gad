
from __future__ import annotations

import logging
import random
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..base import BaseGADModel

log = logging.getLogger("aga.models.dig_in_gnn")


class _Gen(nn.Module):

    def __init__(self, latent_dim: int, gen_hidden: int = 512,
                 latent_cap: int = 256):
        super().__init__()
        self.proj = nn.Linear(latent_dim, min(latent_dim, latent_cap)) \
            if latent_dim > latent_cap else None
        eff = min(latent_dim, latent_cap)
        self.discriminator = _Discriminator(eff)
        self.fc1 = nn.Linear(eff, gen_hidden)
        self.fc2 = nn.Linear(gen_hidden, 4 * gen_hidden)
        self.fc_flat = nn.Linear(4 * gen_hidden, eff)
        self.unproj = nn.Linear(eff, latent_dim) if self.proj is not None else None
        self.bn0 = nn.BatchNorm1d(latent_dim)
        self.bn_out = nn.BatchNorm1d(latent_dim) if self.proj is not None else None

    def forward(self, x, return_eff=False):
        x = self.bn0(x)
        raw_feats = torch.tanh(x)
        h = self.proj(x) if self.proj is not None else x
        h = self.fc1(h)
        h = self.fc2(h)
        eff = torch.tanh(self.fc_flat(h))
        out = self.unproj(eff) if self.unproj is not None else eff
        if return_eff:
            return out, raw_feats, eff
        return out, raw_feats

    def scale_raw(self, raw_feats):
        return self.proj(raw_feats) if self.proj is not None else raw_feats

    def contrast(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        return self.discriminator(
            gen_feats if self.proj is None else self._down(gen_feats),
            self.scale_raw(raw_feats),
            env_feats if self.proj is None else self._down(env_feats),
            self.scale_raw(env_raw_feats),
            labels)

    def _down(self, t):
        return self.proj(t) if self.proj is not None else t


class _Discriminator(nn.Module):

    def __init__(self, n_h: int):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.f_k_env = nn.Bilinear(n_h, n_h, 1)
        for m in (self.f_k, self.f_k_env):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _contrast_samples(gen_feats, raw_feats, labels):
        lab = labels.detach().cpu().view(-1)
        pos_idx = [i for i in range(len(lab)) if int(lab[i]) == 1]
        neg_idx = [i for i in range(len(lab)) if int(lab[i]) == 0]
        pos_idx = pos_idx or neg_idx or [0]
        neg_idx = neg_idx or pos_idx or [0]
        pos_mean = gen_feats[pos_idx].mean(dim=0, keepdim=True)
        neg_mean = gen_feats[neg_idx].mean(dim=0, keepdim=True)
        raw_pos = raw_feats[pos_idx].mean(dim=0, keepdim=True)
        raw_neg = raw_feats[neg_idx].mean(dim=0, keepdim=True)
        is_pos = (labels.view(-1) == 1).unsqueeze(1)
        B = gen_feats.shape[0]
        pm = pos_mean.expand(B, -1)
        nm = neg_mean.expand(B, -1)
        rp = raw_pos.expand(B, -1)
        rn = raw_neg.expand(B, -1)
        pos_anchor = torch.where(is_pos, pm, nm)
        neg_anchor = torch.where(is_pos, nm, pm)
        raw_pos_anchor = torch.where(is_pos, rp, rn)
        raw_neg_anchor = torch.where(is_pos, rn, rp)
        return pos_anchor, neg_anchor, raw_pos_anchor, raw_neg_anchor

    def forward(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        pos_a, neg_a, raw_pos_a, raw_neg_a = self._contrast_samples(
            gen_feats, raw_feats, labels)
        ctx_pos = 2 * self.f_k(pos_a, gen_feats) + self.f_k(raw_pos_a, gen_feats)
        ctx_neg = 2 * self.f_k(neg_a, gen_feats) + self.f_k(raw_neg_a, gen_feats)
        context_logits = torch.cat((ctx_pos, ctx_neg))
        env_contrast = (2 * env_feats + env_raw_feats) / 3
        env_logits = self.f_k_env(env_contrast, gen_feats)
        return context_logits, env_logits


class _PolicyNetwork(nn.Module):

    def __init__(self, lr: float, state_dim: int, action_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.prob = nn.Linear(512, action_dim)
        self.instancenorm = nn.LayerNorm(state_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                          weight_decay=0.001)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.prob(x), dim=-1)


class _Reinforce(nn.Module):

    def __init__(self, lr: float, state_dim: int, action_dim: int,
                 sample_rate: float, tolerance: float, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma
        self.sample_rate = sample_rate
        self.tolerance = tolerance
        self.reward_memory: List[torch.Tensor] = []
        self.log_prob_memory: List[torch.Tensor] = []
        self.action_memory: List[torch.Tensor] = []
        self.rate = 0.0
        self.pre_reward = 0.5
        self.num = 1
        self.policy = _PolicyNetwork(lr, state_dim, action_dim)

    def choose_action(self, observation: torch.Tensor, keep_idx) -> torch.Tensor:
        probabilities = self.policy(observation)
        dist = Categorical(probabilities)
        action = dist.sample()
        self.log_prob_memory.append(dist.log_prob(action)[keep_idx])
        self.action_memory.append(action[keep_idx])
        return action

    def store_reward(self, reward: torch.Tensor):
        self.reward_memory.append(reward)

    def memory_clear(self):
        self.reward_memory.clear()
        self.log_prob_memory.clear()
        self.action_memory.clear()

    def learn(self, batch_labels) -> None:
        if not self.log_prob_memory or not self.reward_memory:
            return
        log_memory = torch.cat(self.log_prob_memory)
        action = torch.cat(self.action_memory)
        n = max(len(self.reward_memory[0]), 1)
        lbl = torch.as_tensor(list(batch_labels), dtype=torch.long)
        if lbl.numel() < len(log_memory):
            rep = torch.ones(len(log_memory), dtype=torch.long)
            rep[:lbl.numel()] = lbl
            if lbl.numel() > 0:
                rep[lbl.numel():] = lbl[-1]
            lbl = rep
        else:
            lbl = lbl[:len(log_memory)]
        self.policy.optimizer.zero_grad()
        loss = 0.0
        reward = 0.0
        reward_pos = 0.0
        rate = max(self.rate, 1e-8)
        for g, log_prob, label, act in zip(self.reward_memory[0], log_memory,
                                           lbl, action):
            if int(act.item()) == 1:
                r = g[label] - (self.pre_reward - self.tolerance / rate)
            else:
                r = (self.pre_reward - self.tolerance / rate) - g[label]
            reward += r
            reward_pos += g[label]
            loss = loss + r * (-log_prob)
        loss = loss / n
        reward = reward / n
        reward_pos = reward_pos / n
        pre = self.pre_reward
        self.pre_reward = (pre * self.num + float(reward_pos)) / (self.num + 1)
        self.num += 1
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward(retain_graph=True)
            self.policy.optimizer.step()
        self.memory_clear()


def _value_head(feat_dim: int, obs_cap: int = 256, out_cap: int = 256) -> nn.Module:

    l1_in = 3 * feat_dim
    l2_in = 2 * feat_dim
    if l1_in > obs_cap:
        mid_in, mid_out = obs_cap, min(l2_in, out_cap)
    else:
        mid_in, mid_out = l1_in, l2_in

    class _W(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(3 * feat_dim, mid_in) if mid_in != 3 * feat_dim else None
            self.reward_weight1 = nn.Parameter(torch.FloatTensor(mid_in, mid_out))
            self.reward_weight2 = nn.Parameter(torch.FloatTensor(mid_out, 2))
            nn.init.xavier_uniform_(self.reward_weight1)
            nn.init.xavier_uniform_(self.reward_weight2)
            self.bn = nn.BatchNorm1d(3 * feat_dim)

        def forward(self, features):
            h = self.bn(features)
            if self.project is not None:
                h = self.project(h)
            h = h.mm(self.reward_weight1)
            return h.mm(self.reward_weight2)

    return _W()


class DiGInGNNModel(BaseGADModel):

    name = "dig_in_gnn"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 1,
                 dropout: float = 0.0, n_relation: int = 3,
                 gen_lr: float = 0.003, rl_lr: float = 0.003,
                 gen_hidden: int = 512, intra_cap: int = 512,
                 sample_rates: Tuple[float, ...] = (0.3, 0.3, 0.3),
                 tolerance: float = 0.06, batch_size: int = 256,
                 gnn_update_start: int = 45, gen_update_end: int = 45,
                 rl_update_start: int = 45, **_):
        super().__init__()
        if len(sample_rates) < n_relation:
            sample_rates = tuple(
                list(sample_rates) + [sample_rates[-1]] *
                (n_relation - len(sample_rates)))
        self.feat_dim = in_dim
        self.embed_dim = hidden
        self.n_relation = n_relation
        self.batch_size = batch_size
        self.gnn_update_start = gnn_update_start
        self.gen_update_end = gen_update_end
        self.rl_update_start = rl_update_start
        self._epoch = 0
        self.xent = nn.CrossEntropyLoss()

        lat = 2 * in_dim
        self.nei_gen = nn.ModuleList(
            [_Gen(lat, gen_hidden) for _ in range(n_relation)])
        self.gen_optimizers = [
            torch.optim.Adam(self.nei_gen[i].parameters(), lr=gen_lr,
                             weight_decay=0.001)
            for i in range(n_relation)
        ]
        state_dim = in_dim + lat
        rl_objs = [_Reinforce(rl_lr, state_dim, 2, sample_rates[i], tolerance)
                   for i in range(n_relation)]
        self.rl_agents = nn.ModuleList(rl_objs)
        self.value_heads = nn.ModuleList(
            [_value_head(in_dim) for _ in range(n_relation)])

        cap = min(3 * in_dim, intra_cap)
        self._intra_out = cap
        self.intra_proj = nn.ModuleList(
            [nn.Linear(3 * in_dim, cap) for _ in range(n_relation)])
        self.intra_w = nn.ModuleList(
            [nn.Linear(cap, cap) for _ in range(n_relation)])
        self.intra_bn = nn.ModuleList(
            [nn.BatchNorm1d(cap) for _ in range(n_relation)])
        inter_in = cap * n_relation + in_dim
        self.inter_bn = nn.BatchNorm1d(inter_in)
        self.inter_w = nn.Linear(inter_in, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 2))

        self._gen_replay: dict = {}
        self._adj_n = -1
        self._adj_lists: Optional[List[List[int]]] = None

    def _get_adj_lists(self, g) -> List[List[int]]:
        if self._adj_lists is None or self._adj_n != g.num_nodes():
            succ: List[List[int]] = [[] for _ in range(g.num_nodes())]
            src, dst = g.edges()
            for s, d in zip(src.tolist(), dst.tolist()):
                succ[s].append(d)
            self._adj_lists = succ
            self._adj_n = g.num_nodes()
        return self._adj_lists

    def _relation_neighbors(self, adj_lists, nodes, rate) -> List[List[int]]:
        out = []
        for n in nodes:
            nbrs = adj_lists[int(n)]
            if not nbrs:
                out.append([int(n)])
                continue
            keep = min(max(1, round(len(nbrs) * rate)), len(nbrs))
            out.append([int(n)] + random.sample(nbrs, keep))
        return out

    @staticmethod
    def _masked_mean(feats, sets: List[Set[int]], device) -> torch.Tensor:
        unique = list(set.union(*sets)) if sets else []
        umap = {n: i for i, n in enumerate(unique)}
        mask = torch.zeros(len(sets), max(len(unique), 1), device=device)
        cols = [umap[n] for s in sets for n in s]
        rows = [i for i in range(len(sets)) for _ in range(len(sets[i]))]
        mask[rows, cols] = 1.0
        mask = mask.div(mask.sum(1, keepdim=True).clamp(min=1.0))
        embed = feats[torch.LongTensor(unique).to(device)] if unique \
            else torch.zeros(1, feats.shape[1], device=device)
        return mask.mm(embed)

    def _rl_choose(self, labels, feats, gen_feats, neighs_list, rl: _Reinforce):
        device = feats.device
        samp_neighs: List[Set[int]] = []
        obs1, obs2 = [], []
        label_list: List[int] = []
        rate = 0.0
        for idx, neighs in enumerate(neighs_list):
            if len(neighs) == 1:
                samp_neighs.append(set(neighs))
                continue
            gen_feat = gen_feats[idx].expand(len(neighs) - 1, -1)
            neighs_feature = feats[torch.LongTensor(neighs[1:]).to(device)]
            observation = torch.cat((neighs_feature, gen_feat), dim=1)
            observation = rl.policy.instancenorm(observation)
            label = labels[idx]
            if rl.sample_rate >= 1:
                l_ids = list(range(len(neighs) - 1))
            else:
                k = min(max(1, int((len(neighs) - 1) * rl.sample_rate)),
                        len(neighs) - 1)
                l_ids = random.sample(range(len(neighs) - 1), k)
            label_list.extend([int(label.item())] * len(l_ids))
            obs1.append(neighs_feature[l_ids])
            obs2.append(gen_feat[l_ids])
            action = rl.choose_action(observation, l_ids)
            selected = [neighs[0]]
            for i in range(1, len(neighs)):
                if int(action[i - 1].item()) == 1:
                    selected.append(neighs[i])
            samp_neighs.append(set(selected))
            rate += (len(selected) - 1) / (len(neighs) - 1)
        agg_feats = self._masked_mean(feats, samp_neighs, device)
        rl.rate = rate / max(len(obs1), 1)
        o1 = torch.cat(obs1) if obs1 else torch.zeros(0, feats.shape[1], device=device)
        o2 = torch.cat(obs2) if obs2 else torch.zeros(0, gen_feats.shape[1], device=device)
        return agg_feats, o1, o2, label_list

    def _intra_forward(self, rel, nodes, labels, neighs_list, x, use_rl):
        device = x.device
        n_ids = torch.LongTensor(nodes).to(device)
        self_feats = x[n_ids]
        agg_feats = self._masked_mean(x, [set(ns) for ns in neighs_list], device)

        cat_feats = torch.cat((self_feats, agg_feats), dim=1)
        gen_feats, raw_feats = self.nei_gen[rel](cat_feats)

        env_sets = [set(ns[1:]) for ns in neighs_list]
        env_agg = self._masked_mean(x, env_sets, device)
        zero = torch.zeros(self_feats.shape[0], self_feats.shape[1], device=device)
        env_cat = torch.cat((zero, env_agg), dim=1)
        env_gen, env_raw = self.nei_gen[rel](env_cat)
        self._gen_replay[rel] = (cat_feats.detach(), env_cat.detach())

        if use_rl:
            rl_agg, o1, o2, label_list = self._rl_choose(
                labels, x, gen_feats.detach(), neighs_list, self.rl_agents[rel])
            used_agg = rl_agg
            observations = torch.cat((o1, o2), dim=1) if o1.numel() else \
                torch.zeros(0, 3 * self.feat_dim, device=device)
        else:
            used_agg = agg_feats
            observations = torch.zeros(0, 3 * self.feat_dim, device=device)
            label_list = []

        feats_cat = torch.cat((used_agg, gen_feats.detach()), dim=1)
        feats_cat = self.intra_proj[rel](feats_cat)
        to_feats = F.relu(self.intra_w[rel](self.intra_bn[rel](feats_cat)))
        return to_feats, gen_feats, raw_feats, env_gen, env_raw, observations, label_list

    def _inter_forward(self, nodes, x, intra_to):
        device = x.device
        self_feats = x[torch.LongTensor(nodes).to(device)]
        cat_feats = torch.cat([self_feats] + intra_to, dim=1)
        combined = F.relu(self.inter_bn(cat_feats))
        combined = self.inter_w(combined)
        scores = self.head(combined)
        return scores

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        device = x.device
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(
                g.num_nodes(), dtype=torch.bool, device=device)
        )
        adj_lists = self._get_adj_lists(g)
        train_idx = torch.where(node_mask)[0]
        labels_np = anomaly.cpu().numpy()

        pos_ids = train_idx[anomaly[train_idx] == 1]
        n_pos = max(int(pos_ids.numel()), 1)
        from .pcgnn import pick_step
        deg = g.out_degrees().tolist()
        sampled = pick_step(train_idx.tolist(),
                            labels_np[train_idx.tolist()], deg, 2 * n_pos)
        random.shuffle(sampled)

        epoch = self._epoch
        self._epoch += 1
        gen_phase = epoch <= self.gen_update_end
        gnn_phase = epoch >= self.gnn_update_start
        rl_phase = epoch >= self.rl_update_start

        total = torch.zeros((), device=device)
        nb = 0
        for start in range(0, len(sampled), self.batch_size):
            batch_nodes = sampled[start:start + self.batch_size]
            if len(batch_nodes) < 2:
                continue
            labels = anomaly[torch.LongTensor(batch_nodes).to(device)].long()

            neighs_per_rel = [
                self._relation_neighbors(adj_lists, batch_nodes,
                                         self.rl_agents[r].sample_rate)
                for r in range(self.n_relation)
            ]
            use_rl = gnn_phase or (rl_phase and epoch > self.gnn_update_start)

            intra_to, ctx_ls, env_ls, reward_ce = [], [], [], 0.0
            for rel in range(self.n_relation):
                to_feats, gen_feats, raw_feats, env_gen, env_raw, obs, lab = \
                    self._intra_forward(rel, batch_nodes, labels,
                                        neighs_per_rel[rel], x, use_rl)
                intra_to.append(to_feats)
                ctx, envl = self.nei_gen[rel].contrast(
                    gen_feats, raw_feats, env_gen, env_raw, labels)
                ctx_ls.append(ctx.detach())
                env_ls.append(envl.detach())
                if rl_phase and obs.shape[0] >= 2:
                    reward = torch.softmax(
                        F.leaky_relu(self.value_heads[rel](obs), 0.2), dim=1)
                    self.rl_agents[rel].store_reward(reward)
                    self.rl_agents[rel].learn(lab)
                    reward_ce = reward_ce + self.xent(
                        reward, torch.as_tensor(lab, dtype=torch.long,
                                                device=device))
            logits = self._inter_forward(batch_nodes, x, intra_to)
            gnn_loss = self.xent(logits, labels)
            if torch.is_tensor(reward_ce) and float(reward_ce) != 0.0:
                gnn_loss = gnn_loss + 0.5 * reward_ce / self.n_relation

            if gen_phase:
                contrast_label = torch.cat((
                    torch.ones(len(batch_nodes), device=device),
                    torch.zeros(len(batch_nodes), device=device))).unsqueeze(1)
                env_label = (1.0 - labels.float()).unsqueeze(1)
                for rel in range(self.n_relation):
                    op = self.gen_optimizers[rel]
                    op.zero_grad()
                    if rel in self._gen_replay:
                        cat_d, env_d = self._gen_replay[rel]
                        gf, rf = self.nei_gen[rel](cat_d)
                        e_g, e_r = self.nei_gen[rel](env_d)
                        con_l, env_l = self.nei_gen[rel].contrast(
                            gf, rf, e_g, e_r, labels)
                    else:
                        con_l, env_l = ctx_ls[rel].detach(), env_ls[rel].detach()
                    con = F.binary_cross_entropy_with_logits(
                        con_l, contrast_label)
                    envm = F.binary_cross_entropy_with_logits(
                        env_l, env_label)
                    gen_loss = (3 * con + envm) / 4
                    gen_loss.backward()
                    op.step()

            if gnn_phase:
                total = total + gnn_loss
                nb += 1

        if nb == 0:
            total = sum(p.sum() * 0.0 for p in self.parameters()
                        if p.requires_grad)
        return total / max(nb, 1)

    @torch.no_grad()
    def score(self, g, x):
        device = x.device
        adj_lists = self._get_adj_lists(g)
        N = g.num_nodes()
        out = torch.zeros(N, device=device)
        rl_has_trained = self._epoch > self.gnn_update_start
        for start in range(0, N, self.batch_size):
            nodes = list(range(start, min(start + self.batch_size, N)))
            if len(nodes) < 2:
                nodes = nodes + nodes[: 2 - len(nodes)]
            labels = torch.zeros(len(nodes), dtype=torch.long, device=device)
            intra_to = []
            for rel in range(self.n_relation):
                neighs = self._relation_neighbors(
                    adj_lists, nodes, self.rl_agents[rel].sample_rate)
                to_feats, *_ = self._intra_forward(
                    rel, nodes, labels, neighs, x, rl_has_trained)
                intra_to.append(to_feats)
            logits = self._inter_forward(nodes, x, intra_to)
            p = torch.softmax(logits, dim=1)[:, 1]
            out[start:start + len(nodes)] = p[:len(nodes)]
        return out.clamp(0.0, 1.0)