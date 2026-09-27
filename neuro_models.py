import os
import time
import math
import numpy as np
import torch
import torch.nn as nn

from neuro_heterarchy_core import DEVICE

MAX_CONCEPTS_CAPACITY = 16

ELECTRODE_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14,
                        -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
ELECTRODE_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73,
                         2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)

class DynamicPilot:
    def __init__(self, sensitivity=0.05, max_speed=9.0, fwd_scale=0.2, strafe_scale=0.2, turn_scale=0.5, intent_gain=1.5):
        self.x, self.y = 0.0, 0.0
        self.vx, self.vy = 0.0, 0.0
        self.persistence = 0.0
        self.last_ix, self.last_iy = 0.0, 0.0
        self.wm_curvature = 0.0
        self.temporal_bias = 0.0
        self.max_speed = max_speed

    def update(self, dt: float, force_x: float, force_y: float, wm_curvature: float, temp_bias: float):
        self.wm_curvature = wm_curvature
        self.temporal_bias = temp_bias

        mag = math.hypot(force_x, force_y)
        alignment = 0.5 * (((force_x * self.last_ix + force_y * self.last_iy) / (mag * (math.hypot(self.last_ix, self.last_iy) + 1e-6))) + 1.0) if mag > 0.01 else 0.5
        self.persistence = self.persistence * 0.94 + 0.06 * alignment * math.tanh(mag * 2.0)
        self.last_ix, self.last_iy = force_x, force_y

        active_boost = 1.0 + self.persistence * 4.0
        base_speed = 3.2 * active_boost
        target_vx, target_vy = force_x * base_speed, force_y * base_speed

        t_mag = math.hypot(target_vx, target_vy)
        if t_mag > self.max_speed:
            target_vx = (target_vx / t_mag) * self.max_speed
            target_vy = (target_vy / t_mag) * self.max_speed

        self.vx = self.vx * 0.88 + target_vx * 0.12
        self.vy = self.vy * 0.88 + target_vy * 0.12
        self.x += self.vx * dt
        self.y += self.vy * dt
        return active_boost

class CanonicalHTMColumn(nn.Module):
    def __init__(self, node_id: str = "Node", num_columns: int = 4096, k_active: int = 80, num_slots: int = 32):
        super().__init__()
        self.num_columns, self.k_active, self.num_slots = num_columns, k_active, num_slots
        ex, ey = [], []
        for i in range(16):
            for j in range(i + 1, 16):
                ex.append((ELECTRODE_X[i] + ELECTRODE_X[j]) / 2.0)
                ey.append((ELECTRODE_Y[i] + ELECTRODE_Y[j]) / 2.0)
        self.register_buffer("edge_x", torch.tensor(ex, device=DEVICE, dtype=torch.float32))
        self.register_buffer("edge_y", torch.tensor(ey, device=DEVICE, dtype=torch.float32))

        # РАВНОМЕРНАЯ ДВУМЕРНАЯ ТОПОЛОГИЯ 64x64 ПО ВСЕМУ ПОЛЮ
        grid_dim = int(math.isqrt(num_columns))
        grid_x = torch.linspace(-12.0, 12.0, grid_dim, device=DEVICE)
        grid_y = torch.linspace(-12.0, 12.0, grid_dim, device=DEVICE)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        cx = mesh_x.reshape(-1, 1) # [4096, 1]
        cy = mesh_y.reshape(-1, 1) # [4096, 1]
        
        # Расстояние от каждой из 4096 колонок ко всем 120 биполярным диполям
        d_sq = (cx - self.edge_x.view(1, 120))**2 + (cy - self.edge_y.view(1, 120))**2
        spatial_rf = torch.exp(-d_sq / 35.0) # [4096, 120]
        
        # Гарантируем, что у каждой колонки есть связность (нормализация синапсов)
        connected = (spatial_rf >= 0.20).float()
        self.register_buffer("connected_T", connected.T.contiguous()) # [120, 4096]
        self.register_buffer("synapse_counts", torch.sum(connected, dim=1).clamp(min=1.0).view(1, -1))

    def compute_sdr(self, pac_iplv_32x120: torch.Tensor) -> torch.Tensor:
        """100% GPU бинарный L4 SDR (32 слота x 4096 колонок)"""
        x_clean = torch.relu(pac_iplv_32x120)
        if torch.max(x_clean) < 1e-5: 
            return torch.zeros((self.num_slots, self.num_columns), device=DEVICE)
        overlap = torch.matmul(x_clean, self.connected_T) / self.synapse_counts
        _, active_indices = torch.topk(overlap, self.k_active, dim=-1)
        sdr_seq = torch.zeros((self.num_slots, self.num_columns), device=DEVICE)
        sdr_seq.scatter_(1, active_indices, 1.0)
        return sdr_seq

class FrontalExecutiveHeterarchy(nn.Module):
    def __init__(self, num_nodes: int = 4, max_capacity: int = MAX_CONCEPTS_CAPACITY, 
                 num_columns_per_node: int = 4096, k_active_per_node: int = 80, num_slots: int = 32):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_slots = num_slots
        self.num_cols = num_columns_per_node
        self.max_capacity = max_capacity
        self.k_active = k_active_per_node
        
        self.nodes = nn.ModuleList([
            CanonicalHTMColumn(node_id=f"Node_{i}", num_columns=num_columns_per_node, k_active=k_active_per_node, num_slots=num_slots)
            for i in range(self.num_nodes)
        ])
        
        # Веса L4: [4 узла, 16 концептов, 32 слота, 4096 колонок]
        self.register_buffer("local_weights", torch.zeros((num_nodes, max_capacity, num_slots, num_columns_per_node), device=DEVICE))
        
        # Пластичный слой L2/3
        self.register_buffer("l23_docking_weights", torch.zeros((max_capacity, num_columns_per_node), device=DEVICE))
        gen = torch.Generator(device=DEVICE).manual_seed(42)
        rand_codes = torch.randn((max_capacity, num_columns_per_node), generator=gen, device=DEVICE)
        _, obj_topk = torch.topk(rand_codes, k_active_per_node, dim=1)
        object_sdr_table = torch.zeros((max_capacity, num_columns_per_node), device=DEVICE)
        object_sdr_table.scatter_(1, obj_topk, 1.0)
        self.register_buffer("object_sdr_table", object_sdr_table)
        self.l23_docking_weights.copy_(self.object_sdr_table)

        self.register_buffer("calcium_traces", torch.zeros((num_nodes, num_slots, num_columns_per_node), device=DEVICE))
        self.register_buffer("membrane_potential", torch.zeros(max_capacity, device=DEVICE))
        self.satiation_counter = np.zeros(max_capacity, dtype=np.float32)
        self.agent_votes = np.zeros((num_nodes, max_capacity), dtype=np.float32)

    def get_all_sdrs(self, iplv_gamma_nodes: list[torch.Tensor]) -> list[torch.Tensor]:
        node_sdrs = []
        for i in range(self.num_nodes):
            if i < len(iplv_gamma_nodes): 
                node_sdrs.append(self.nodes[i].compute_sdr(iplv_gamma_nodes[i]))
            else: 
                node_sdrs.append(torch.zeros((self.num_slots, self.num_cols), device=DEVICE))
        return node_sdrs

    def reset_calcium(self):
        self.calcium_traces.zero_()
        self.satiation_counter.fill(0.0)

    def inherit_synapses(self, parent_idx: int, child_idx: int):
        if child_idx < self.max_capacity and parent_idx < self.max_capacity:
            self.local_weights[:, child_idx] = self.local_weights[:, parent_idx].clone() * 0.95
            self.l23_docking_weights[child_idx] = self.l23_docking_weights[parent_idx].clone() * 0.95

    def learn_contrastive_node(self, node_idx: int, target_idx: int, active_count: int, lr: float = 0.05, ltd_factor: float = 0.5):
        """Обучение на GPU по правилу Ойя без тотального подавления старых классов"""
        trace = self.calcium_traces[node_idx]
        if torch.max(trace) > 1e-4:
            # 1. Специфическое LTP
            delta = trace - self.local_weights[node_idx, target_idx]
            self.local_weights[node_idx, target_idx] += lr * delta
            
            # 2. Селективное LTD ТОЛЬКО на синапсы, активные в текущем стимуле
            # Ограничиваем сумму LTD, чтобы старые концепты не стирались в ноль
            active_mask = (trace > 0.1).float()
            for other in range(active_count):
                if other != target_idx:
                    overlap = active_mask * self.local_weights[node_idx, other]
                    self.local_weights[node_idx, other] -= (lr * ltd_factor) * overlap
            self.local_weights[node_idx].clamp_(0.0, 1.0)

            # Пластичность L2/3 на GPU
            last_trace = trace[-1]
            if torch.max(last_trace) > 0.1:
                self.l23_docking_weights[target_idx] += lr * 0.4 * (last_trace - self.l23_docking_weights[target_idx])
                self.l23_docking_weights[target_idx].clamp_(0.0, 1.0)

    def learn_contrastive_all(self, target_idx: int, active_count: int, lr: float = 0.05, ltd_factor: float = 0.5):
        for i in range(self.num_nodes):
            self.learn_contrastive_node(i, target_idx, active_count, lr, ltd_factor)

    def get_ltm_scores(self, count: int) -> np.ndarray:
        """100% ВЕКТОРИЗОВАННЫЙ РАСЧЁТ LTM НА GPU (ноль циклов .item(), мгновенно)"""
        if count <= 1: return np.array([100.0] * count, dtype=np.float32)
        with torch.no_grad():
            # w_flat: [num_nodes, count, 131072]
            w_flat = self.local_weights[:, :count].reshape(self.num_nodes, count, -1)
            w_max = torch.max(w_flat, dim=-1).values # [num_nodes, count]
            
            w_norm = torch.nn.functional.normalize(w_flat, p=2, dim=-1)
            # Батчевое матричное умножение на GPU: [num_nodes, count, count]
            cosine_sim = torch.bmm(w_norm, w_norm.transpose(1, 2))
            
            # Обнуляем диагональ
            diag_mask = torch.eye(count, device=DEVICE, dtype=torch.bool).unsqueeze(0)
            cosine_sim.masked_fill_(diag_mask, -1.0)
            
            # Максимальный оверлап с конкурентами для каждого класса
            max_overlap = torch.max(cosine_sim, dim=-1).values # [num_nodes, count]
            
            sel = torch.clamp((0.85 - max_overlap) / 0.50, 0.0, 1.0) * 100.0
            
            # Если синапсы еще не обучены (w_max < 0.10), селективность = 0
            sel = torch.where(w_max >= 0.10, sel, torch.zeros_like(sel))
            
            # Среднее по 4 узлам
            return torch.mean(sel, dim=0).cpu().numpy().astype(np.float32)

    def predict_evidence_and_docking(self, node_sdrs: list[torch.Tensor], active_count: int, 
                                     dt: float = 0.016, tau: float = 0.150, is_calibrating: bool = False):
        """Интеграция 60 FPS на GPU"""
        for i in range(self.num_nodes):
            if i < len(node_sdrs):
                self.calcium_traces[i] = torch.max(self.calcium_traces[i] * 0.88, node_sdrs[i])

        ltm_scores = self.get_ltm_scores(active_count)
        
        # 1. Overlap по полной 131072-траектории на GPU
        node_evidences = torch.zeros((self.num_nodes, active_count), device=DEVICE)
        for n in range(self.num_nodes):
            w_flat = self.local_weights[n, :active_count].view(active_count, -1)
            s_flat = (self.calcium_traces[n].view(-1) > 0.1).float()
            overlap = torch.mv(w_flat, s_flat)
            # Защита от деления на малую норму: база + sqrt(w_sums)
            w_sums = torch.sum(w_flat, dim=1).clamp(min=10.0)
            node_evidences[n] = overlap / torch.sqrt(w_sums)
            self.agent_votes[n, :active_count] = node_evidences[n].cpu().numpy()

        # 2. Мембранный потенциал L2/3
        total_current = torch.sum(node_evidences, dim=0)

        alpha = dt / tau
        self.membrane_potential[:active_count] = (1.0 - alpha) * self.membrane_potential[:active_count] + alpha * total_current
        effective_mp = self.membrane_potential[:active_count].clone()

        # 3. Мягкая анти-deadlock сатиация (максимум -0.35, концепт никогда не убивается в 0.0)
        if not is_calibrating:
            dom_idx = int(torch.argmax(effective_mp).item())
            self.satiation_counter[dom_idx] += dt
            for i in range(active_count):
                if i != dom_idx: 
                    self.satiation_counter[i] = max(0.0, self.satiation_counter[i] - dt * 0.6)

            fatigue = float(np.clip((self.satiation_counter[dom_idx] - 4.0) * 0.10, 0.0, 0.35))
            effective_mp[dom_idx] -= fatigue
            effective_mp += torch.rand(active_count, device=DEVICE) * 0.005
        else:
            self.satiation_counter.fill(0.0)

        wm_scores = torch.clamp(effective_mp * 30.0, 0.0, 100.0).cpu().numpy()
        
        # Адаптивный Softmax (температура 0.50 удерживает и лидера, и второго призера активными)
        temp = 0.50 if not is_calibrating else 0.15
        weights_t = torch.softmax(effective_mp / temp, dim=0)
        weights = weights_t.cpu().numpy()

        # 4. L2/3 Consensus SDR на GPU
        consensus_activation = torch.mv(self.l23_docking_weights[:active_count].T, weights_t)
        
        for n in range(self.num_nodes):
            if n < len(node_sdrs):
                sdr_last = node_sdrs[n][-1] if node_sdrs[n].ndim == 2 else node_sdrs[n]
                consensus_activation += sdr_last * 0.20

        _, top_cols = torch.topk(consensus_activation, self.k_active)
        consensus_sdr = torch.zeros(self.num_cols, device=DEVICE)
        consensus_sdr[top_cols] = 1.0

        return weights, wm_scores, ltm_scores, consensus_sdr

    def compute_contrastive_margin(self, node_sdrs: list[torch.Tensor], target_idx: int, active_count: int) -> float:
        if active_count <= 1: return 1.0
        with torch.no_grad():
            node_ev = []
            for n in range(self.num_nodes):
                w_flat = self.local_weights[n, :active_count].view(active_count, -1)
                s_flat = (node_sdrs[n].view(-1) > 0.1).float()
                overlap = torch.mv(w_flat, s_flat)
                w_sums = torch.sum(w_flat, dim=1).clamp(min=10.0)
                node_ev.append(overlap / torch.sqrt(w_sums))
            total = torch.sum(torch.stack(node_ev, dim=0), dim=0)
            target_score = total[target_idx]
            other_scores = torch.cat([total[:target_idx], total[target_idx+1:]])
            return float((target_score - torch.max(other_scores)).item())

    def save_to_file(self, filepath: str, class_names: list):
        torch.save({
            'format_version': '17.0_tbt_gpu_optimized',
            'local_weights': self.local_weights[:, :len(class_names)].cpu(), 
            'l23_docking_weights': self.l23_docking_weights[:len(class_names)].cpu(),
            'class_names': class_names, 'timestamp': time.time()
        }, filepath)
        print(f"💾 [LTM] Банк весов TBT сохранен: {filepath}")

    def load_from_file(self, filepath: str) -> tuple[bool, int]:
        if not os.path.exists(filepath): return False, 0
        try:
            ckpt = torch.load(filepath, map_location=DEVICE, weights_only=True)
            cnt = min(self.max_capacity, len(ckpt.get('class_names', [])))
            if 'local_weights' in ckpt:
                self.local_weights[:, :cnt].copy_(ckpt['local_weights'][:, :cnt].to(DEVICE))
                if 'l23_docking_weights' in ckpt:
                    self.l23_docking_weights[:cnt].copy_(ckpt['l23_docking_weights'][:cnt].to(DEVICE))
                print(f"📂 [LTM] Загружен банк весов: {filepath} ({cnt} классов)")
                return True, cnt
            return False, 0
        except Exception: return False, 0

class BrainSubject:
    def __init__(self, subject_id: str, regions_map: dict, max_capacity: int = MAX_CONCEPTS_CAPACITY, **pilot_kwargs):
        self.subject_id = subject_id
        self.regions_map = regions_map
        self.heterarchy = FrontalExecutiveHeterarchy(num_nodes=max(1, len(regions_map)), max_capacity=max_capacity).to(DEVICE)
        self.pilot = DynamicPilot(**pilot_kwargs)
        
        self.decoded_leader_idx = 0
        self.confidence = 0.0
        self.wm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.ltm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.blended_weights = np.zeros(max_capacity, dtype=np.float32)
        
        self.last_node_sdrs = []
        self.consensus_sdr = None

    def has_region(self, region_name: str) -> bool: return region_name in self.regions_map
    def get_region_node(self, region_name: str, frame_nodes: list) -> any:
        if region_name in self.regions_map:
            dev_idx = self.regions_map[region_name]
            if 0 <= dev_idx < len(frame_nodes): return frame_nodes[dev_idx]
        return None

    def process_neurophysiology(self, frame_nodes: list, dt: float, active_count: int, is_calibrating: bool = False):
        my_tensors = []
        for reg_name, dev_idx in self.regions_map.items():
            if 0 <= dev_idx < len(frame_nodes):
                my_tensors.append(torch.tensor(frame_nodes[dev_idx].iplv_gamma, dtype=torch.float32, device=DEVICE))
        if not my_tensors: my_tensors = [torch.zeros((32, 120), device=DEVICE)]

        with torch.no_grad():
            node_sdrs = self.heterarchy.get_all_sdrs(my_tensors)
            self.last_node_sdrs = node_sdrs
            weights, wm, ltm, consensus_sdr = self.heterarchy.predict_evidence_and_docking(
                node_sdrs, active_count, dt=dt, is_calibrating=is_calibrating
            )
            self.consensus_sdr = consensus_sdr
            self.blended_weights = weights
            self.wm_scores = wm
            self.ltm_scores = ltm
            self.decoded_leader_idx = int(np.argmax(wm)) if active_count > 0 else 0
            self.confidence = float(np.max(wm))

    def learn_contrastive(self, target_idx: int, num_active: int, lr: float = 0.05, ltd_factor: float = 0.5):
        if self.last_node_sdrs:
            self.heterarchy.learn_contrastive_all(target_idx, num_active, lr=lr, ltd_factor=ltd_factor)

class OverBrainCollective:
    def __init__(self, subjects: list[BrainSubject]): self.subjects = subjects
        
    def resolve_heterarchy(self, active_count: int, worker_parent_map: dict, agent_parent_map: dict, lead_sign: float, leader_idx: int, child_idx: int):
        parent_map = {i: None for i in range(active_count)}
        for child, parent in worker_parent_map.items():
            if child < active_count and parent is not None and parent < active_count: parent_map[child] = parent
                
        if agent_parent_map:
            for child, parent in agent_parent_map.items():
                if child < active_count and parent is not None and parent < active_count and child != parent:
                    parent_map[child] = parent

        if leader_idx < active_count and child_idx < active_count and leader_idx != child_idx:
            if lead_sign >= 0.03:
                if parent_map.get(child_idx) is None: parent_map[child_idx] = leader_idx
            elif lead_sign <= -0.03:
                if parent_map.get(leader_idx) is None: parent_map[leader_idx] = child_idx

        for node in list(parent_map.keys()):
            curr, visited = parent_map[node], {node}
            while curr is not None:
                if curr in visited: parent_map[node] = None; break
                visited.add(curr)
                curr = parent_map.get(curr)
        return parent_map
