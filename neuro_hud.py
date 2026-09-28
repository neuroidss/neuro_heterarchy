#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS HUD (ДЕКУПЛИРОВАННЫЙ МОДУЛЬ ВИЗУАЛИЗАЦИИ)
===================================================================================
- Все приборы показывают ЧИСТЫЕ ФАЗОВЫЕ МЕТРИКИ (ноль скалярной мощности).
- F3:  2D-манифолд рангов (скользящая шайба [x_local, y_global]) + фазовый замок.
- F4:  Диафрагма фазовой энтропии (Курамото R_sync).
- FCz: 4D-радар: [lx, ly] + кольцо сагитты rx + стрела темпоральности ry.
- AFz: Резонансный туннель Дельты (ciPLV).
- Fpz: Эпистемический склон напряжения воли (фазовый дрейф).
- Горячее переключение: TAB (выбор девайса), 1-5 (роли), 0 (выкл).
===================================================================================
"""

import math
import colorsys
import numpy as np
import pygame
import torch

WIDTH, HEIGHT = 1800, 960

class GPUDynamicGraph:
    def __init__(self, max_nodes=16, cx=1340, cy=280, width=820, height=440, speed=4.5, repulsion=5500.0, attraction=0.45, device="cuda"):
        self.max_nodes, self.cx, self.cy = max_nodes, cx, cy
        self.half_w, self.half_h = width * 0.46, height * 0.44
        self.speed, self.repulsion_k, self.attraction_k = speed, repulsion, attraction
        self.device = device
        angles = torch.linspace(0, 2.0 * math.pi, max_nodes + 1, device=device)[:max_nodes]
        self.pos = torch.stack([self.cx + torch.cos(angles) * (self.half_w * 0.55), self.cy + torch.sin(angles) * (self.half_h * 0.55)], dim=-1)
        self.vel = torch.zeros((max_nodes, 2), device=device)
        self.radii_gpu = torch.full((max_nodes,), 22.0, device=device, dtype=torch.float32)
        self.center_t = torch.tensor([self.cx, self.cy], device=device)

    def update_physics(self, active_count, causal_flow_np, dt=0.016):
        flow_sub = torch.from_numpy(causal_flow_np[:active_count, :active_count]).to(self.device)
        net_lead = torch.sum(flow_sub, dim=1) - torch.sum(flow_sub, dim=0)
        self.radii_gpu[:active_count] = torch.clamp(20.0 + net_lead * 35.0, 16.0, 38.0)

        sub_steps = 2
        sub_dt = (dt * self.speed) / sub_steps
        for _ in range(sub_steps):
            p_sub = self.pos[:active_count]
            diff = p_sub.unsqueeze(1) - p_sub.unsqueeze(0)
            dist_sq = torch.sum(diff**2, dim=-1).clamp(min=25.0)
            dist = torch.sqrt(dist_sq)

            repulsion = (diff / (dist_sq.unsqueeze(-1) * dist.unsqueeze(-1))) * self.repulsion_k
            rep_force = torch.sum(torch.nan_to_num(repulsion), dim=1)
            flow_w = torch.clamp(torch.abs(flow_sub), 0.0, 1.0).unsqueeze(-1)
            attr_force = torch.sum(diff * flow_w * self.attraction_k, dim=1)
            grav_force = (self.center_t - p_sub) * 0.04

            total_force = rep_force + attr_force + grav_force
            self.vel[:active_count] = (self.vel[:active_count] + total_force * sub_dt) * 0.78
            self.pos[:active_count] += self.vel[:active_count] * sub_dt
            self.pos[:active_count, 0] = torch.clamp(self.pos[:active_count, 0], self.cx - self.half_w, self.cx + self.half_w)
            self.pos[:active_count, 1] = torch.clamp(self.pos[:active_count, 1], self.cy - self.half_h, self.cy + self.half_h)

        return self.pos.cpu().numpy(), self.radii_gpu[:active_count].cpu().numpy()

class DeltaSpiralHistory:
    def __init__(self, max_rings=5):
        self.max_rings = max_rings
        self.rings = []

    def push(self, total_sectors, concepts_in_sectors):
        self.rings.insert(0, (total_sectors, list(concepts_in_sectors[:total_sectors])))
        if len(self.rings) > self.max_rings: self.rings.pop()

    def render(self, screen, cx, cy, R_max, R_core, dl_phase, total_sectors, current_concepts, concept_colors, is_active_sec):
        dr = (R_max - R_core) / float(self.max_rings + 1)
        for ring_idx in range(len(self.rings) - 1, -1, -1):
            r_out = R_max - (ring_idx + 1) * dr
            r_in = r_out - dr + 2.0
            n_sec, ring_concepts = self.rings[ring_idx]
            alpha = int(240 * (1.0 - (ring_idx + 1) / float(self.max_rings + 2)))
            sec_step = (2.0 * math.pi) / float(n_sec)

            for s_i in range(n_sec):
                a_start = -math.pi * 0.5 + s_i * sec_step
                a_end = a_start + sec_step
                c_id = ring_concepts[s_i] if s_i < len(ring_concepts) else 0
                c_rgb = concept_colors[c_id % len(concept_colors)]
                arc_pts = []
                for st in range(9):
                    ang = a_start + (a_end - a_start) * (st / 8)
                    arc_pts.append((cx + math.cos(ang) * r_out, cy + math.sin(ang) * r_out))
                for st in range(8, -1, -1):
                    ang = a_start + (a_end - a_start) * (st / 8)
                    arc_pts.append((cx + math.cos(ang) * r_in, cy + math.sin(ang) * r_in))

                surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(surf, (c_rgb[0], c_rgb[1], c_rgb[2], alpha), arc_pts)
                screen.blit(surf, (0, 0))
                pygame.draw.polygon(screen, (15, 20, 30), arc_pts, 1)

        r_out_live = R_max
        r_in_live = R_max - dr + 2.0
        sec_w_live = (2.0 * math.pi) / float(total_sectors)

        for s_i in range(total_sectors):
            a_start = -math.pi * 0.5 + s_i * sec_w_live
            a_end = a_start + sec_w_live
            is_active = (s_i == is_active_sec)
            c_id = current_concepts[s_i % len(current_concepts)] if len(current_concepts) > 0 else 0
            c_rgb = concept_colors[c_id % len(concept_colors)]
            arc_pts = []
            for st in range(11):
                ang = a_start + (a_end - a_start) * (st / 10)
                arc_pts.append((cx + math.cos(ang) * r_out_live, cy + math.sin(ang) * r_out_live))
            for st in range(10, -1, -1):
                ang = a_start + (a_end - a_start) * (st / 10)
                arc_pts.append((cx + math.cos(ang) * r_in_live, cy + math.sin(ang) * r_in_live))

            surf_live = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(surf_live, (c_rgb[0], c_rgb[1], c_rgb[2], 255 if is_active else 180), arc_pts)
            screen.blit(surf_live, (0, 0))
            pygame.draw.polygon(screen, (255, 255, 255) if is_active else (40, 55, 70), arc_pts, 2 if is_active else 1)

        dl_norm = (dl_phase + math.pi) % (2.0 * math.pi)
        hand_ang = -math.pi * 0.5 + dl_norm
        hx = cx + math.cos(hand_ang) * (R_max + 6)
        hy = cy + math.sin(hand_ang) * (R_max + 6)
        pygame.draw.line(screen, (0, 255, 255), (cx, cy), (hx, hy), 3)
        pygame.draw.circle(screen, (255, 255, 255), (int(hx), int(hy)), 7)
        pygame.draw.circle(screen, (10, 14, 20), (cx, cy), int(R_core) - 2)
        pygame.draw.circle(screen, (0, 255, 200), (cx, cy), int(R_core) - 2, 2)

class NeuroCanvasHUD:
    def __init__(self, initial_minimal: bool = False, device="cuda"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("NeuroCanvas: Pure Phase-Coherence Heterarchy HUD (TBT 2.0)")
        self.clock = pygame.time.Clock()
        self.font_b = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_s = pygame.font.SysFont("consolas", 11)
        self.font_large = pygame.font.SysFont("consolas", 18, bold=True)
        self.show_minimal = initial_minimal
        self.spiral = DeltaSpiralHistory(max_rings=5)
        self.gpu_graph = GPUDynamicGraph(device=device)
        self.selected_dev_idx = 0
        self.concept_colors = []

    def poll_events(self, num_devices: int) -> tuple[float, list]:
        dt = self.clock.tick(60) / 1000.0
        commands = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT: raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h: self.show_minimal = not self.show_minimal
                elif event.key == pygame.K_r: commands.append(("reset_seals", None))
                elif event.key == pygame.K_c: commands.append(("flush_canvas", None))
                elif event.key == pygame.K_TAB:
                    if num_devices > 0:
                        self.selected_dev_idx = (self.selected_dev_idx + 1) % num_devices
                        print(f"👉 [ВЫБОР ДЕВАЙСА ДЛЯ СМЕНЫ РОЛИ] Dev {self.selected_dev_idx}")
                elif event.key == pygame.K_1: commands.append(("set_role", (self.selected_dev_idx, "AFz")))
                elif event.key == pygame.K_2: commands.append(("set_role", (self.selected_dev_idx, "F3")))
                elif event.key == pygame.K_3: commands.append(("set_role", (self.selected_dev_idx, "F4")))
                elif event.key == pygame.K_4: commands.append(("set_role", (self.selected_dev_idx, "FCz")))
                elif event.key == pygame.K_5: commands.append(("set_role", (self.selected_dev_idx, "Fpz")))
                elif event.key == pygame.K_0: commands.append(("set_role", (self.selected_dev_idx, None)))
        return dt, commands

    def render_waiting(self):
        self.screen.fill((8, 12, 18))
        self.screen.blit(self.font_large.render("ОЖИДАНИЕ LSL ПОТОКОВ (САМОДОСТАТОЧНАЯ КОРА)...", True, (255, 140, 50)), (380, 175))
        pygame.display.flip()

    def render(self, canvas_rgb: np.ndarray, telemetry: dict, fps: float = 0.0, dt: float = 0.016):
        if not self.concept_colors:
            num_c = max(16, telemetry.get("active_vocab_size", 16))
            for i in range(num_c):
                hue = (i * 0.618033988749895) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
                self.concept_colors.append((int(r * 255), int(g * 255), int(b * 255)))

        dag_parts = telemetry.get("dag_log_parts", ["WORLD"])
        if self.show_minimal:
            self.screen.fill((5, 5, 8))
            h, w = canvas_rgb.shape[:2]
            surf = pygame.image.frombuffer(canvas_rgb.tobytes(), (w, h), 'RGB')
            scaled = pygame.transform.scale(surf, (WIDTH - 80, HEIGHT - 80))
            self.screen.blit(scaled, (40, 40))
            hud_txt = self.font_b.render(f"FPS: {fps:.1f} | [H] ВЕРНУТЬ HUD | DAG: {' '.join(dag_parts[:3])}", True, (0, 255, 200))
            self.screen.blit(hud_txt, (50, 15))
            pygame.display.flip()
            return

        self.screen.fill((8, 12, 18))

        # Холст диффузии
        h, w = canvas_rgb.shape[:2]
        surf = pygame.image.frombuffer(canvas_rgb.tobytes(), (w, h), 'RGB')
        if (w, h) != (512, 384): surf = pygame.transform.scale(surf, (512, 384))
        self.screen.blit(surf, (370, 45))
        pygame.draw.rect(self.screen, (40, 55, 75), (370, 45, 512, 384), 2, border_radius=8)

        # -------------------------------------------------------------
        # 1. FPZ (Верхняя панель) — Эпистемический фазовый склон
        # -------------------------------------------------------------
        top_bar = pygame.Rect(370, 8, 512, 32)
        pygame.draw.rect(self.screen, (16, 22, 32), top_bar, border_radius=4)
        fpz_active = telemetry.get("fpz_active", False)
        if fpz_active:
            tension = telemetry.get("fpz_tension", 0.0)
            txt = f"Fpz [ФАКТ: {telemetry.get('fpz_concept', '')}]: {' '.join(dag_parts[:4])}"
            self.screen.blit(self.font_b.render(txt, True, (100, 255, 200)), (380, 12))
            
            # Индикатор натяжения фазового дрейфа
            arc_w = int(np.clip(tension, 0.0, 1.0) * 80)
            pygame.draw.rect(self.screen, (30, 40, 50), (790, 14, 80, 10), border_radius=2)
            col_t = (255, 80, 80) if tension > 0.45 else (100, 255, 180)
            pygame.draw.rect(self.screen, col_t, (790, 14, arc_w, 10), border_radius=2)
        else:
            self.screen.blit(self.font_b.render("Fpz: [ОТКЛЮЧЕН / НЕТ ДАТЧИКА]", True, (70, 80, 95)), (380, 12))

        # -------------------------------------------------------------
        # 2. Левая панель (Слоты первичного региона)
        # -------------------------------------------------------------
        px, py = 20, 45
        pygame.draw.rect(self.screen, (14, 18, 26), (px, py, 330, 555), border_radius=8)
        p_role = telemetry.get("primary_role", "НЕТ")
        p_active = telemetry.get("primary_active", False)
        col = (0, 255, 200) if p_active else (60, 70, 85)
        pygame.draw.rect(self.screen, col, (px, py, 330, 555), 1, border_radius=8)
        self.screen.blit(self.font_large.render(f"{p_role} [ФАКТ]: СЛОТЫ", True, col), (px + 15, py + 10))

        slot_names = telemetry.get("slot_names", [])
        slot_indices = telemetry.get("slot_indices", [])
        sealed_mask = telemetry.get("sealed_mask", [])
        stabs = telemetry.get("slot_stabilities", [])
        n_sectors = telemetry.get("total_clock_sectors", 8)
        cur_sec = telemetry.get("current_clock_sec", 0)

        if p_active:
            row_h = max(24, int(390 / max(1, n_sectors)))
            for s in range(n_sectors):
                sy = py + 52 + s * row_h
                s_box = pygame.Rect(px + 10, sy, 310, row_h - 2)
                name = slot_names[s] if s < len(slot_names) else ""
                c_id = slot_indices[s] if s < len(slot_indices) else 0
                is_seal = sealed_mask[s] if s < len(sealed_mask) else False
                c_rgb = (255, 140, 50) if is_seal else self.concept_colors[c_id % len(self.concept_colors)]

                pygame.draw.rect(self.screen, (20, 35, 45), s_box, border_radius=4)
                pygame.draw.rect(self.screen, c_rgb, s_box, 1, border_radius=4)
                seal_i = "🔒" if is_seal else "●"
                self.screen.blit(self.font_b.render(f"{seal_i}S{s+1} {name[:10]}", True, c_rgb), (px + 14, sy + 3))

                stab_w = int(stabs[s] * 50) if s < len(stabs) else 0
                pygame.draw.rect(self.screen, (30, 45, 40), (px + 185, sy + 6, 50, 7), border_radius=2)
                pygame.draw.rect(self.screen, (80, 255, 160), (px + 185, sy + 6, stab_w, 7), border_radius=2)
                star = " *" if (s == cur_sec) else ""
                self.screen.blit(self.font_s.render(f"Сек {s+1}{star}", True, (150, 160, 170)), (px + 245, sy + 4))

        # -------------------------------------------------------------
        # 3. Правая панель (Каузальный граф 89.5 Гц)
        # -------------------------------------------------------------
        gx, gy, gw, gh = 910, 45, 860, 555
        pygame.draw.rect(self.screen, (12, 16, 24), (gx, gy, gw, gh), border_radius=8)
        c_flow = telemetry.get("causal_flow_np")

        if p_active and c_flow is not None:
            pygame.draw.rect(self.screen, (100, 180, 255), (gx, gy, gw, gh), 1, border_radius=8)
            self.screen.blit(self.font_large.render(f"{p_role}: ГЕТЕРАРХИЯ СЛОТОВ (S={n_sectors})", True, (100, 180, 255)), (gx + 20, gy + 15))
            live_pos, dynamic_radii = self.gpu_graph.update_physics(n_sectors, c_flow, dt=dt)

            for i in range(n_sectors):
                for j in range(n_sectors):
                    if i != j and c_flow[i, j] > 0.015:
                        pygame.draw.line(self.screen, (40, 140, 200), (int(live_pos[i][0]), int(live_pos[i][1])), (int(live_pos[j][0]), int(live_pos[j][1])), 2)

            for s in range(n_sectors):
                pt = (int(live_pos[s][0]), int(live_pos[s][1]))
                name = slot_names[s] if s < len(slot_names) else ""
                c_id = slot_indices[s] if s < len(slot_indices) else 0
                c_rgb = self.concept_colors[c_id % len(self.concept_colors)]
                nr = int(dynamic_radii[s])
                pygame.draw.circle(self.screen, (c_rgb[0]//3, c_rgb[1]//3, c_rgb[2]//3), pt, nr + 6)
                pygame.draw.circle(self.screen, c_rgb, pt, nr)
                pygame.draw.circle(self.screen, (200, 220, 240), pt, nr, 2)
                txt = self.font_b.render(name[:8], True, (10, 20, 30))
                self.screen.blit(txt, (pt[0] - txt.get_width() // 2, pt[1] - 8))
        else:
            pygame.draw.rect(self.screen, (30, 35, 45), (gx, gy, gw, gh), 1, border_radius=8)
            self.screen.blit(self.font_large.render("ГЕТЕРАРХИЯ СЛОТОВ: [ОТКЛЮЧЕНА]", True, (70, 80, 95)), (gx + 20, gy + 15))

        # -------------------------------------------------------------
        # 4. Нижняя панель — ЧИСТЫЕ ФАЗОВЫЕ ПРИБОРЫ (НОЛЬ СКАЛЯРНОГО POWER)
        # -------------------------------------------------------------
        by = 620
        pygame.draw.rect(self.screen, (10, 14, 20), (20, by, 1750, 310), border_radius=8)
        pygame.draw.rect(self.screen, (40, 60, 80), (20, by, 1750, 310), 1, border_radius=8)

        # 4.1 Спираль Дельты
        dl_phase = telemetry.get("dl_phase", 0.0)
        if p_active:
            self.screen.blit(self.font_b.render(f"{p_role} [ФАКТ]: ТУННЕЛЬ ДЕЛЬТЫ", True, (0, 255, 200)), (35, by + 12))
            if telemetry.get("is_delta_tick", False):
                self.spiral.push(n_sectors, slot_indices)
            self.spiral.render(self.screen, 170, by + 160, 115, 26, dl_phase, n_sectors, slot_indices, self.concept_colors, cur_sec)
        else:
            self.screen.blit(self.font_b.render("ДЕЛЬТА: [ОТКЛЮЧЕНА]", True, (70, 80, 95)), (35, by + 12))
            pygame.draw.circle(self.screen, (16, 20, 28), (170, by + 160), 75, 1)

        # 4.2 F3 / F4: 2D Ранги SWM (Fan 2024) + Координатная шайба
        f3f4_x = 350
        f3_act, f4_act = telemetry.get("f3_active", False), telemetry.get("f4_active", False)
        f3_lbl = f"F3 [{telemetry.get('f3_concept','')}]" if f3_act else "F3 [НЕТ]"
        f4_lbl = f"F4 [{telemetry.get('f4_concept','')}]" if f4_act else "F4 [НЕТ]"
        col_f = (255, 200, 100) if (f3_act or f4_act) else (70, 80, 95)
        self.screen.blit(self.font_b.render(f"{f3_lbl} | {f4_lbl}", True, col_f), (f3f4_x, by + 12))

        # Манифолд 2D рангов со скользящей фазовой шайбой (Fan et al. 2024)
        pad_x, pad_y, pad_sz = f3f4_x + 10, by + 45, 190
        pygame.draw.rect(self.screen, (18, 24, 36) if (f3_act or f4_act) else (12, 15, 20), (pad_x, pad_y, pad_sz, pad_sz), border_radius=6)
        pygame.draw.rect(self.screen, (60, 80, 110) if (f3_act or f4_act) else (30, 40, 50), (pad_x, pad_y, pad_sz, pad_sz), 1, border_radius=6)
        pygame.draw.line(self.screen, (40, 55, 75), (pad_x, pad_y + pad_sz // 2), (pad_x + pad_sz, pad_y + pad_sz // 2), 1)
        pygame.draw.line(self.screen, (40, 55, 75), (pad_x + pad_sz // 2, pad_y), (pad_x + pad_sz // 2, pad_y + pad_sz), 1)
        self.screen.blit(self.font_s.render("Локальный (L)", True, (120, 140, 160)), (pad_x + 5, pad_y + pad_sz - 16))
        self.screen.blit(self.font_s.render("Глобал (G)", True, (120, 140, 160)), (pad_x + 5, pad_y + 4))

        if f3_act:
            xl, yg = telemetry.get("f3_x_local", 0.0), telemetry.get("f3_y_global", 0.0)
            puck_x = int(pad_x + (pad_sz / 2.0) + (xl * (pad_sz / 2.0) * 0.85))
            puck_y = int(pad_y + (pad_sz / 2.0) - (yg * (pad_sz / 2.0) * 0.85))
            r_sync = telemetry.get("f3_kuramoto", 0.0)
            puck_col = (int(100 + 155 * r_sync), int(255 * (1.0 - r_sync * 0.5)), 120)
            pygame.draw.circle(self.screen, puck_col, (puck_x, puck_y), 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (puck_x, puck_y), 9, 1)

        # Диафрагма фазовой энтропии F4 (справа от планшета)
        if f4_act:
            ent_x, ent_y = f3f4_x + 280, by + 140
            f4_ent = telemetry.get("f4_entropy", 0.0)
            ent_rad = int(np.clip(20 + f4_ent * 50, 15, 65))
            pygame.draw.circle(self.screen, (30, 20, 40), (ent_x, ent_y), 65)
            pygame.draw.circle(self.screen, (255, 100, 255), (ent_x, ent_y), ent_rad, 2)
            self.screen.blit(self.font_s.render(f"F4 Энтропия: {f4_ent:.2f}", True, (255, 180, 255)), (ent_x - 50, ent_y + 70))

        # 4.3 FCz: 4D Кинематический Гироскоп ([lx, ly] + rx сагитта + ry темпоральность)
        fcz_x = 750
        fcz_act = telemetry.get("fcz_active", False)
        if fcz_act:
            self.screen.blit(self.font_b.render(f"FCz [ФАКТ: {telemetry.get('fcz_concept','')}]", True, (100, 255, 120)), (fcz_x, by + 12))
            rcx, rcy = fcz_x + 100, by + 140
            pygame.draw.circle(self.screen, (20, 30, 42), (rcx, rcy), 70)
            pygame.draw.circle(self.screen, (40, 60, 80), (rcx, rcy), 70, 1)
            
            # Кольцо сагитты (кручение rx)
            rx_tilt = telemetry.get("fcz_rx", 0.0)
            ring_w, ring_h = int(70 * math.cos(rx_tilt * 1.5)), 70
            if ring_w > 5:
                pygame.draw.ellipse(self.screen, (100, 200, 255), (rcx - ring_w//2, rcy - ring_h//2, ring_w, ring_h), 1)

            # Вектор перемещения (lx, ly)
            lx, ly = telemetry.get("fcz_lx", 0.0), telemetry.get("fcz_ly", 0.0)
            # Стрела времени ry (темпоральный цвет)
            ry_val = telemetry.get("fcz_ry", 0.0)
            v_col = (100, 255, 120) if ry_val >= 0 else (100, 180, 255)
            pygame.draw.line(self.screen, v_col, (rcx, rcy), (int(rcx + lx * 60), int(rcy - ly * 60)), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(rcx + lx * 60), int(rcy - ly * 60)), 4)
            self.screen.blit(self.font_s.render(f"rx:{rx_tilt:+.2f} | ry:{ry_val:+.2f}", True, v_col), (fcz_x + 35, by + 235))
        else:
            self.screen.blit(self.font_b.render("FCz: [ОТКЛЮЧЕН]", True, (70, 80, 95)), (fcz_x, by + 12))
            pygame.draw.circle(self.screen, (16, 20, 28), (fcz_x + 100, by + 140), 70, 1)

        # 4.4 Палитра активного лора
        pal_x = 1050
        v_size = telemetry.get("active_vocab_size", 0)
        self.screen.blit(self.font_b.render(f"АКТИВНЫЙ ЛОР ({v_size} КОНЦЕПТОВ):", True, (100, 180, 255)), (pal_x, by + 12))
        for i in range(min(8, n_sectors)):
            name = slot_names[i] if (p_active and i < len(slot_names)) else ""
            c_id = slot_indices[i] if (p_active and i < len(slot_indices)) else 0
            c_col = self.concept_colors[c_id % len(self.concept_colors)] if p_active else (40, 50, 65)
            cx_b = pal_x + (i % 4) * 175
            cy_b = by + 45 + (i // 4) * 85
            pygame.draw.rect(self.screen, (20, 28, 40), (cx_b, cy_b, 165, 70), border_radius=6)
            pygame.draw.rect(self.screen, c_col, (cx_b, cy_b, 165, 70), 1, border_radius=6)
            self.screen.blit(self.font_b.render(f"S{i+1}: {name[:10]}", True, c_col), (cx_b + 10, cy_box := cy_b + 24))

        # Статус переключения ролей
        hot_s = f"👉 [TAB] Девайс: Dev {self.selected_dev_idx} | Роли: [1]AFz [2]F3 [3]F4 [4]FCz [5]Fpz [0]ВЫКЛ | [R]Сброс [C]Очистить"
        self.screen.blit(self.font_s.render(hot_s, True, (0, 255, 200)), (25, HEIGHT - 20))
        pygame.display.flip()

    def close(self):
        pygame.quit()
