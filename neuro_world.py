#!/usr/bin/env python3
"""
neuro_world.py — Онтология N агентов рабочей памяти с поддержкой циклического катализа
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ArchetypeSpec:
    archetype_id: str
    element_tag: str = "WM"
    role_tag: str = "WM_ENTITY"
    display_name: str = "WM-Entity"
    essence_desc: str = ""
    action_desc: str = ""
    defense_desc: str = ""
    impact_effect_desc: str = ""


@dataclass
class StageEntitySlot:
    slot_id: str
    control_binding: str = "eeg:FCz"
    party_size: int = 1
    allowed_pool: list[str] = field(default_factory=list)

    @property
    def start_state_id(self) -> str:
        return self.allowed_pool[0] if self.allowed_pool else ""

    @property
    def target_state_id(self) -> str:
        return self.allowed_pool[-1] if self.allowed_pool else ""


@dataclass
class GenerativeQuestSpec:
    ongoing_hypothesis: str = ""
    completed_hypothesis: str = ""
    failed_hypothesis: str = ""
    completion_gain: float = 3.5
    entropy_drain: float = 0.2
    active_slot_idx: int = 0
    catalyst_slot_idx: int = -1
    driver_tag: str = "SELF_WM"
    reaction_formula: str = ""


@dataclass
class StageSpec:
    global_stage: int
    stage_title: str
    entities: list[StageEntitySlot]
    quest: GenerativeQuestSpec


@dataclass
class GameWorldManifest:
    world_name: str
    arena_context_desc: str = ""
    static_hold_desc: str = ""
    negative_desc: str = ""
    archetypes: dict[str, ArchetypeSpec] = field(default_factory=dict)
    stages: list[StageSpec] = field(default_factory=list)


def load_game_manifest(manifest_path: str) -> GameWorldManifest:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: '{manifest_path}'")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    archetypes: dict[str, ArchetypeSpec] = {}
    for k, v in raw.get("archetypes", {}).items():
        archetypes[k] = ArchetypeSpec(
            archetype_id=k,
            element_tag=str(v.get("element_tag", "WM")),
            role_tag=str(v.get("role_tag", "WM_ENTITY")),
            display_name=str(v.get("display_name", k)),
            essence_desc=str(v.get("essence_desc", "")),
            action_desc=str(v.get("action_desc", "")),
            defense_desc=str(v.get("defense_desc", "")),
            impact_effect_desc=str(v.get("impact_effect_desc", "")),
        )

    stages: list[StageSpec] = []
    for s_raw in raw.get("stages", []):
        st_num = int(s_raw.get("global_stage", len(stages) + 1))
        st_title = str(s_raw.get("stage_title", f"Stage_{st_num}"))

        entities_list: list[StageEntitySlot] = []
        for idx_e, e_raw in enumerate(s_raw.get("entities", [])):
            entities_list.append(
                StageEntitySlot(
                    slot_id=str(e_raw.get("slot_id", f"e{idx_e}")),
                    control_binding=str(e_raw.get("control_binding", "eeg:FCz")),
                    party_size=int(e_raw.get("party_size", 1)),
                    allowed_pool=list(e_raw.get("allowed_pool", [])),
                )
            )

        q_raw = s_raw.get("quest", {})
        quest_spec = GenerativeQuestSpec(
            ongoing_hypothesis=str(q_raw.get("ongoing_hypothesis", "")),
            completed_hypothesis=str(q_raw.get("completed_hypothesis", "")),
            failed_hypothesis=str(q_raw.get("failed_hypothesis", "")),
            completion_gain=float(q_raw.get("completion_gain", 3.5)),
            entropy_drain=float(q_raw.get("entropy_drain", 0.2)),
            active_slot_idx=int(q_raw.get("active_slot_idx", 0)),
            catalyst_slot_idx=int(q_raw.get("catalyst_slot_idx", -1)),
            driver_tag=str(q_raw.get("driver_tag", "SELF_WM")),
            reaction_formula=str(q_raw.get("reaction_formula", st_title)),
        )

        stages.append(StageSpec(global_stage=st_num, stage_title=st_title, entities=entities_list, quest=quest_spec))

    return GameWorldManifest(
        world_name=str(raw.get("world_name", "")),
        arena_context_desc=str(raw.get("arena_context_desc", "")),
        static_hold_desc=str(raw.get("static_hold_desc", "")),
        negative_desc=str(raw.get("negative_desc", "")),
        archetypes=archetypes,
        stages=stages,
    )


def compose_entity_semantics(
    entity_parties: list[list[str]],
    manifest: GameWorldManifest,
    stage: StageSpec | None = None,
) -> tuple[str, str, list[str], list[str], str]:
    """
    Формирует индивидуальные семантические зонды для всех N агентов, а в визуальном промпте
    фокусируется на взаимодействующей паре (Активный Агент + Катализатор), чтобы избежать
    текстурного слипания при большом числе N сущностей.
    """
    start_probes, target_probes, tags = [], [], []
    focused_start_sentences, focused_target_sentences = [], []

    if stage is not None:
        act_idx = stage.quest.active_slot_idx
        cat_idx = stage.quest.catalyst_slot_idx
        for idx, slot in enumerate(stage.entities):
            s0_id = slot.start_state_id
            s1_id = slot.target_state_id
            arch0 = manifest.archetypes.get(s0_id)
            arch1 = manifest.archetypes.get(s1_id, arch0)
            if arch0 and arch1:
                p0 = f"{manifest.arena_context_desc}. {arch0.essence_desc} {arch0.action_desc}."
                p1 = f"{manifest.arena_context_desc}. {arch1.essence_desc} {arch1.action_desc}."
                start_probes.append(p0)
                target_probes.append(p1)
                if s0_id != s1_id:
                    tags.append(f"{slot.slot_id}:{s0_id}->{s1_id}")
                else:
                    tags.append(f"{slot.slot_id}:{s0_id}")

                if idx == act_idx or idx == cat_idx:
                    focused_start_sentences.append(f"{arch0.essence_desc} {arch0.action_desc}.")
                    focused_target_sentences.append(f"{arch1.essence_desc} {arch1.action_desc}.")
    else:
        for party in entity_parties:
            for aid in party:
                arch = manifest.archetypes.get(aid)
                if arch:
                    s_txt = f"{arch.essence_desc} {arch.action_desc}."
                    focused_start_sentences.append(s_txt)
                    focused_target_sentences.append(s_txt)
                    p_txt = f"{manifest.arena_context_desc}. {s_txt}"
                    start_probes.append(p_txt)
                    target_probes.append(p_txt)
                    tags.append(arch.display_name)

    walk_in_prompt = f"{manifest.arena_context_desc}. " + " ".join(focused_start_sentences)
    scene_prompt = f"{manifest.arena_context_desc}. " + " ".join(focused_target_sentences)
    matchup_tag = " | ".join(tags)

    return scene_prompt, walk_in_prompt, start_probes, target_probes, matchup_tag
