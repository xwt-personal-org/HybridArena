"""Serializable planner state summaries derived from GameState."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PlannerState:
    step: int
    team: str
    ally_summary: dict
    enemy_summary: dict
    objective_summary: dict
    score_summary: dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_game_state(game_state, team: str) -> PlannerState:
    enemy_team = "blue" if team == "red" else "red"
    allies = [hero for hero in game_state.heroes.values() if hero.team == team]
    enemies = [hero for hero in game_state.heroes.values() if hero.team == enemy_team]
    visible_enemies = [
        enemy
        for enemy in enemies
        if enemy.alive and any(ally.alive and game_state._is_visible_to(ally, enemy) for ally in allies)
    ]

    ally_hp = [hero.hp_ratio for hero in allies if hero.alive]
    enemy_hp = [hero.hp_ratio for hero in visible_enemies]
    ally_gold = game_state.red_gold if team == "red" else game_state.blue_gold
    enemy_gold = game_state.blue_gold if team == "red" else game_state.red_gold

    return PlannerState(
        step=game_state.step_count,
        team=team,
        ally_summary={
            "alive": sum(1 for hero in allies if hero.alive),
            "low_hp": sum(1 for hero in allies if hero.alive and hero.hp_ratio < 0.3),
            "avg_hp": sum(ally_hp) / len(ally_hp) if ally_hp else 0.0,
        },
        enemy_summary={
            "visible": len(visible_enemies),
            "avg_visible_hp": sum(enemy_hp) / len(enemy_hp) if enemy_hp else 0.0,
        },
        objective_summary={
            "ally_tower_hp": game_state._structure_hp_sum(team, "tower"),
            "enemy_tower_hp": game_state._structure_hp_sum(enemy_team, "tower"),
            "ally_base_hp": game_state._structure_hp_sum(team, "base"),
            "enemy_base_hp": game_state._structure_hp_sum(enemy_team, "base"),
        },
        score_summary={
            "ally_gold": ally_gold,
            "enemy_gold": enemy_gold,
            "gold_diff": ally_gold - enemy_gold,
            "ally_kills": game_state.red_kills if team == "red" else game_state.blue_kills,
            "enemy_kills": game_state.blue_kills if team == "red" else game_state.red_kills,
        },
    )
