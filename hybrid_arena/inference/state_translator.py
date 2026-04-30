"""StateTranslator: Convert game state to natural language for LLM planner.

Why this layer exists (interview talking point):
    1. LLMs process natural language far better than raw numeric vectors.
    2. Translation filters irrelevant details, letting LLM focus on strategy.
    3. The translation quality itself becomes a tunable hyperparameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hybrid_arena.minimoba.game_engine import GameState


class StateTranslator:
    """Translate MiniMOBA GameState into a natural-language tactical summary."""

    def translate(self, game_state: GameState, team: str = "red") -> str:
        """Generate a text summary of the current game state.

        Args:
            game_state: Current GameState instance.
            team: Team to generate summary for ("red" or "blue").

        Returns:
            A formatted multi-line string describing the tactical situation.
        """
        parts = []

        # Header
        parts.append(f"## 当前局势 (第{game_state.step_count}步 / 最大{game_state.max_steps}步)")

        # Scoreboard
        if team == "red":
            my_kills, opp_kills = game_state.red_kills, game_state.blue_kills
            my_gold, opp_gold = game_state.red_gold, game_state.blue_gold
        else:
            my_kills, opp_kills = game_state.blue_kills, game_state.red_kills
            my_gold, opp_gold = game_state.blue_gold, game_state.red_gold

        parts.append(f"我方击杀:{my_kills}  敌方击杀:{opp_kills}  经济差:{my_gold - opp_gold:+.0f}")

        # My team status
        parts.append("\n## 我方队伍状态")
        for hero_id, hero in game_state.heroes.items():
            if not hero_id.startswith(team):
                continue
            hp_desc = self._hp_description(hero.hp / hero.max_hp)
            mp_desc = self._mp_description(hero.mp / hero.max_mp)
            cd_desc = self._cd_description(hero)
            alive = "存活" if hero.alive else "阵亡"
            parts.append(
                f"- {hero.role}({hero_id}): {alive}, {hp_desc}, {mp_desc}, "
                f"位置({hero.x},{hero.y}), {cd_desc}"
            )

        # Visible enemies
        enemy_team = "blue" if team == "red" else "red"
        visible = [
            h for hid, h in game_state.heroes.items()
            if hid.startswith(enemy_team) and h.alive
            and game_state._is_visible_to_team(h, team)
        ]
        parts.append(f"\n## 可见敌方 ({len(visible)}人)")
        for e in visible:
            parts.append(
                f"- 敌方{e.role}: {self._hp_description(e.hp / e.max_hp)}, 位置({e.x},{e.y})"
            )
        if len(visible) < game_state.team_size:
            parts.append(f"（有{game_state.team_size - len(visible)}名敌人在视野外）")

        # Buildings
        my_towers = sum(1 for t in game_state.towers if t.team == team and t.alive)
        opp_towers = sum(1 for t in game_state.towers if t.team == enemy_team and t.alive)
        parts.append(f"\n## 建筑: 我方塔{my_towers}座存活, 敌方塔{opp_towers}座存活")

        return "\n".join(parts)

    @staticmethod
    def _hp_description(ratio: float) -> str:
        if ratio > 0.8:
            return "血量充足"
        if ratio > 0.5:
            return "血量一般"
        if ratio > 0.2:
            return "血量较低⚠️"
        return "危险！血量极低🚨"

    @staticmethod
    def _mp_description(ratio: float) -> str:
        if ratio > 0.7:
            return "蓝量充足"
        if ratio > 0.3:
            return "蓝量一般"
        return "蓝量不足⚠️"

    @staticmethod
    def _cd_description(hero) -> str:
        cds = []
        if hero.skill_1_cd == 0:
            cds.append("技能1就绪")
        else:
            cds.append(f"技能1冷却{hero.skill_1_cd}")
        if hero.skill_2_cd == 0:
            cds.append("技能2就绪")
        else:
            cds.append(f"技能2冷却{hero.skill_2_cd}")
        return ", ".join(cds)
