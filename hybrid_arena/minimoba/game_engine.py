"""Core game engine for MiniMOBA: state, simulation, fog of war, combat."""

from __future__ import annotations

import numpy as np

from hybrid_arena.minimoba.action_encoding import N_ACTIONS, encode_action
from hybrid_arena.minimoba.hero import DEFAULT_HERO_ASSIGNMENTS, HERO_POOL, HeroState
from hybrid_arena.minimoba.map_generator import (
    BLUE_BASE,
    BLUE_TOWER,
    BUSH,
    EMPTY,
    OBSTACLE,
    RED_BASE,
    RED_TOWER,
    generate_map,
)
from hybrid_arena.minimoba.objectives import StructureState
from hybrid_arena.minimoba.reward_shaper import DEFAULT_REWARD_CONFIG, RewardConfig

MOVEMENT_DELTA = {
    0: (0, 0),
    1: (0, -1),
    2: (1, -1),
    3: (1, 0),
    4: (1, 1),
    5: (0, 1),
    6: (-1, 1),
    7: (-1, 0),
    8: (-1, -1),
}


class GameState:
    """Full game simulation state for MiniMOBA-4v4."""

    def __init__(
        self,
        map_size: int = 32,
        team_size: int = 4,
        hero_assignments: dict[str, str] | None = None,
        reward_config: RewardConfig | None = None,
        fog_of_war: bool = True,
        max_steps: int = 1000,
        seed: int | None = None,
    ):
        self.map_size = map_size
        self.team_size = team_size
        self.fog_of_war = fog_of_war
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)

        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG

        self.possible_agents = [f"red_{i}" for i in range(team_size)] + [
            f"blue_{i}" for i in range(team_size)
        ]
        self.agents: list[str] = self.possible_agents[:]

        self._default_assignments = {}
        for i, role in enumerate(DEFAULT_HERO_ASSIGNMENTS):
            self._default_assignments[f"red_{i}"] = role
            self._default_assignments[f"blue_{i}"] = role
        self.hero_assignments = hero_assignments or self._default_assignments

        # Placeholders — initialized in reset
        self.terrain: np.ndarray = np.zeros((map_size, map_size), dtype=np.int8)
        self.heroes: dict[str, HeroState] = {}
        self.structures: dict[str, StructureState] = {}
        self.step_count = 0
        self.red_kills = 0
        self.blue_kills = 0
        self.red_gold = 0
        self.blue_gold = 0
        self.red_towers = 2
        self.blue_towers = 2
        self.game_winner: str | None = None
        self.terminal_reason: str | None = None

        self.episode_rewards: dict[str, float] = {}

    def reset(self, seed: int | None = None):
        """Reset the game to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.red_kills = 0
        self.blue_kills = 0
        self.red_gold = 0
        self.blue_gold = 0
        self.red_towers = 2
        self.blue_towers = 2
        self.game_winner = None
        self.terminal_reason = None
        self.agents = self.possible_agents[:]

        self.terrain, spawns = generate_map(self.map_size, self.team_size, seed=self.rng.randint(0, 2**31))
        self._initialize_structures()

        self.heroes = {}
        for agent_id in self.possible_agents:
            config_name = self.hero_assignments[agent_id]
            config = HERO_POOL[config_name]
            team = "red" if agent_id.startswith("red") else "blue"
            team_spawns = spawns[team]
            idx = int(agent_id.split("_")[1])
            spawn_x, spawn_y = team_spawns[idx % len(team_spawns)]
            hero = HeroState(agent_id, team, config_name, config, x=spawn_x, y=spawn_y)
            self.heroes[agent_id] = hero

        self.episode_rewards = dict.fromkeys(self.possible_agents, 0.0)

    def _initialize_structures(self) -> None:
        """Initialize runtime state for towers and bases from the terrain map."""
        self.structures = {}
        structure_specs = [
            (RED_TOWER, "red", "tower", 1200.0),
            (BLUE_TOWER, "blue", "tower", 1200.0),
        ]
        for terrain_code, team, structure_type, max_hp in structure_specs:
            positions = np.argwhere(self.terrain == terrain_code)
            for idx, (y, x) in enumerate(positions):
                structure_id = f"{team}_{structure_type}_{idx}"
                self.structures[structure_id] = StructureState(
                    structure_id=structure_id,
                    team=team,
                    structure_type=structure_type,
                    x=int(x),
                    y=int(y),
                    max_hp=max_hp,
                    hp=max_hp,
                )

        for terrain_code, team in ((RED_BASE, "red"), (BLUE_BASE, "blue")):
            positions = np.argwhere(self.terrain == terrain_code)
            if positions.size == 0:
                continue
            y = int(round(float(positions[:, 0].mean())))
            x = int(round(float(positions[:, 1].mean())))
            structure_id = f"{team}_base_0"
            self.structures[structure_id] = StructureState(
                structure_id=structure_id,
                team=team,
                structure_type="base",
                x=x,
                y=y,
                max_hp=2000.0,
                hp=2000.0,
            )

        self._sync_structure_counts()

    def step(self, actions: dict[str, np.ndarray]) -> dict[str, float]:
        """Execute one game step with simultaneous actions.

        Args:
            actions: {agent_id: np.array([move_dir, skill_choice, target_choice])}

        Returns:
            Per-agent rewards for this step.
        """
        self.step_count += 1
        step_rewards = dict.fromkeys(self.possible_agents, self.reward_config.time_penalty)

        # Process only alive, active agents
        active = {}
        for agent_id in self.agents:
            hero = self.heroes.get(agent_id)
            if hero is None or not hero.can_act():
                continue
            if agent_id in actions:
                active[agent_id] = self._parse_action(actions[agent_id])

        # Phase 1: Movement (simultaneous, with collision)
        movements = {}
        for agent_id, (move_dir, _, _) in active.items():
            hero = self.heroes[agent_id]
            dx, dy = MOVEMENT_DELTA.get(move_dir, (0, 0))
            speed = hero.config.move_speed
            # Step through intermediate tiles to validate path
            target_x, target_y = hero.x, hero.y
            valid = True
            for _ in range(speed):
                nx = target_x + dx
                ny = target_y + dy
                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                    valid = False
                    break
                if self.terrain[ny, nx] == OBSTACLE:
                    valid = False
                    break
                target_x, target_y = nx, ny
            if valid and (target_x != hero.x or target_y != hero.y):
                movements[agent_id] = (target_x, target_y)

        # Resolve collisions: no two heroes can occupy the same target tile
        target_counts: dict[tuple[int, int], list[str]] = {}
        for agent_id, pos in movements.items():
            target_counts.setdefault(pos, []).append(agent_id)

        for agent_id, pos in movements.items():
            if len(target_counts[pos]) == 1:
                hero = self.heroes[agent_id]
                hero.x, hero.y = pos

        # Phase 2: Combat (simultaneous attacks)
        for agent_id, (_, skill_choice, target_choice) in active.items():
            hero = self.heroes[agent_id]
            if not hero.can_act():
                continue
            self._execute_attack(agent_id, hero, skill_choice, target_choice, step_rewards)

        # Phase 3: Tick effects, cooldowns, respawns
        for hero in self.heroes.values():
            hero.tick_cooldowns()
            if hero.is_dead:
                hero.death_timer -= 1
                if hero.death_timer <= 0:
                    hero.respawn()
                    if hero.team == "red":
                        hero.x, hero.y = self.map_size - 2, 2
                    else:
                        hero.x, hero.y = 2, self.map_size - 2

            passive = hero.config.passive
            if hero.alive:
                if passive.passive_type == "mp_regen":
                    hero.mp = min(hero.max_mp, hero.mp + passive.value)
                if passive.passive_type == "hp_regen":
                    hero.hp = min(hero.max_hp, hero.hp + passive.value)

        # Game end: win/lose rewards added by env.py, not here
        if self.is_game_over():
            self.game_winner = self.get_winner()

        # Track cumulative episode rewards
        for a in self.possible_agents:
            self.episode_rewards[a] = self.episode_rewards.get(a, 0.0) + step_rewards.get(a, 0.0)

        return step_rewards

    def _parse_action(self, action) -> tuple[int, int, int]:
        """Parse action into (move_dir, skill_choice, target_choice)."""
        action = np.atleast_1d(np.asarray(action, dtype=np.int64))
        if len(action) < 3:
            return (0, 3, 8)
        return int(action[0]) % 9, int(action[1]) % 4, int(action[2]) % 9

    def _execute_attack(
        self, agent_id: str, hero: HeroState, skill_choice: int, target_choice: int,
        step_rewards: dict[str, float],
    ):
        """Execute an attack or skill."""
        if skill_choice == 3:
            return  # no attack

        # Determine target
        target_hero: HeroState | None = None
        if skill_choice == 0:
            enemies = self._get_nearby_enemies(hero, hero.config.attack_range)
            if enemies and target_choice < len(enemies):
                target_hero = enemies[target_choice]
        elif skill_choice == 1:
            if hero.skill_1_cd > 0 or hero.mp < hero.config.skill_1.mp_cost:
                return
            enemies = self._get_nearby_enemies(hero, hero.config.skill_1.range)
            if enemies and target_choice < len(enemies):
                target_hero = enemies[target_choice]
        elif skill_choice == 2:
            if hero.skill_2_cd > 0 or hero.mp < hero.config.skill_2.mp_cost:
                return
            enemies = self._get_nearby_enemies(hero, hero.config.skill_2.range)
            if enemies and target_choice < len(enemies):
                target_hero = enemies[target_choice]

        if skill_choice == 0:
            if target_hero and target_hero.alive:
                raw_dmg = hero.config.attack_damage
                if hero.config.passive.passive_type == "crit_chance" and self.rng.random() < hero.config.passive.value:
                    raw_dmg *= 2.0
                actual = target_hero.take_damage(raw_dmg)
                hero.damage_dealt += actual
                step_rewards[agent_id] += self.reward_config.damage * actual
                if target_hero.is_dead:
                    self._handle_kill(agent_id, target_hero, step_rewards)
            else:
                target_structure = self._get_nearby_enemy_structure(hero, hero.config.attack_range)
                if target_structure is not None:
                    self._damage_structure(agent_id, hero, target_structure, step_rewards)

        elif skill_choice == 1:
            skill = hero.config.skill_1
            hero.mp -= skill.mp_cost
            hero.skill_1_cd = skill.cooldown
            self._apply_skill(agent_id, hero, skill, target_hero, step_rewards)

        elif skill_choice == 2:
            skill = hero.config.skill_2
            hero.mp -= skill.mp_cost
            hero.skill_2_cd = skill.cooldown
            self._apply_skill(agent_id, hero, skill, target_hero, step_rewards)

    def _apply_skill(
        self, agent_id: str, caster: HeroState, skill, target: HeroState | None,
        step_rewards: dict[str, float],
    ):
        """Apply a skill's effects."""
        skill_type = skill.skill_type

        if skill_type == "heal":
            heal_target = caster
            best_ratio = caster.hp_ratio
            for _aid, h in self.heroes.items():
                if h.team == caster.team and h.alive and h.hp_ratio < best_ratio:
                    best_ratio = h.hp_ratio
                    heal_target = h
            amount = heal_target.heal(skill.heal)
            caster.heal_done += amount
            step_rewards[agent_id] += self.reward_config.heal * amount
            return

        if skill_type == "self":
            if skill.effect.startswith("shield_"):
                shield_val = float(skill.effect.split("_")[1])
                caster.grant_shield(shield_val, skill.effect_duration)
            return

        # Offensive skills
        targets: list[HeroState] = []
        if skill_type == "aoe":
            if target:
                targets.append(target)
                for _aid, h in self.heroes.items():
                    if h.team != caster.team and h.alive and h is not target:
                        if abs(h.x - target.x) + abs(h.y - target.y) <= 1:
                            targets.append(h)
        elif skill_type == "line":
            if target and target.alive:
                targets.append(target)
                dx = target.x - caster.x
                dy = target.y - caster.y
                length = max(abs(dx), abs(dy))
                if length > 0:
                    step_x = dx / length
                    step_y = dy / length
                    for i in range(1, length):
                        cx = int(caster.x + step_x * i)
                        cy = int(caster.y + step_y * i)
                        for _aid, h in self.heroes.items():
                            if h.team != caster.team and h.alive and h.x == cx and h.y == cy:
                                if h not in targets:
                                    targets.append(h)
        else:
            if target and target.alive:
                targets = [target]

        for tgt in targets:
            raw_dmg = skill.damage
            actual = tgt.take_damage(raw_dmg)
            caster.damage_dealt += actual
            step_rewards[agent_id] += self.reward_config.damage * actual

            if skill.effect.startswith("stun_"):
                tgt.stunned_turns = skill.effect_duration
            elif skill.effect.startswith("slow_"):
                tgt.slowed_turns = skill.effect_duration
            elif skill.effect.startswith("vision_"):
                pass  # fog of war benefit

            if tgt.is_dead:
                self._handle_kill(agent_id, tgt, step_rewards)

    def _handle_kill(
        self, killer_id: str, victim: HeroState, step_rewards: dict[str, float],
    ):
        victim.deaths += 1
        killer = self.heroes[killer_id]
        killer.kills += 1
        gold_reward = 150.0 + 20.0 * (victim.level - 1)
        exp_reward = 80.0 + 40.0 * (victim.level - 1)
        killer.gold += gold_reward
        killer.gain_exp(exp_reward)
        if killer.team == "red":
            self.red_gold += gold_reward
        else:
            self.blue_gold += gold_reward

        step_rewards[killer_id] += self.reward_config.kill
        step_rewards[victim.hero_id] += self.reward_config.death

        # Assist: nearby allies get half rewards
        for aid, hero in self.heroes.items():
            if aid == killer_id or hero.team != killer.team:
                continue
            if hero.alive and abs(hero.x - victim.x) + abs(hero.y - victim.y) <= 5:
                hero.assists += 1
                hero.gold += gold_reward * 0.5
                hero.gain_exp(exp_reward * 0.5)
                step_rewards[aid] += self.reward_config.assist

        if killer.team == "red":
            self.red_kills += 1
        else:
            self.blue_kills += 1

    def _get_nearby_enemy_structure(
        self,
        hero: HeroState,
        max_range: int,
    ) -> StructureState | None:
        candidates = []
        for structure in self.structures.values():
            if structure.team == hero.team or not structure.alive:
                continue
            if structure.structure_type == "base" and self._alive_tower_count(structure.team) > 0:
                continue
            dist = abs(hero.x - structure.x) + abs(hero.y - structure.y)
            if dist <= max_range:
                candidates.append((dist, structure))
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1] if candidates else None

    def _damage_structure(
        self,
        attacker_id: str,
        attacker: HeroState,
        structure: StructureState,
        step_rewards: dict[str, float],
    ) -> None:
        was_alive = structure.alive
        actual = structure.take_damage(attacker.config.attack_damage)
        if actual <= 0:
            return
        attacker.damage_dealt += actual
        step_rewards[attacker_id] += self.reward_config.damage * actual

        if was_alive and not structure.alive:
            self._handle_structure_destroy(attacker_id, attacker.team, structure, step_rewards)

    def _handle_structure_destroy(
        self,
        attacker_id: str,
        attacker_team: str,
        structure: StructureState,
        step_rewards: dict[str, float],
    ) -> None:
        if structure.structure_type == "tower":
            step_rewards[attacker_id] += self.reward_config.tower
            for agent_id, hero in self.heroes.items():
                if hero.team == structure.team:
                    step_rewards[agent_id] += self.reward_config.tower_lost
            if attacker_team == "red":
                self.red_gold += 300
            else:
                self.blue_gold += 300
            self._sync_structure_counts()
            return

        if structure.structure_type == "base":
            step_rewards[attacker_id] += getattr(self.reward_config, "base", 0.0)
            self.game_winner = attacker_team
            self.terminal_reason = "base_destroyed"

    def _sync_structure_counts(self) -> None:
        self.red_towers = self._alive_tower_count("red")
        self.blue_towers = self._alive_tower_count("blue")

    def _alive_tower_count(self, team: str) -> int:
        return sum(
            1
            for structure in self.structures.values()
            if structure.team == team and structure.structure_type == "tower" and structure.alive
        )

    def _structure_hp_sum(self, team: str, structure_type: str) -> float:
        return float(
            sum(
                structure.hp
                for structure in self.structures.values()
                if structure.team == team and structure.structure_type == structure_type
            )
        )

    def _objective_hp_sum(self, team: str) -> float:
        return self._structure_hp_sum(team, "tower") + self._structure_hp_sum(team, "base")

    def get_objective_info(self, team: str) -> dict[str, float]:
        enemy = "blue" if team == "red" else "red"
        return {
            "team_gold": self.red_gold if team == "red" else self.blue_gold,
            "enemy_gold": self.blue_gold if team == "red" else self.red_gold,
            "ally_tower_hp": self._structure_hp_sum(team, "tower"),
            "enemy_tower_hp": self._structure_hp_sum(enemy, "tower"),
            "ally_base_hp": self._structure_hp_sum(team, "base"),
            "enemy_base_hp": self._structure_hp_sum(enemy, "base"),
        }

    def _get_nearby_enemies(self, hero: HeroState, max_range: int) -> list[HeroState]:
        """Return visible enemy heroes within range, sorted by distance."""
        enemies = []
        for aid, other in self.heroes.items():
            if other.team == hero.team or not other.alive:
                continue
            if not self._is_visible_to(hero, other):
                continue
            dist = abs(hero.x - other.x) + abs(hero.y - other.y)
            if dist <= max_range:
                enemies.append((dist, other))
        enemies.sort(key=lambda x: x[0])
        return [e[1] for e in enemies]

    def _is_visible_to(self, observer: HeroState, target: HeroState) -> bool:
        """Check if target is visible to observer (fog of war)."""
        if not self.fog_of_war:
            return True
        dist = abs(observer.x - target.x) + abs(observer.y - target.y)
        vision_range = 5
        # Enemies in bushes harder to see
        tx, ty = target.x, target.y
        if 0 <= ty < self.map_size and 0 <= tx < self.map_size and self.terrain[ty, tx] == BUSH:
            vision_range = 2
        return dist <= vision_range

    def get_observation(self, agent_id: str) -> dict:
        """Build the observation dict for a specific agent."""
        hero = self.heroes.get(agent_id)
        if hero is None:
            return self._empty_obs()

        team = hero.team
        radius = 5

        # Local map channels: (channels, 11, 11) -> will be (11, 11, channels) in wrapper
        local_map = self._build_local_map(hero, radius)

        # Self state: (20,)
        self_state = self._build_self_state(hero)

        # Teammate states: (3, 15)
        teammates = [a for a in self.possible_agents if a.startswith(team) and a != agent_id]
        teammate_states = np.zeros((3, 15), dtype=np.float32)
        for i, tid in enumerate(teammates[:3]):
            th = self.heroes.get(tid)
            if th and th.alive:
                teammate_states[i] = self._build_teammate_vector(th)

        # Global info: (10,)
        total_objective_hp = sum(s.max_hp for s in self.structures.values()) or 1.0
        # Fixed global red-minus-blue perspective; agents can combine this with team id.
        objective_advantage = (
            self._objective_hp_sum("red") - self._objective_hp_sum("blue")
        ) / total_objective_hp
        global_info = np.array(
            [
                min(self.step_count / self.max_steps, 1.0),
                min(self.red_kills / 30.0, 1.0),
                min(self.blue_kills / 30.0, 1.0),
                np.clip((self.red_gold - self.blue_gold) / 5000.0, -1.0, 1.0),
                self.red_towers / 3.0,
                self.blue_towers / 3.0,
                hero.hp_ratio,
                hero.mp_ratio,
                hero.level / 15.0,
                np.clip(objective_advantage, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        # Action mask: (324,) reshaped from [9, 4, 9]
        action_mask = self._build_action_mask(hero)

        return {
            "local_map": local_map,  # (11, 11, 11)
            "self_state": self_state,  # (20,)
            "teammate_states": teammate_states,  # (3, 15)
            "global_info": global_info,  # (10,)
            "action_mask": action_mask,  # (324,)
        }

    def _empty_obs(self) -> dict:
        return {
            "local_map": np.zeros((11, 11, 11), dtype=np.float32),
            "self_state": np.zeros((20,), dtype=np.float32),
            "teammate_states": np.zeros((3, 15), dtype=np.float32),
            "global_info": np.zeros((10,), dtype=np.float32),
            "action_mask": np.ones((N_ACTIONS,), dtype=np.int8),
        }

    def _build_local_map(self, hero: HeroState, radius: int) -> np.ndarray:
        """Build (11, 11, 11) local map centered on hero."""
        channels = 11
        local = np.zeros((channels, radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
        hx, hy = hero.x, hero.y

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                wx, wy = hx + dx, hy + dy
                if not (0 <= wx < self.map_size and 0 <= wy < self.map_size):
                    continue
                lx, ly = dx + radius, dy + radius

                tile = self.terrain[wy, wx]
                # Channel 0: terrain/path
                local[0, ly, lx] = 1.0 if tile in (EMPTY, RED_BASE, BLUE_BASE) else 0.0
                # Channel 1: obstacle
                local[1, ly, lx] = 1.0 if tile == OBSTACLE else 0.0
                # Channel 2: bush
                local[2, ly, lx] = 1.0 if tile == BUSH else 0.0
                # Channel 3: red tower
                local[3, ly, lx] = 1.0 if tile == RED_TOWER else 0.0
                # Channel 4: blue tower
                local[4, ly, lx] = 1.0 if tile == BLUE_TOWER else 0.0
                # Channel 5: red base
                local[5, ly, lx] = 1.0 if tile == RED_BASE else 0.0
                # Channel 6: blue base
                local[6, ly, lx] = 1.0 if tile == BLUE_BASE else 0.0

                # Unit channels
                for aid, other in self.heroes.items():
                    if other.x != wx or other.y != wy or not other.alive:
                        continue
                    if not self._is_visible_to(hero, other) and other.team != hero.team:
                        continue
                    if other.team == hero.team:
                        ch = 7  # allies
                        local[ch, ly, lx] = 1.0
                    else:
                        ch = 8  # enemies
                        local[ch, ly, lx] = 1.0

        # Position encoding: normalized to [0, 1]
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        pos_enc = (offsets / radius + 1.0) / 2.0  # maps [-radius, radius] to [0, 1]
        local[9, :, :] = pos_enc.reshape(1, -1)  # x positions
        local[10, :, :] = pos_enc.reshape(-1, 1)  # y positions

        # Return as (11, 11, channels) for the protocol
        return local.transpose(1, 2, 0)

    def _build_self_state(self, hero: HeroState) -> np.ndarray:
        skill1_cd = hero.config.skill_1.cooldown
        skill2_cd = hero.config.skill_2.cooldown
        return np.array(
            [
                hero.hp_ratio,
                hero.mp_ratio,
                hero.level / 15.0,
                hero.skill_1_cd / max(skill1_cd, 1),
                hero.skill_2_cd / max(skill2_cd, 1),
                hero.x / self.map_size,
                hero.y / self.map_size,
                hero.config.attack_damage / 200.0,
                hero.gold / 5000.0,
                hero.exp / 1500.0,
                1.0 if hero.stunned_turns > 0 else 0.0,
                1.0 if hero.slowed_turns > 0 else 0.0,
                hero.shield_amount / 500.0,
                float(hero.kills) / 20.0,
                float(hero.deaths) / 10.0,
                float(hero.assists) / 20.0,
                hero.damage_dealt / 5000.0,
                hero.heal_done / 3000.0,
                float(hero.alive),
                self.step_count / self.max_steps,
            ],
            dtype=np.float32,
        )

    def _build_teammate_vector(self, hero: HeroState) -> np.ndarray:
        skill1_cd = hero.config.skill_1.cooldown
        skill2_cd = hero.config.skill_2.cooldown
        return np.array(
            [
                hero.hp_ratio,
                hero.mp_ratio,
                hero.level / 15.0,
                hero.skill_1_cd / max(skill1_cd, 1),
                hero.skill_2_cd / max(skill2_cd, 1),
                hero.x / self.map_size,
                hero.y / self.map_size,
                float(hero.config.role == "tank"),
                float(hero.config.role == "dps"),
                float(hero.config.role == "support"),
                float(hero.kills) / 20.0,
                float(hero.deaths) / 10.0,
                float(hero.alive),
                hero.shield_amount / 500.0,
                1.0 if hero.stunned_turns > 0 else 0.0,
            ],
            dtype=np.float32,
        )

    def _build_action_mask(self, hero: HeroState) -> np.ndarray:
        """Build (324,) boolean mask. 1 = legal, 0 = illegal."""
        mask = np.ones(N_ACTIONS, dtype=np.int8)
        if not hero.can_act():
            # Only "no move + no attack" is valid
            mask[:] = 0
            mask[encode_action(0, 3, 8)] = 1  # move=0, skill=none, target=none
            return mask

        # Movement: all 9 directions valid by default (walls handled in step)
        # Skill: if on cooldown or insufficient MP, mask out
        for move_dir in range(9):
            # Skill 0 (auto-attack): always valid
            # Skill 1: valid if not on cooldown and enough MP
            if hero.skill_1_cd > 0 or hero.mp < hero.config.skill_1.mp_cost:
                for t in range(9):
                    mask[encode_action(move_dir, 1, t)] = 0
            # Skill 2: valid if not on cooldown and enough MP
            if hero.skill_2_cd > 0 or hero.mp < hero.config.skill_2.mp_cost:
                for t in range(9):
                    mask[encode_action(move_dir, 2, t)] = 0
            # Skill 3 (no attack): always valid
            # But target selection only matters for skill 3's "no target" (target 8)
            for t in range(8):
                mask[encode_action(move_dir, 3, t)] = 0  # only target 8 valid for "no attack"

        return mask

    def is_game_over(self) -> bool:
        if self.step_count >= self.max_steps:
            if self.terminal_reason is None:
                self.terminal_reason = "timeout"
            return True
        if self.game_winner is not None:
            return True
        return False

    def get_winner(self) -> str:
        if self.game_winner:
            return self.game_winner
        if self.red_towers > self.blue_towers:
            return "red"
        if self.blue_towers > self.red_towers:
            return "blue"
        if self.red_gold > self.blue_gold:
            return "red"
        if self.blue_gold > self.red_gold:
            return "blue"
        if self.red_kills > self.blue_kills:
            return "red"
        elif self.blue_kills > self.red_kills:
            return "blue"
        return "draw"
