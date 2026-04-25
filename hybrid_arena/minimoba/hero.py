"""Hero definitions: configs, pool, and runtime state for MiniMOBA."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillConfig:
    damage: float = 0.0
    range: int = 1
    cooldown: int = 3
    mp_cost: float = 30.0
    skill_type: str = "single"  # "aoe" | "single" | "heal" | "self" | "line"
    effect: str = ""  # "stun_N" | "slow_N" | "shield_N" | "vision_N" | ""
    heal: float = 0.0
    effect_duration: int = 0


@dataclass
class PassiveConfig:
    passive_type: str = ""  # "damage_reduction" | "crit_chance" | "mp_regen" | "hp_regen"
    value: float = 0.0


@dataclass
class HeroConfig:
    name: str
    role: str  # "tank" | "dps" | "support"
    max_hp: float
    max_mp: float
    attack_damage: float
    attack_range: int
    move_speed: int
    skill_1: SkillConfig = field(default_factory=SkillConfig)
    skill_2: SkillConfig = field(default_factory=SkillConfig)
    passive: PassiveConfig = field(default_factory=PassiveConfig)


HERO_POOL: dict[str, HeroConfig] = {
    "tank": HeroConfig(
        name="Guardian",
        role="tank",
        max_hp=1000.0,
        max_mp=300.0,
        attack_damage=30.0,
        attack_range=1,
        move_speed=1,
        skill_1=SkillConfig(
            damage=80.0,
            range=2,
            cooldown=5,
            mp_cost=40.0,
            skill_type="aoe",
            effect="stun_1",
            effect_duration=1,
        ),
        skill_2=SkillConfig(
            damage=0.0,
            range=0,
            cooldown=8,
            mp_cost=60.0,
            skill_type="self",
            effect="shield_200",
            effect_duration=3,
        ),
        passive=PassiveConfig(passive_type="damage_reduction", value=0.15),
    ),
    "dps": HeroConfig(
        name="Striker",
        role="dps",
        max_hp=600.0,
        max_mp=400.0,
        attack_damage=60.0,
        attack_range=3,
        move_speed=2,
        skill_1=SkillConfig(
            damage=150.0,
            range=4,
            cooldown=3,
            mp_cost=50.0,
            skill_type="single",
        ),
        skill_2=SkillConfig(
            damage=200.0,
            range=5,
            cooldown=10,
            mp_cost=80.0,
            skill_type="line",
            effect="slow_2",
            effect_duration=2,
        ),
        passive=PassiveConfig(passive_type="crit_chance", value=0.2),
    ),
    "support": HeroConfig(
        name="Sage",
        role="support",
        max_hp=500.0,
        max_mp=600.0,
        attack_damage=20.0,
        attack_range=2,
        move_speed=1,
        skill_1=SkillConfig(
            damage=0.0,
            range=3,
            cooldown=4,
            mp_cost=50.0,
            skill_type="heal",
            heal=120.0,
        ),
        skill_2=SkillConfig(
            damage=60.0,
            range=4,
            cooldown=6,
            mp_cost=40.0,
            skill_type="aoe",
            effect="vision_3",
            effect_duration=3,
        ),
        passive=PassiveConfig(passive_type="mp_regen", value=5.0),
    ),
}

DEFAULT_HERO_ASSIGNMENTS: list[str] = ["tank", "dps", "dps", "support"]


class HeroState:
    """Runtime state for a single hero during gameplay."""

    __slots__ = (
        "hero_id",
        "team",
        "config_name",
        "config",
        "x",
        "y",
        "hp",
        "mp",
        "level",
        "exp",
        "gold",
        "skill_1_cd",
        "skill_2_cd",
        "is_dead",
        "death_timer",
        "kills",
        "deaths",
        "assists",
        "damage_dealt",
        "heal_done",
        "stunned_turns",
        "slowed_turns",
        "shield_amount",
        "shield_turns",
    )

    def __init__(
        self,
        hero_id: str,
        team: str,
        config_name: str,
        config: HeroConfig,
        x: int = 0,
        y: int = 0,
    ):
        self.hero_id = hero_id
        self.team = team
        self.config_name = config_name
        self.config = config
        self.x = x
        self.y = y
        self.hp = config.max_hp
        self.mp = config.max_mp
        self.level = 1
        self.exp = 0.0
        self.gold = 0.0
        self.skill_1_cd = 0
        self.skill_2_cd = 0
        self.is_dead = False
        self.death_timer = 0
        self.kills = 0
        self.deaths = 0
        self.assists = 0
        self.damage_dealt = 0.0
        self.heal_done = 0.0
        self.stunned_turns = 0
        self.slowed_turns = 0
        self.shield_amount = 0.0
        self.shield_turns = 0

    @property
    def max_hp(self) -> float:
        return self.config.max_hp * (1.0 + 0.08 * (self.level - 1))

    @property
    def max_mp(self) -> float:
        return self.config.max_mp * (1.0 + 0.05 * (self.level - 1))

    @property
    def hp_ratio(self) -> float:
        return self.hp / self.max_hp if self.max_hp > 0 else 0.0

    @property
    def mp_ratio(self) -> float:
        return self.mp / self.max_mp if self.max_mp > 0 else 0.0

    @property
    def alive(self) -> bool:
        return not self.is_dead and self.hp > 0

    def take_damage(self, raw_damage: float) -> float:
        """Apply damage, accounting for shields and damage reduction."""
        if self.shield_amount > 0:
            if raw_damage <= self.shield_amount:
                self.shield_amount -= raw_damage
                return 0.0
            else:
                raw_damage -= self.shield_amount
                self.shield_amount = 0.0

        reduction = self.config.passive.value if self.config.passive.passive_type == "damage_reduction" else 0.0
        actual = raw_damage * (1.0 - reduction)
        self.hp = max(0.0, self.hp - actual)
        if self.hp <= 0:
            self.is_dead = True
            self.death_timer = 5
        return actual

    def heal(self, amount: float) -> float:
        """Heal the hero, capped at max HP."""
        old_hp = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - old_hp

    def grant_shield(self, amount: float, duration: int):
        self.shield_amount = amount
        self.shield_turns = duration

    def tick_cooldowns(self):
        """Decrement cooldowns and effect durations by one turn."""
        self.skill_1_cd = max(0, self.skill_1_cd - 1)
        self.skill_2_cd = max(0, self.skill_2_cd - 1)

        self.stunned_turns = max(0, self.stunned_turns - 1)
        self.slowed_turns = max(0, self.slowed_turns - 1)
        if self.shield_turns > 0:
            self.shield_turns -= 1
            if self.shield_turns == 0:
                self.shield_amount = 0.0

    def respawn(self):
        """Reset state on respawn."""
        self.is_dead = False
        self.death_timer = 0
        self.hp = self.max_hp * 0.7
        self.mp = self.max_mp * 0.5
        self.skill_1_cd = 0
        self.skill_2_cd = 0
        self.shield_amount = 0.0
        self.shield_turns = 0
        self.stunned_turns = 0
        self.slowed_turns = 0

    def gain_exp(self, amount: float):
        """Gain experience and level up if threshold met."""
        self.exp += amount
        exp_needed = 100.0 + (self.level - 1) * 80.0
        while self.exp >= exp_needed and self.level < 15:
            self.exp -= exp_needed
            self.level += 1
            self.hp = min(self.max_hp, self.hp + self.max_hp * 0.15)
            self.mp = min(self.max_mp, self.mp + self.max_mp * 0.10)
            exp_needed = 100.0 + (self.level - 1) * 80.0

    def can_act(self) -> bool:
        return self.alive and self.stunned_turns == 0
