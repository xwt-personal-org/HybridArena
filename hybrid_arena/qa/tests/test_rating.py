from hybrid_arena.qa.rating import EloRatingSystem, Rating, TrueSkillLikeRating


def test_elo_updates_winner_up_and_loser_down():
    system = EloRatingSystem(k_factor=32)
    winner, loser = system.update(Rating("a"), Rating("b"), score=1.0)
    assert winner.value > 1000.0
    assert loser.value < 1000.0


def test_trueskill_like_is_optional_interface():
    try:
        TrueSkillLikeRating().update(Rating("a"), Rating("b"), 1.0)
    except NotImplementedError as exc:
        assert "TrueSkill" in str(exc)
