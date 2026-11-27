"""M2d adds a task and changes nothing of M2b's: its observations for a fixed seed hash to what they did before M2d."""

from m2b_trace import trace

# computed on the commit before M2d (4e13c2e); regenerate only if M2b is meant to change
GOLDEN = "a29ee7dea3caa8fe8cc0302e8318ee6dd676c0ad8a06b9bb39e4dfb11823f78c"


def test_m2b_observations_for_a_fixed_seed_are_unchanged():
    assert trace() == GOLDEN
