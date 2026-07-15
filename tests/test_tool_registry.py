"""Searchable tool registry: retrieval returns the tools relevant to a goal, so a large registry
stays usable by the agent (focused subset per problem = broader scope + more reliable selection)."""

import pytest

from mixle_mlops.agent.tool_registry import AgentToolRegistry


def simulate_groundwater_contaminant(rate: float, x: float) -> dict:
    """Simulate a groundwater contaminant leak and return downstream concentrations."""
    return {}


def compute_mine_npv(grade: float, price: float) -> float:
    """Monte-Carlo net present value of a mining project from grade and commodity price."""
    return 0.0


def invert_gravity_survey(readings: list) -> dict:
    """Invert a gravity survey for subsurface density structure."""
    return {}


def fuse_model_estimates(values: list, variances: list) -> dict:
    """Fuse several independent numeric estimates into one, precision-weighted."""
    return {}


def classify_species_image(image_ref: str) -> str:
    """Identify a species from a photograph."""
    return ""


_TOOLS = {
    "simulate_groundwater_contaminant": simulate_groundwater_contaminant,
    "compute_mine_npv": compute_mine_npv,
    "invert_gravity_survey": invert_gravity_survey,
    "fuse_model_estimates": fuse_model_estimates,
    "classify_species_image": classify_species_image,
}


def test_retrieval_surfaces_the_on_topic_tool_for_a_geophysics_goal():
    reg = AgentToolRegistry(_TOOLS)
    top = reg.retrieve("invert a gravity survey to find dense buried rock", k=2)
    assert "invert_gravity_survey" in top


def test_retrieval_surfaces_the_economics_tool_for_a_valuation_goal():
    reg = AgentToolRegistry(_TOOLS)
    top = reg.retrieve("what is the net present value of this mine given grade and copper price", k=2)
    assert "compute_mine_npv" in top


def test_retrieval_returns_exactly_k_tools_ready_for_the_agent():
    reg = AgentToolRegistry(_TOOLS)
    top = reg.retrieve("estimate contaminant spread in groundwater", k=3)
    assert len(top) == 3
    assert all(callable(fn) for fn in top.values())


def test_scores_are_ranked_and_cover_every_tool():
    reg = AgentToolRegistry(_TOOLS)
    scored = reg.scores("fuse noisy estimates into one number")
    assert [name for name, _ in scored][0] == "fuse_model_estimates"  # best match on top
    assert len(scored) == len(_TOOLS)
    vals = [s for _, s in scored]
    assert vals == sorted(vals, reverse=True)


def test_empty_registry_is_rejected():
    with pytest.raises(ValueError):
        AgentToolRegistry({})
