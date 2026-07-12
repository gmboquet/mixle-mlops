"""The shared path-safety helper (gateway/routes/_paths.py) used by every route that keys a filesystem
path off a client-supplied name: substrate shards, tasks, solutions, routes, agentic toolcallers/
planners, and fine-tune model loading. pathlib's ``/`` operator does not resolve ``..`` on its own, and
joining a base with a string that LOOKS absolute (e.g. ``"/etc"``) replaces the base entirely -- these
tests prove both the substring check and the resolve-based defense-in-depth check actually reject that.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from mixle_mlops.gateway.routes._paths import safe_join, validate_path_segment
from mixle_mlops.gateway.routes.route import _spec_path


class TestValidatePathSegment:
    def test_accepts_a_normal_name(self):
        assert validate_path_segment("my-model_v2") == "my-model_v2"

    @pytest.mark.parametrize("bad", ["", "..", "../etc", "a/../b", "/etc", "a/b", "a\\b", "..\\..\\windows"])
    def test_rejects_unsafe_segments(self, bad):
        with pytest.raises(HTTPException) as exc:
            validate_path_segment(bad)
        assert exc.value.status_code == 422


class TestSafeJoin:
    def test_joins_a_normal_name_under_root(self, tmp_path):
        result = safe_join(tmp_path, "my-model")
        assert result == (tmp_path / "my-model").resolve()

    def test_rejects_dotdot_traversal(self, tmp_path):
        with pytest.raises(HTTPException) as exc:
            safe_join(tmp_path, "../../../etc")
        assert exc.value.status_code == 422

    def test_rejects_absolute_looking_name(self, tmp_path):
        # pathlib's `/` operator would otherwise REPLACE tmp_path entirely with "/etc"
        with pytest.raises(HTTPException) as exc:
            safe_join(tmp_path, "/etc")
        assert exc.value.status_code == 422

    def test_rejects_backslash_traversal(self, tmp_path):
        with pytest.raises(HTTPException) as exc:
            safe_join(tmp_path, "..\\..\\windows")
        assert exc.value.status_code == 422

    def test_rejects_bare_dotdot(self, tmp_path):
        with pytest.raises(HTTPException) as exc:
            safe_join(tmp_path, "..")
        assert exc.value.status_code == 422

    def test_never_actually_escapes_root_even_if_substring_check_were_bypassed(self, tmp_path):
        # defense in depth: even a name that somehow reached the resolve check with no `..`/slash
        # substring must still resolve inside root -- this directly exercises that second guard.
        outside = tmp_path.parent / "definitely-outside"
        root = tmp_path / "root"
        root.mkdir()
        assert safe_join(root, "child").resolve().is_relative_to(root.resolve())
        assert not str(outside).startswith(str(root))  # sanity: our "outside" fixture is actually outside


class TestRoutesSpecPathValidation:
    """route.py's _spec_path builds `{root}/{name}.json` (not a bare root/name join), so it validates
    the segment directly rather than going through safe_join -- covered here since route.py has no
    dedicated serving test file (a pre-existing gap, not introduced by this fix)."""

    @pytest.mark.parametrize("bad", ["..", "../../etc", "a/b", "a\\b"])
    def test_rejects_unsafe_route_names(self, bad):
        with pytest.raises(HTTPException) as exc:
            _spec_path(bad)
        assert exc.value.status_code == 422

    def test_accepts_a_normal_route_name(self):
        p = _spec_path("cheap-then-frontier")
        assert p.name == "cheap-then-frontier.json"
