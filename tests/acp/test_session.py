"""Tests for acp_adapter.session — SessionManager and SessionState."""

import contextlib
import io
import json
import time
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from acp_adapter import session as acp_session
from acp_adapter.session import SessionManager, SessionState
from hermes_state import SessionDB


def _mock_agent():
    return MagicMock(name="MockAIAgent")


@pytest.fixture()
def manager():
    """SessionManager with a mock agent factory (avoids needing API keys)."""
    return SessionManager(agent_factory=_mock_agent)


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_create_session_returns_state(self, manager):
        state = manager.create_session(cwd="/tmp/work")
        assert isinstance(state, SessionState)
        assert state.cwd == "/tmp/work"
        assert state.session_id
        assert state.history == []
        assert state.agent is not None



    def test_register_task_cwd_translates_windows_drive_for_wsl_tools(self, monkeypatch):
        captured = {}

        def fake_register_task_env_overrides(task_id, overrides):
            captured["task_id"] = task_id
            captured["overrides"] = overrides

        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        monkeypatch.setattr(
            "tools.terminal_tool.register_task_env_overrides",
            fake_register_task_env_overrides,
        )

        acp_session._register_task_cwd("session-1", r"E:\Projects\AI\paperclip")

        assert captured == {
            "task_id": "session-1",
            "overrides": {"cwd": "/mnt/e/Projects/AI/paperclip"},
        }


    def test_get_session(self, manager):
        state = manager.create_session()
        fetched = manager.get_session(state.session_id)
        assert fetched is state


    def test_make_agent_stamps_session_cwd_for_codex_runtime(self, monkeypatch):
        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "acp_adapter.session.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.session_cwd == "/tmp/project"




# ---------------------------------------------------------------------------
# WSL cwd translation
# ---------------------------------------------------------------------------


class TestWslCwdTranslation:
    def test_translate_acp_cwd_converts_windows_drive_path_when_wsl(self, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)

        assert acp_session._translate_acp_cwd(r"E:\Projects\AI\paperclip") == "/mnt/e/Projects/AI/paperclip"





    def test_fork_session_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        original = manager.create_session(cwd="/tmp/base")

        forked = manager.fork_session(original.session_id, cwd=r"D:\work\project")

        assert forked is not None
        assert forked.cwd == "/mnt/d/work/project"

    def test_update_cwd_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        state = manager.create_session(cwd="/tmp/old")

        updated = manager.update_cwd(state.session_id, cwd=r"C:\Users\foo\project")

        assert updated is not None
        assert updated.cwd == "/mnt/c/Users/foo/project"

# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# list / cleanup / remove
# ---------------------------------------------------------------------------


class TestSymlinkAliasNormalization:
    """Ported from PrimeIntellect-ai/prime-agent#628 — symlink aliases of the
    same directory (macOS ``/var`` vs ``/private/var``, ``/tmp`` vs
    ``/private/tmp``) must compare equal, or ACP history filters silently drop
    a workspace's own sessions."""

    def test_symlink_alias_compares_equal(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(real)
        assert acp_session._normalize_cwd_for_compare(
            str(alias)
        ) == acp_session._normalize_cwd_for_compare(str(real))

    def test_distinct_dirs_still_compare_different(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert acp_session._normalize_cwd_for_compare(
            str(a)
        ) != acp_session._normalize_cwd_for_compare(str(b))

    def test_missing_path_keeps_lexical_normalization(self):
        # realpath(strict=False) is lexical for nonexistent paths, so cwds
        # that don't exist on this host (e.g. WSL-translated drives) behave
        # exactly as the old normpath comparison did.
        assert acp_session._normalize_cwd_for_compare(
            "/nonexistent-hermes-test/x/../y"
        ) == "/nonexistent-hermes-test/y"

    def test_list_sessions_matches_symlink_alias_cwd(self, manager, tmp_path):
        real = tmp_path / "proj"
        real.mkdir()
        alias = tmp_path / "link"
        alias.symlink_to(real)
        state = manager.create_session(cwd=str(real))
        state.history.append({"role": "user", "content": "hello"})
        listed = manager.list_sessions(cwd=str(alias))
        assert [s["session_id"] for s in listed] == [state.session_id]


# ---------------------------------------------------------------------------
# list / cleanup
# ---------------------------------------------------------------------------


class TestListAndCleanup:
    def test_list_sessions_empty(self, manager):
        assert manager.list_sessions() == []

    def test_list_sessions_includes_cli_session_for_workspace(self, manager):
        db = manager._get_db()
        db.create_session(
            session_id="cli-session-list",
            source="cli",
            model="test",
            cwd="/work/project",
        )
        db.append_message(
            session_id="cli-session-list",
            role="user",
            content="created from the TUI",
        )

        listed = manager.list_sessions(cwd="/work/project")

        assert [session["session_id"] for session in listed] == ["cli-session-list"]
        assert listed[0]["cwd"] == "/work/project"



    def test_save_session_preserves_existing_messages_on_encode_failure(self, manager):
        """Regression for #13675: a bad message in state.history must not
        clobber the previously-persisted transcript.  replace_messages()
        wraps DELETE + INSERT in a single rolled-back-on-exception txn.
        """
        state = manager.create_session()
        state.history.append({"role": "user", "content": "original"})
        manager.save_session(state.session_id)

        # Now swap history with a message whose tool_calls is non-JSON-serializable.
        # _execute_write rolls back; the previously persisted "original" stays.
        state.history = [
            {"role": "user", "content": "replacement"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"bad": object()}],
            },
        ]
        manager.save_session(state.session_id)

        db = manager._get_db()
        messages = db.get_messages_as_conversation(state.session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "original"
        assert isinstance(messages[0].get("timestamp"), (int, float))




    def test_cleanup_clears_all(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        s1.history.append({"role": "user", "content": "one"})
        s2.history.append({"role": "user", "content": "two"})
        assert len(manager.list_sessions()) == 2
        manager.cleanup()
        assert manager.list_sessions() == []

    def test_cleanup_does_not_delete_restored_cli_session(self, manager):
        db = manager._get_db()
        db.create_session(
            session_id="cli-session-cleanup",
            source="cli",
            model="test",
            cwd="/work/project",
        )
        db.append_message(
            session_id="cli-session-cleanup",
            role="user",
            content="owned by the CLI",
        )
        assert manager.get_session("cli-session-cleanup") is not None

        manager.cleanup()

        assert db.get_session("cli-session-cleanup") is not None
        assert db.get_messages_as_conversation("cli-session-cleanup")

    def test_remove_session(self, manager):
        state = manager.create_session()
        assert manager.remove_session(state.session_id) is True
        assert manager.get_session(state.session_id) is None
        # Removing again returns False
        assert manager.remove_session(state.session_id) is False


# ---------------------------------------------------------------------------
# persistence — sessions survive process restarts (via SessionDB)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Verify that sessions are persisted to SessionDB and can be restored."""














    def test_restores_cli_session_with_its_workspace(self, manager):
        db = manager._get_db()
        db.create_session(
            session_id="cli-session-123",
            source="cli",
            model="test",
            cwd="/work/project",
        )
        db.append_message(
            session_id="cli-session-123",
            role="user",
            content="resume this history",
        )

        restored = manager.get_session("cli-session-123")

        assert restored is not None
        assert restored.source == "cli"
        assert restored.cwd == "/work/project"
        assert restored.history[0]["content"] == "resume this history"

    def test_remove_does_not_delete_restored_cli_session(self, manager):
        db = manager._get_db()
        db.create_session(
            session_id="cli-session-remove",
            source="cli",
            model="test",
            cwd="/work/project",
        )
        db.append_message(
            session_id="cli-session-remove",
            role="user",
            content="owned by the CLI",
        )
        assert manager.get_session("cli-session-remove") is not None

        assert manager.remove_session("cli-session-remove") is False
        assert db.get_session("cli-session-remove") is not None
        assert db.get_messages_as_conversation("cli-session-remove")

    def test_sessions_searchable_via_fts(self, manager):
        """ACP sessions stored in SessionDB are searchable via FTS5."""
        state = manager.create_session()
        state.history.append({"role": "user", "content": "how do I configure nginx"})
        state.history.append({"role": "assistant", "content": "Here is the nginx config..."})
        manager.save_session(state.session_id)

        db = manager._get_db()
        results = db.search_messages("nginx")
        assert len(results) > 0
        session_ids = {r["session_id"] for r in results}
        assert state.session_id in session_ids


    def test_assistant_reasoning_fields_persisted(self, manager):
        """ACP session restore should preserve assistant reasoning context."""
        state = manager.create_session()
        state.history.append({
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        })
        manager.save_session(state.session_id)

        with manager._lock:
            del manager._sessions[state.session_id]

        restored = manager.get_session(state.session_id)
        assert restored is not None
        msg = restored.history[0]
        assert isinstance(msg.pop("timestamp", None), (int, float))
        # Load-time durability stamp (#92231): rows materialized from the DB
        # are marked persisted so a later flush can't re-append them.
        assert msg.pop("_db_persisted", None) is True
        assert restored.history == [{
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        }]


    def test_acp_agents_route_human_output_to_stderr(self, tmp_path, monkeypatch):
        """ACP agents must keep stdout clean for JSON-RPC stdio transport."""

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            return {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.example/v1",
                "api_key": "test-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(model=kwargs.get("model"), _print_fn=None)

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "test-model"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            state.agent._print_fn("ACP noise")

        assert stdout_buf.getvalue() == ""
        assert stderr_buf.getvalue() == "ACP noise\n"


# ---------------------------------------------------------------------------
# named custom provider identity across process restarts
# ---------------------------------------------------------------------------


def _fake_named_runtime_resolver(captured):
    def fake_resolve_runtime_provider(requested=None, **kwargs):
        captured["resolve_requested"] = requested
        resolved_base_url = (
            "https://openrouter.example/v1"
            if requested == "openrouter"
            else "https://autodl.example/v1"
        )
        return {
            "provider": "custom",
            "requested_provider": requested,
            "api_mode": "chat_completions",
            "base_url": resolved_base_url,
            "api_key": "no-key-required",
        }

    return fake_resolve_runtime_provider


class _CapturingAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.requested_provider = kwargs.get("requested_provider")
        self.base_url = kwargs.get("base_url")
        self.api_mode = kwargs.get("api_mode")
        self.api_key = kwargs.get("api_key")


class TestNamedCustomProviderRestore:
    def _setup(self, monkeypatch, tmp_path):
        captured = {}

        monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": {"provider": "custom", "default": "qwen3.8-27b"}},
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            _fake_named_runtime_resolver(captured),
        )
        monkeypatch.setattr(
            "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
            lambda **kwargs: None,
        )

        db = SessionDB(tmp_path / "state.db")
        return db, captured

    def test_restore_keeps_named_billing_provider_over_bare_model_config(self, monkeypatch, tmp_path):
        """A named custom provider must not be downgraded to bare ``custom``."""
        db, captured = self._setup(monkeypatch, tmp_path)
        db.create_session(
            session_id="sess-1",
            source="acp",
            model="qwen3.8-27b",
            model_config={"cwd": "/work", "provider": "custom"},
        )
        db.update_token_counts(
            "sess-1",
            input_tokens=10,
            output_tokens=5,
            billing_provider="custom:autodl",
            billing_base_url="https://autodl.example/v1",
        )

        manager = SessionManager(db=db)
        state = manager.get_session("sess-1")

        assert state is not None
        assert captured["resolve_requested"] == "custom:autodl"
        assert state.agent.requested_provider == "custom:autodl"
        assert state.agent.base_url == "https://autodl.example/v1"

    def test_restore_uses_persisted_requested_provider_when_present(self, monkeypatch, tmp_path):
        """A persisted precise ``requested_provider`` wins over a bare ``provider``."""
        db, captured = self._setup(monkeypatch, tmp_path)
        db.create_session(
            session_id="sess-2",
            source="acp",
            model="qwen3.8-27b",
            model_config={
                "cwd": "/work",
                "provider": "custom",
                "requested_provider": "custom:autodl",
            },
        )

        manager = SessionManager(db=db)
        state = manager.get_session("sess-2")

        assert state is not None
        assert captured["resolve_requested"] == "custom:autodl"

    def test_restore_prefers_current_non_custom_provider_over_stale_named_billing(self, monkeypatch, tmp_path):
        """Stale billing metadata must not override a later provider switch."""
        db, captured = self._setup(monkeypatch, tmp_path)
        db.create_session(
            session_id="sess-switched",
            source="acp",
            model="qwen3.8-27b",
            model_config={"cwd": "/work", "provider": "openrouter"},
        )
        db.update_token_counts(
            "sess-switched",
            input_tokens=10,
            output_tokens=5,
            billing_provider="custom:autodl",
            billing_base_url="https://autodl.example/v1",
        )

        manager = SessionManager(db=db)
        state = manager.get_session("sess-switched")

        assert state is not None
        assert captured["resolve_requested"] == "openrouter"
        assert state.agent.base_url == "https://openrouter.example/v1"

    def test_restore_uses_current_base_url_with_current_provider(self, monkeypatch, tmp_path):
        """Current model metadata must replace the stale billing route atomically."""
        db, captured = self._setup(monkeypatch, tmp_path)
        db.create_session(
            session_id="sess-current-route",
            source="acp",
            model="qwen3.8-27b",
            model_config={
                "cwd": "/work",
                "provider": "openrouter",
                "base_url": "https://current-openrouter.example/v1",
            },
        )
        db.update_token_counts(
            "sess-current-route",
            input_tokens=10,
            output_tokens=5,
            billing_provider="custom:autodl",
            billing_base_url="https://autodl.example/v1",
        )

        manager = SessionManager(db=db)
        state = manager.get_session("sess-current-route")

        assert state is not None
        assert captured["resolve_requested"] == "openrouter"
        assert state.agent.base_url == "https://current-openrouter.example/v1"

    def test_make_agent_passes_requested_provider_to_agent(self, monkeypatch, tmp_path):
        """_make_agent must propagate the precise identity into the agent."""
        db, captured = self._setup(monkeypatch, tmp_path)

        manager = SessionManager(db=db)
        agent = manager._make_agent(
            session_id="sess-3",
            cwd="/work",
            requested_provider="custom:autodl",
        )

        assert agent.requested_provider == "custom:autodl"

    def test_persist_records_named_requested_provider(self, monkeypatch, tmp_path):
        """_persist must persist the precise identity, not only the base type."""
        db, captured = self._setup(monkeypatch, tmp_path)

        manager = SessionManager(db=db)
        state = manager.create_session(cwd="/work")
        state.agent.provider = "custom"
        state.agent.requested_provider = "custom:autodl"
        state.agent.base_url = "https://autodl.example/v1"
        state.agent.api_mode = "chat_completions"
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        meta = json.loads(row["model_config"])
        assert meta.get("requested_provider") == "custom:autodl"
        assert meta.get("provider") == "custom"
