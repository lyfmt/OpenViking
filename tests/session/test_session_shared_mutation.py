# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for shared user-scoped session mutations across agents."""

import asyncio

import pytest

from openviking.message import TextPart, ToolPart
from openviking.storage.transaction import get_lock_manager
from openviking_cli.exceptions import ConflictError
from tests.utils.mock_context import make_test_ctx

TEST_ACCOUNT_ID = "default"
TEST_USER_ID = "shared_session_user"
AGENT_A = "agent_alpha"
AGENT_B = "agent_beta"


def _make_shared_ctx(agent_id: str):
    return make_test_ctx(
        account_id=TEST_ACCOUNT_ID,
        user_id=TEST_USER_ID,
        agent_id=agent_id,
    )


async def _make_shared_sessions(client, session_id: str):
    service = client._client.service
    ctx_a = _make_shared_ctx(AGENT_A)
    ctx_b = _make_shared_ctx(AGENT_B)
    session_a = service.sessions.session(ctx_a, session_id)
    session_b = service.sessions.session(ctx_b, session_id)
    return service, ctx_a, ctx_b, session_a, session_b


class TestSharedSessionMutations:
    async def test_add_message_waits_for_short_lock_release(self, client, monkeypatch):
        service, ctx_a, _, session_a, session_b = await _make_shared_sessions(
            client, "shared_add_message_wait_test"
        )
        await session_a.ensure_exists()

        monkeypatch.setattr(
            "openviking.session.session._DEFAULT_SESSION_MUTATION_LOCK_TIMEOUT", 0.2
        )

        lock_manager = get_lock_manager()
        session_path = service.viking_fs._uri_to_path(session_a.uri, ctx=ctx_a)
        handle = lock_manager.create_handle()
        assert await lock_manager.acquire_subtree(handle, session_path, timeout=0.0) is True

        async def release_soon():
            await asyncio.sleep(0.05)
            await lock_manager.release(handle)

        release_task = asyncio.create_task(release_soon())
        try:
            msg = await session_b._add_message_async("user", [TextPart("waited write")])
        finally:
            await release_task

        await session_a.load(force=True)
        assert msg.content == "waited write"
        assert len(session_a.messages) == 1

    async def test_add_message_conflicts_after_timeout_when_shared_session_is_locked(
        self, client, monkeypatch
    ):
        service, ctx_a, _, session_a, session_b = await _make_shared_sessions(
            client, "shared_add_message_lock_test"
        )
        await session_a.ensure_exists()

        monkeypatch.setattr(
            "openviking.session.session._DEFAULT_SESSION_MUTATION_LOCK_TIMEOUT",
            0.01,
        )

        lock_manager = get_lock_manager()
        session_path = service.viking_fs._uri_to_path(session_a.uri, ctx=ctx_a)
        handle = lock_manager.create_handle()
        assert await lock_manager.acquire_subtree(handle, session_path, timeout=0.0) is True
        try:
            with pytest.raises(ConflictError, match="being modified by another operation"):
                await session_b._add_message_async("user", [TextPart("blocked write")])
        finally:
            await lock_manager.release(handle)

    async def test_update_tool_part_reload_preserves_newer_messages(self, client):
        service, ctx_a, _, session_a, session_b = await _make_shared_sessions(
            client, "shared_tool_update_reload_test"
        )

        tool_id = "tool_reload_001"
        tool_part = ToolPart(
            tool_id=tool_id,
            tool_name="search_tool",
            tool_uri=f"viking://session/shared_tool_update_reload_test/tools/{tool_id}",
            skill_uri="viking://agent/skills/search",
            tool_input={"query": "shared session"},
            tool_status="running",
        )
        msg = await session_a._add_message_async(
            "assistant",
            [TextPart("Running tool"), tool_part],
        )

        await session_b.load()
        await session_a._add_message_async("user", [TextPart("message added after stale load")])

        await session_b._update_tool_part_async(
            message_id=msg.id,
            tool_id=tool_id,
            output="done",
            status="completed",
        )

        reloaded = service.sessions.session(ctx_a, "shared_tool_update_reload_test")
        await reloaded.load(force=True)

        assert len(reloaded.messages) == 2
        assert reloaded.messages[1].content == "message added after stale load"
        tool_msg = reloaded.messages[0]
        updated_tool = tool_msg.find_tool_part(tool_id)
        assert updated_tool is not None
        assert updated_tool.tool_status == "completed"
        assert updated_tool.tool_output == "done"
