"""Provider-independent coding-agent harness with tools and durable sessions."""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import Agent, FatalPersistenceError
from .cancellation import Cancellation, OperationAbortedError
from .compaction import CompactionResult
from .config import AgentConfig, CompactionSettings, RetrySettings, default_session_dir
from .events import event_to_json
from .harness import AsyncResource, ResourceOwner, create_harness
from .permissions import PermissionPolicy, PermissionRequest, PermissionRule
from .provider import ModelValidator, Provider, ProviderOptions, ProviderRequest
from .session import AgentSession, create_agent_session
from .session_store import (
    SessionCorruptError,
    SessionInfo,
    SessionLockError,
    SessionStore,
)
from .skills import ResourceDiagnostic, SkillInfo, SkillSettings
from .tool_hooks import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolHooks,
    ToolProvenance,
)
from .tools import create_builtin_tools, create_coding_tools
from .tools.base import Tool, ToolResult, ToolValidationError
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    ModelInfo,
    StopReason,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    Usage,
    UserMessage,
)

__all__ = [
    "AfterToolCallContext",
    "AfterToolCallResult",
    "Agent",
    "AgentConfig",
    "AgentSession",
    "AsyncResource",
    "AssistantMessage",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "Cancellation",
    "CompactionResult",
    "CompactionSettings",
    "ContentBlock",
    "FatalPersistenceError",
    "Message",
    "ModelInfo",
    "ModelValidator",
    "OperationAbortedError",
    "PermissionPolicy",
    "PermissionRequest",
    "PermissionRule",
    "Provider",
    "ProviderOptions",
    "ProviderRequest",
    "RetrySettings",
    "ResourceDiagnostic",
    "ResourceOwner",
    "SkillInfo",
    "SkillSettings",
    "SessionCorruptError",
    "SessionInfo",
    "SessionLockError",
    "SessionStore",
    "StopReason",
    "SystemMessage",
    "TextBlock",
    "ThinkingBlock",
    "ThinkingLevel",
    "Tool",
    "ToolHooks",
    "ToolProvenance",
    "ToolCallBlock",
    "ToolDeclaration",
    "ToolResult",
    "ToolResultMessage",
    "ToolValidationError",
    "Usage",
    "UserMessage",
    "__version__",
    "create_agent_session",
    "create_harness",
    "create_coding_tools",
    "create_builtin_tools",
    "default_session_dir",
    "event_to_json",
]
