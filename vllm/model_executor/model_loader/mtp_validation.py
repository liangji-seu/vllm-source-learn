# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped controls for MTP checkpoint completeness validation."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_mtp_completeness_check_enabled: ContextVar[bool] = ContextVar(
    "mtp_completeness_check_enabled", default=True
)


def is_mtp_completeness_check_enabled() -> bool:
    """Return whether MTP completeness validation is enabled in this scope."""
    # ------【投机解码】读取当前作用域的 ContextVar 判断 MTP 完整性校验是否开启 ------
    return _mtp_completeness_check_enabled.get()


@contextmanager
def disable_mtp_completeness_check() -> Iterator[None]:
    """Temporarily disable MTP completeness validation for one weight load."""
    # ------【投机解码】把 ContextVar 置为 False，临时关掉本次权重加载的完整性校验 ------
    token = _mtp_completeness_check_enabled.set(False)
    try:
        yield
    finally:
        # ------【投机解码】退出作用域时恢复原值，保证校验状态不泄漏到其他加载 ------
        _mtp_completeness_check_enabled.reset(token)
