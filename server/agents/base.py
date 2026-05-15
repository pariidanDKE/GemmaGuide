from __future__ import annotations

import logging
import time
from typing import Any

from pipeline.session import Session
from . import runtime


logger = logging.getLogger(__name__)


class BaseAgent:
    name = "agent"
    system_prompt = ""
    llm_timing_stage = "agent.llm"
    total_timing_stage = "agent.total"

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_client(self) -> Any:
        return runtime.create_agent_client()

    def build_system_message(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt}


class SingleCallAgent(BaseAgent):
    def run_single_call(
        self,
        *,
        messages: list[dict],
        request_overrides: dict[str, Any] | None = None,
    ) -> Any:
        t_total = time.monotonic()
        client = self.create_client()
        request_payload = runtime.make_request_payload(messages, **(request_overrides or {}))
        response, llm_latency = runtime.execute_model_call(
            client=client,
            session=self.session,
            request_payload=request_payload,
            round_idx=0,
            timing_stage=self.llm_timing_stage,
        )
        total_latency = time.monotonic() - t_total
        self.session.add_timing(self.total_timing_stage, total_latency)
        logger.info(
            "%s llm_latency=%.3fs total_latency=%.3fs",
            self.name,
            llm_latency,
            total_latency,
        )
        return response


class ToolLoopAgent(BaseAgent):
    max_rounds = runtime.MAX_TOOL_ROUNDS

    def build_request_overrides(self, round_idx: int) -> dict[str, Any]:
        raise NotImplementedError

    def handle_no_tool_calls(self, messages: list[dict], choice: Any, round_idx: int) -> tuple[bool, bool]:
        raise NotImplementedError

    def run_tool_loop(self, messages: list[dict]) -> int:
        t_total = time.monotonic()
        client = self.create_client()

        for round_idx in range(self.max_rounds):
            request_payload = runtime.make_request_payload(messages, **self.build_request_overrides(round_idx))
            response, llm_latency = runtime.execute_model_call(
                client=client,
                session=self.session,
                request_payload=request_payload,
                round_idx=round_idx,
                timing_stage=self.llm_timing_stage,
                timing_meta={"round": round_idx},
            )
            logger.info("%s round=%s llm_latency=%.3fs", self.name, round_idx, llm_latency)

            choice = response.choices[0]
            if choice.message.tool_calls:
                messages.append(runtime.assistant_tool_message(choice))
                messages.extend(runtime.dispatch_tool_calls(choice.message.tool_calls, self.session))
                continue

            should_continue, is_done = self.handle_no_tool_calls(messages, choice, round_idx)
            if should_continue:
                continue
            if is_done:
                total_latency = time.monotonic() - t_total
                self.session.add_timing(
                    self.total_timing_stage,
                    total_latency,
                    rounds=round_idx + 1,
                    measurements=len(self.session.measurements),
                )
                logger.info(
                    "%s total_latency=%.3fs rounds=%s measurements=%s",
                    self.name,
                    total_latency,
                    round_idx + 1,
                    len(self.session.measurements),
                )
                return round_idx + 1

        total_latency = time.monotonic() - t_total
        self.session.add_timing(
            self.total_timing_stage,
            total_latency,
            rounds=self.max_rounds,
            measurements=len(self.session.measurements),
        )
        logger.info(
            "%s total_latency=%.3fs rounds=%s measurements=%s",
            self.name,
            total_latency,
            self.max_rounds,
            len(self.session.measurements),
        )
        return self.max_rounds
