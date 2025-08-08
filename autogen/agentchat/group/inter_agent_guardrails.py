# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from pydantic import BaseModel, Field

from .guardrails import GuardrailResult, LLMGuardrail, RegexGuardrail
from .targets.transition_target import StayTarget

if TYPE_CHECKING:
    from ..groupchat import GroupChat
    from ..conversable_agent import ConversableAgent
    from ...llm_config import LLMConfig


class MaskingAgent:
    """Specialized agent for intelligent content masking using LLM.
    
    This is a lightweight wrapper around ConversableAgent specifically designed
    for masking sensitive content while preserving message structure.
    """
    
    def __init__(self, llm_config: "LLMConfig"):
        """Initialize the masking agent with LLM configuration.
        
        Args:
            llm_config: LLM configuration for the masking agent
        """
        # Import here to avoid circular imports
        from ..conversable_agent import ConversableAgent
        
        self._agent = ConversableAgent(
            name="masking_agent",
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message=(
                "You are a content masking assistant. Your task is to replace sensitive information "
                "with [SENSITIVE_INFO] while keeping the rest of the message intact. "
                "Return ONLY the masked message, nothing else. Do not include explanations or formatting."
            ),
            max_consecutive_auto_reply=1,
        )
    
    def mask_content(self, original_text: str, justification: str) -> str:
        """Intelligently mask sensitive content in the given text.
        
        Args:
            original_text: The original message content
            justification: Description of what sensitive content was detected
            
        Returns:
            The masked content with sensitive information replaced by [SENSITIVE_INFO]
        """
        masking_prompt = (
            f"Original message: {original_text}\n\n"
            f"Sensitive content detected: {justification}\n\n"
            f"Please return the message with only the sensitive information replaced by [SENSITIVE_INFO]. "
            f"Keep everything else exactly the same."
        )
        
        try:
            # Generate reply using the agent's LLM
            response = self._agent.generate_oai_reply(
                messages=[{"role": "user", "content": masking_prompt}]
            )
            
            # Extract the reply content
            if response[0] and response[1]:  # (success, reply)
                if isinstance(response[1], dict):
                    return response[1].get("content", original_text)
                else:
                    return str(response[1])
            else:
                print(f"[WARNING] MaskingAgent failed to generate response")
                return original_text
                
        except Exception as e:
            print(f"[WARNING] MaskingAgent error: {e}")
            return original_text


class InterAgentRule(BaseModel):
    """Configuration for an inter-agent safety rule.

    Schema inspired by MARIS manifest format.
    """

    message_src: str = Field(..., description="Name of source agent or '*' for any")
    message_dst: str = Field(..., description="Name of destination agent or '*' for any")
    disallow_item: Optional[List[str]] = Field(
        default=None, description="List of categories to disallow (used by LLM guardrail)"
    )
    check_method: str = Field(
        default="llm", description="'llm' (uses LLMGuardrail) or 'regex' (uses RegexGuardrail)"
    )
    pattern: Optional[str] = Field(
        default=None, description="Regex pattern to detect when check_method == 'regex'"
    )
    action: str = Field(
        default="block",
        description="Action when triggered: 'mask' (redact matches) or 'block' (replace message with activation text)",
    )
    activation_message: Optional[str] = Field(
        default=None,
        description="Custom activation message appended when rule triggers",
    )


class InterAgentGuardrail(BaseModel):
    """Runtime object representing a compiled inter-agent guardrail."""

    applies_from: str
    applies_to: str
    check_method: str
    activation_message: str
    _llm_guardrail: Optional[LLMGuardrail] = None
    _regex_guardrail: Optional[RegexGuardrail] = None
    _masking_agent: Optional[MaskingAgent] = None
    _action: str = "block"
    _pattern: Optional[str] = None

    def check_and_act(
        self,
        *,
        groupchat: "GroupChat",
        src_agent_name: str,
        dst_agent_name: str,
        message_content: Union[str, list[dict[str, Any]]],
    ) -> Optional[str]:
        """Run the guardrail when src -> dst message is being broadcast.

        Returns optional string to send instead of original content (e.g., activation notice).
        If None, caller may proceed with sending original content.
        """
        # Fast path: filter by applies_from/applies_to (wildcards handled by caller expansion)
        if src_agent_name != self.applies_from or dst_agent_name != self.applies_to:
            return None

        # Prepare context for check
        if isinstance(message_content, list):
            context = message_content
        elif isinstance(message_content, dict):
            content_text = str(message_content.get("content", ""))
            context = [
                {
                    "role": "system",
                    "content": f"Inter-agent check. From: {src_agent_name}. To: {dst_agent_name}.",
                },
                {"role": "user", "content": content_text},
            ]
        else:
            # Include minimal directional context so LLM checks can reason about transition
            context = [
                {
                    "role": "system",
                    "content": f"Inter-agent check. From: {src_agent_name}. To: {dst_agent_name}.",
                },
                {"role": "user", "content": str(message_content)},
            ]

        # Execute underlying guardrail
        if self.check_method == "llm":
            assert self._llm_guardrail is not None
            result: GuardrailResult = self._llm_guardrail.check(context=context)
        else:
            assert self._regex_guardrail is not None
            # RegexGuardrail expects str or list of messages; pass the raw content if possible
            text_ctx: Union[str, list[dict[str, Any]]]
            if isinstance(message_content, str):
                text_ctx = message_content
            elif isinstance(message_content, list):
                # Concatenate contents of chat messages
                text_ctx = "\n".join([m.get("content", "") for m in message_content])
            elif isinstance(message_content, dict):
                text_ctx = str(message_content.get("content", ""))
            else:
                text_ctx = str(message_content)
            result = self._regex_guardrail.check(context=text_ctx)

        if result.activated:
            justification = result.justification or "Triggered"
            # Handle actions
            if self._action == "block":
                return f"{self.activation_message}\nJustification: {justification}"
            if self._action == "mask":
                # Extract string to mask
                if isinstance(message_content, str):
                    text = message_content
                elif isinstance(message_content, dict):
                    text = str(message_content.get("content", ""))
                else:  # list of messages
                    text = "\n".join([m.get("content", "") for m in message_content])

                if self.check_method == "regex" and self._pattern:
                    # Regex-based masking: substitute pattern matches with [SENSITIVE_INFO]
                    import re
                    try:
                        masked = re.sub(self._pattern, "[SENSITIVE_INFO]", text)
                        return masked
                    except re.error:
                        # On regex error, block instead
                        return f"{self.activation_message}\nJustification: {justification}"
                        
                elif self.check_method == "llm" and self._masking_agent:
                    # LLM-based masking: use the MaskingAgent for intelligent content masking
                    try:
                        print(f"[DEBUG] Attempting MaskingAgent masking for: {text[:50]}...")
                        
                        masked_content = self._masking_agent.mask_content(text, justification)
                        
                        print(f"[DEBUG] MaskingAgent result: {masked_content[:100]}...")
                        return masked_content
                        
                    except Exception as e:
                        # Fallback to block if masking fails
                        print(f"[WARNING] MaskingAgent failed: {e}")
                        return f"{self.activation_message}\nJustification: {justification}"
                else:
                    # No masking method available; block
                    return f"{self.activation_message}\nJustification: {justification}"

        return None


def _expand_wildcards(rules: List[InterAgentRule], agent_names: List[str]) -> List[Tuple[str, str, InterAgentRule]]:
    expanded: List[Tuple[str, str, InterAgentRule]] = []
    for rule in rules:
        src_list = agent_names if rule.message_src == "*" else [rule.message_src]
        dst_list = agent_names if rule.message_dst == "*" else [rule.message_dst]
        for s in src_list:
            for d in dst_list:
                if s == d:
                    continue
                expanded.append((s, d, rule))
    return expanded


def apply_inter_agent_guardrails(
    *,
    groupchat: "GroupChat",
    manifest: Union[Dict[str, Any], str],
    llm_config: Optional["LLMConfig"] = None,
) -> None:
    """Attach inter-agent guardrails to a GroupChat instance from a manifest.

    The manifest should contain a top-level key 'inter_agent_safeguards' with
    an 'agent_transitions' list. Each entry follows InterAgentRule.
    
    Args:
        groupchat: The GroupChat instance to attach guardrails to
        manifest: Manifest dict or path to JSON file containing guardrail rules
        llm_config: LLM configuration used for both detection and masking operations
    """
    # Load manifest if path provided
    config: Dict[str, Any]
    if isinstance(manifest, str):
        import json

        with open(manifest, "r") as f:
            config = json.load(f)
    else:
        config = manifest

    inter = config.get("inter_agent_safeguards", {})
    transitions_cfg = inter.get("agent_transitions", [])

    rules = [InterAgentRule(**item) for item in transitions_cfg]

    # Build guardrails
    guardrails: List[InterAgentGuardrail] = []
    agent_names = [a.name for a in groupchat.agents]
    expanded = _expand_wildcards(rules, agent_names)
    
    # Create shared MaskingAgent for LLM-based masking if needed
    masking_agent: Optional[MaskingAgent] = None
    has_llm_masking = any(rule.check_method == "llm" and rule.action == "mask" for _, _, rule in expanded)
    if has_llm_masking and llm_config is not None:
        masking_agent = MaskingAgent(llm_config)
        print(f"[DEBUG] Created shared MaskingAgent for LLM-based content masking")

    for src, dst, rule in expanded:

        activation_message = (
            rule.activation_message
            or f"Guardrail activated for transition {src} -> {dst}. Action: {rule.action}"
        )

        inter_gr = InterAgentGuardrail(
            applies_from=src,
            applies_to=dst,
            check_method=rule.check_method,
            activation_message=activation_message,
        )
        inter_gr._action = (rule.action or "block").lower()

        # Underlying checker
        if rule.check_method == "llm":
            if llm_config is None:
                raise ValueError("llm_config is required for LLM-based inter-agent guardrails")

            categories = ", ".join(rule.disallow_item or [])
            condition = (
                "You check if the following message (from source agent to destination agent) contains any of the "
                f"disallowed categories: [{categories}]. Return a JSON with fields 'activated' and 'justification'."
            )

            # Create LLM guardrail for detection
            inter_gr._llm_guardrail = LLMGuardrail(
                name=f"inter_{src}_to_{dst}",
                condition=condition,
                target=StayTarget(),
                llm_config=llm_config,
                activation_message=activation_message,
            )
            
            # Assign shared MaskingAgent if masking action is needed
            if rule.action == "mask" and masking_agent is not None:
                inter_gr._masking_agent = masking_agent
                
        else:
            pattern = rule.pattern or ""
            if not pattern:
                # If regex chosen without pattern, skip
                continue
            inter_gr._regex_guardrail = RegexGuardrail(
                name=f"inter_{src}_to_{dst}",
                condition=pattern,
                target=StayTarget(),
                activation_message=activation_message,
            )
            inter_gr._pattern = pattern

        guardrails.append(inter_gr)

    # Attach to groupchat dynamically
    setattr(groupchat, "_inter_agent_guardrails", guardrails)


