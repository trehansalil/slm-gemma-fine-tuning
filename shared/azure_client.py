"""Shared Azure OpenAI async client setup.

Hoisted from gemma2b/create_preference_data.py so later stages
(create_instruction_data, train_raft) reuse a single client factory.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()


def make_client() -> tuple[AsyncAzureOpenAI, str]:
    """Build the async Azure client and resolve the deployment name."""
    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        api_version=os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=os.environ["AZURE_BASE_URL"],
    )
    model = os.environ.get("AZURE_MODEL", "gpt-5.4-mini")
    if model.startswith("azure/"):
        model = model[len("azure/"):]
    return client, model
