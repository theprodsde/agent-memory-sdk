"""Framework adapters for agent-memory-sdk.

Provides drop-in memory backends for LangChain and LlamaIndex.
Each adapter is a thin wrapper that satisfies the framework's memory
interface while delegating storage and retrieval to the Memory SDK.
"""
