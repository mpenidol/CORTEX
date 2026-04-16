from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from schemas import SceneGraph, AssetMappingReport

# --- Model Configuration ---

ollama_model = ChatOllama(
    model="gpt-oss:20b",
    base_url="http://localhost:11434",
)

use_model = ollama_model

# --------------------------------

# --- Semantic Agent ---

SEMANTIC_SYSTEM_PROMPT = """
You are an Art Director and 3D Scene Architect.
The user will provide an input describing a desired environment.

This input can range from:
  - Direct Input: "A lab with 2 desks, 1 chair, and a server rack."
  - Indirect Input: "I want to be in a big forest" or "A place to study."

Your goal is to create a rich, logical, and immersive environment based on this idea.

CREATION RULES:
  1. Create between 5 to 10 objects that make sense for the theme. Do not wait for the user to ask for them. Provide a clear 'named_asset' for each.
  2. Establish a realistic physical hierarchy using 'parent_id':
     - Large furniture (beds, tables, cabinets) have parent_id='world'.
     - Smaller objects (monitors, cups, books) must have the parent_id of the furniture they rest on (e.g., parent_id='desk_1').
  3. You must output strictly valid json only.
  4. Do not include any conversational text. Do not include markdown tags like ```json.
"""

semantic_prompt = ChatPromptTemplate.from_messages([
    ("system", SEMANTIC_SYSTEM_PROMPT),
    ("human", "{input}"),
])

semantic_chain = semantic_prompt | use_model.with_structured_output(SceneGraph, include_raw=True)

# --------------------------------

# --- Asset Agent ---

ASSET_SYSTEM_PROMPT = """
You are a 3D asset matcher. You receive a list of requested objects, each with pre-filtered database candidates.

YOUR TASK:
- For EVERY item in the list, produce exactly one mapping entry.
- Use the EXACT instance_id shown after "ID:" as the instance_id field.
- Use the EXACT named_asset shown after "Requested:" as the requested_asset field.
- Choose the best matching db_id from the candidates list (marked with *). Use the exact db_id string shown in brackets [...].
- If NO candidate fits functionally, set matched_db_id to null.

You MUST produce one entry per item. Do not skip any item.
"""

asset_prompt = ChatPromptTemplate.from_messages([
    ("system", ASSET_SYSTEM_PROMPT),
    ("human", "{input}"),
])

asset_chain = asset_prompt | use_model.with_structured_output(AssetMappingReport, include_raw=True)

# --------------------------------


def init_agents(provider: str = "ollama", model: str = "gpt-oss:20b", port: int | None = None):
    """Rebuild semantic_chain and asset_chain for the given provider/model/port."""
    global semantic_chain, asset_chain

    if provider == "vllm":
        from langchain_openai import ChatOpenAI
        _port = port or 8000
        llm = ChatOpenAI(
            model=model,
            base_url=f"http://localhost:{_port}/v1",
            api_key="EMPTY",
        )
    else:  # ollama (default)
        _port = port or 11434
        llm = ChatOllama(model=model, base_url=f"http://localhost:{_port}")

    semantic_chain = semantic_prompt | llm.with_structured_output(SceneGraph, include_raw=True)
    asset_chain    = asset_prompt    | llm.with_structured_output(AssetMappingReport, include_raw=True)
    print(f"[Agents] Initialized — provider={provider}  model={model}  port={_port}")
