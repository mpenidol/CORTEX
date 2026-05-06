"""Architecture: Full DB hardcodeada en el system prompt del asset agent.

Pipeline: toda la DB → system prompt del asset agent → el LLM elige directamente.
Sin vectorstore, sin reranker. El asset agent recibe el listado completo de assets
en su system prompt y solo el listado de objetos pedidos en el human prompt.

  K = NO APLICA (registrado como 0). No hay etapa de recuperacion.
  N = NO APLICA (registrado como 0). El LLM tiene acceso a todos los assets
      directamente; no hay filtrado previo.

La unica variable relevante es db_size: determina cuantos assets entran en el
system prompt y por tanto cuantos tokens de entrada consume el asset agent.
Parámetros sweep: solo db_size.
"""

from database import ASSET_DATABASE

ARCH_NAME = "full_db"

_db_subset: list[dict] = []


def init_retrieval(db_size: int | None = None):
    global _db_subset
    _db_subset = ASSET_DATABASE[:db_size] if db_size else ASSET_DATABASE
    print(f"[full_db] DB cargada: {len(_db_subset)} assets en system prompt")


def retrieve_all_objects(scene_objects, k: int = 0, top_n: int = 0) -> dict[str, list[dict]]:
    """No hay recuperación — la DB entera está en el system prompt del asset agent."""
    return {obj.instance_id: [] for obj in scene_objects}


def format_asset_prompt(scene_objects, candidates_per_object: dict) -> str:
    """Human prompt minimalista: solo la lista de objetos a mapear.
    Los candidatos van en el system prompt (toda la DB), no aquí.
    """
    lines = ["Match EVERY item below. One output entry per item.\n"]
    for i, obj in enumerate(scene_objects, 1):
        lines.append(f"ITEM {i}: ID={obj.instance_id} | Requested={obj.named_asset}")
        lines.append("")
    return "\n".join(lines)


def get_asset_chain():
    """Construye un asset_chain con toda la DB embebida en el system prompt."""
    import agents
    from langchain_core.prompts import ChatPromptTemplate
    from schemas import AssetMappingReport

    db_lines = "\n".join(
        f"[{a['db_id']}] {a['name']}: {a['description']} | size={a['size']}"
        for a in _db_subset
    )

    system_prompt = f"""You are a 3D asset matcher.

AVAILABLE ASSET DATABASE ({len(_db_subset)} assets):
{db_lines}

YOUR TASK:
- For EVERY item listed below, produce exactly one mapping entry.
- Use the EXACT instance_id shown after "ID:" as the instance_id field.
- Use the EXACT named_asset shown after "Requested:" as the requested_asset field.
- Choose the best matching db_id from the database above. Use the exact db_id string shown in [...].
- If NO asset fits functionally, set matched_db_id to null.

You MUST produce one entry per item. Do not skip any item."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt | agents._llm.with_structured_output(AssetMappingReport, include_raw=True)
