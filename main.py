from fastapi import FastAPI
from database import ASSET_DATABASE
from schemas import UnityMessage
from agents import semantic_chain, asset_chain
from retrieval import retrieve_all_objects
from layout import calculate_unity_layout

import json

app = FastAPI()


def _format_candidates_prompt(scene_objects, candidates_per_object: dict) -> str:
    lines = []
    for obj in scene_objects:
        lines.append(f"- ID: {obj.instance_id} | Requested: {obj.named_asset}")
        candidates = candidates_per_object.get(obj.instance_id, [])
        if candidates:
            for c in candidates:
                lines.append(f"    * [{c['db_id']}] {c['name']}: {c['description']}")
        else:
            lines.append("    * No candidates found")
    return "Requested Items with top candidates:\n" + "\n".join(lines)


@app.post("/generate_scene")
def generate_scene(message: UnityMessage):
    """
    Main API endpoint that Unity calls. It receives a text prompt,
    runs it through the AI agents, calculates the math, and returns the 3D layout.
    """

    print(f"\n Unity Requested: {message.content}")

    print("1. Semantic Agent is running...")
    semantic_result = semantic_chain.invoke({"input": message.content})
    print(f"   → SceneGraph objects: {[o.instance_id + '=' + o.named_asset for o in semantic_result.objects]}")

    print("2. Retrieving candidates from vectorstore...")
    candidates_per_object = retrieve_all_objects(semantic_result.objects, k=50, top_n=20)
    for iid, cands in candidates_per_object.items():
        print(f"   → {iid}: {[c['db_id'] for c in cands]}")

    print("3. Asset Agent is running...")
    asset_prompt = _format_candidates_prompt(semantic_result.objects, candidates_per_object)
    asset_result = asset_chain.invoke({"input": asset_prompt})
    print(f"   → AssetMappingReport mappings: {[(m.instance_id, m.matched_db_id) for m in asset_result.mappings]}")

    print("4. Calculating Mathematical Layout...")
    final_result = calculate_unity_layout(
        semantic_objects=semantic_result.objects,
        mapped_assets=asset_result.mappings,
        database=ASSET_DATABASE
    )

    print("Final result generated and sent to Unity!")
    return final_result


if __name__ == "__main__":
    print("Test Without Unity")

    test_message = UnityMessage(content="Create a Study Room")

    test_result = generate_scene(test_message)

    print("Final Result")
    print(json.dumps(test_result, indent=2))
