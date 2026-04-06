from fastapi import FastAPI
from database import ASSET_DATABASE
from schemas import UnityMessage
from agents import semantic_chain, asset_chain
from layout import calculate_unity_layout

import json

app = FastAPI()

@app.post("/generate_scene")
def generate_scene(message: UnityMessage):
    """
    Main API endpoint that Unity calls. It receives a text prompt, 
    runs it through the AI agents, calculates the math, and returns the 3D layout.
    """

    print(f"\n Unity Requested: {message.content}")
    
    print("1. Semantic Agent is running...")
    semantic_result = semantic_chain.invoke({"input": message.content})

    print("2. Asset Agent is running...")

    requested_items_str = "\n".join(
        [f"- ID: {obj.instance_id} | Requested: {obj.named_asset}" for obj in semantic_result.objects]
    )

    # Create the prompt combining the Agent Request and the database CSV
    asset_prompt = f"Requested Items:\n{requested_items_str}\n\nLocal Database:\n{ASSET_DATABASE}"
    asset_result = asset_chain.invoke({"input": asset_prompt})

    print("3. Calculating Mathematical Layout...")
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