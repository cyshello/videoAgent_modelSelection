import pickle
from typing import List
from fastmcp.client import Client
import base64
import io
import os

import numpy as np

cache_clip = pickle.load(open("cache_clip.pkl", "rb"))

# def get_embeddings(inputs: List[str]) -> np.ndarray:
#     embeddings = [cache_clip[input] for input in inputs]
#     return np.array(embeddings)


async def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):

    frame_embeddings = np.load(
        f"/home/intern/youngseo/a2a-tools/mcpserver/resources/embeddings/{video_id}.npy"
    )
    print(f"frame_embeddings shape: {frame_embeddings.shape}")
    # TODO : Fix path
    # TODO : replace frame_embedding to custom video embedding, 512 channel로 맞춰주기. 같은 모델을 사용한다면 가능할 것. 지금 8B 모델이 없고 import clip으로 하고 있는거라서 차원이 안 맞음.

    text_embedding = []

    async with Client("http://0.0.0.0:8081/sse") as client:
        tools = await client.list_tools()
        print("Trying to make text embedding.. Available tools:", [t.name for t in tools])

        # 각각의 텍스트별로 호출하여 결과 수집
        for description in descriptions:
            result = await client.call_tool(
                "clip_text",
                {"text": description["description"]},
            )
            # print("Tool result:", result.data)

            # base64 문자열 추출 (result.data 구조에 따라 조정)
            b64_string = result.data["data_base64"]
            byte_data = base64.b64decode(b64_string)
            buffer = io.BytesIO(byte_data)
            embedding = np.load(buffer)
            print(f"Embedding shape: {embedding.shape}")

            text_embedding.append(embedding)

    # text_embedding = get_embeddings(
    #     [description["description"] for description in descriptions]
    # )
    # TODO : replace text_embedding to custom video embedding ; to do so, use text embedding of clip
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]

        if seg_frame_embeddings.shape[0] < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        #print(type(seg_similarity.argmax()))
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)

    print(f"frame_idx: {frame_idx}")
    return frame_idx


if __name__ == "__main__":
    pass
