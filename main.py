import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
import asyncio
from fastmcp.client import Client
import base64
import io

import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

from utils_clip import frame_retrieval_seg_ego
from utils_general import get_from_cache, save_to_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("egoschema_subset_free.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def parse_json(text):
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        # print("No valid JSON found in the text.")
        return None


def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def get_llm_response(
    system_prompt, prompt, json_format=True, model="gpt-4-1106-preview"
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    logger.info(messages)
    cached_value = get_from_cache(key)
    if cached_value is not None:
        logger.info("Cache Hit")
        logger.info(cached_value)
        return cached_value

    # print("Not hit cache", key)
    # input() maybe to debug?

    for _ in range(3):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)
            save_to_cache(key, response)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"


def generate_final_answer(question, caption, num_frames, captioner):
    # answer_format = {"final_answer": "xxx"}
    caption_explanation = ""
    if captioner == "lavila":
        caption_explanation = """
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    People are denoted as alphabets. However, since it is only representation of caption, user cannot understand such notation.
    Thus, you MUST NOT contain any of these notations in your final response. Just represent them as 'man', 'woman' or 'person','people'.
    For example, in case of 'man X' or 'woman Y', express them as just 'man' or 'woman'.
    Also, express C as "camera wearer"."""

    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}{caption_explanation}
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer.
    """
    system_prompt = "You are a helpful assistant to understand video through frame captions"
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return response

def generate_abstract_answer(answer,query):
    prompt = f"""
    Given the following query: {query}, you have made an answer as follows: {answer}
    However, it might contain too long thinking step containing frame explanations. 
    Thus, I want you to make an abstract answer that answers given query, so that users can understand intuitively.
    Make sure that you do not contain your own opinion or abstraction, but only the information asked from given query.
    """
    system_prompt = "You are a helpful assistant designed to make reasonable inferences based on the provided information."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return response

def generate_description_step(question, caption, num_frames, segment_des, captioner):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    caption_explanation = ""
    if captioner == "lavila":
        caption_explanation = """
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    People are denoted as alphabets. However, since it is only representation of caption, user cannot understand such notation.
    Thus, you MUST NOT contain any of these notations in your final response. Just represent them as 'man', 'woman' or 'person', 'people'.
    For example, in case of 'man X' or 'woman Y', express them as just 'man' or 'woman'.
    Also, express C as "camera wearer"."""
    
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}{caption_explanation}
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def self_eval(previous_prompt, answer):
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def ask_gpt_caption(question, caption, num_frames, captioner):
    answer_format = {"final_answer": "xxx"}
    caption_explanation = ""
    if captioner == "lavila":
        caption_explanation = """
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer."""
    
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:
    {caption}{caption_explanation}
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and answer.
    """
    system_prompt = "You are a helpful assistant to understand video through frame captions"
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def ask_gpt_caption_step(question, caption, num_frames, captioner):
    caption_explanation = ""
    if captioner == "lavila":
        caption_explanation = """
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer."""
    
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}{caption_explanation}
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and answer.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption


async def run_one_question_free(video_id, query, caps, logs, captioner):
    question = query
    formatted_question = f"Here is the question: {question}\n"
    num_frames = len(caps)
    logger.info(f"Video ID: {video_id}, Number of Frames: {num_frames}")
    if os.path.exists(
        f"/home/intern/youngseo/a2a-tools/mcpserver/resources/embeddings/{video_id}.npy"
    ):
        logger.info(f"Video embedding for {video_id} already exists.")

    else:
        logger.info(
            f"Video embedding for {video_id} does not exist. Generating new embeddings."
        )
        async with Client("http://0.0.0.0:8081/sse") as client:
            tools = await client.list_tools()
            print("Trying to make video embedding. Available tools:", [t.name for t in tools])

            result = await client.call_tool(
                "clip_video",
                {
                    "video_path": "/home/intern/youngseo/a2a-tools/mcpserver/resources/vidoes/1.mp4",
                    "video_id": video_id,
                },
            ) 
            #print("Tool result:", result.data)  # path of the video embedding
            frame_embeddings = np.load(result.data["output_path"])
            print(f"frame_embeddings shape: {frame_embeddings.shape}")

    ## make video embedding before starting the main process
    
    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    logger.info(f"Processing video {video_id} with question: {question}")
    sampled_caps = read_caption(caps, sample_idx)
    # Take only the sampled captions to glance at whole video.
    # TODO : replace this into video captioning using ShareGPT4Video sliding captioning. Maybe slicing the video uniformly will be helpful too.
    previous_prompt, answer_str = ask_gpt_caption(
        formatted_question, sampled_caps, num_frames, captioner
    )
    confidence_str = self_eval(previous_prompt, answer_str)
    confidence = parse_text_find_confidence(confidence_str)
    
    ### Step 2 ###
    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
                captioner,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = await frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 2: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))

            sampled_caps = read_caption(caps, sample_idx)
            previous_prompt, answer_str = ask_gpt_caption_step(
                formatted_question, sampled_caps, num_frames, captioner
            )
            # answer = parse_text_find_number(answer_str)
            confidence_str = self_eval(previous_prompt, answer_str)
            confidence = parse_text_find_confidence(confidence_str)
        except Exception as e:
            logger.error(f"Step 2 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames, captioner
            )
            # answer = parse_text_find_number(answer_str)

    ### Step 3 ###
    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
                captioner,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = await frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 3: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))
            sampled_caps = read_caption(caps, sample_idx)
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames, captioner
            )
            # answer = parse_text_find_number(answer_str)
        except Exception as e:
            logger.error(f"Step 3 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames, captioner
            )

    count_frame = len(sample_idx)

    final = generate_abstract_answer(answer_str, question)

    logs[video_id] = {
        "answer": final,
        "count_frame": count_frame,
    }

    return {
        "answer": final,
        "count_frame": count_frame,
    }


async def run_one_question_free_wrapper(video_id, query, caps, logs, captioner):
    """비동기 함수를 위한 래퍼"""
    return await run_one_question_free(video_id, query, caps, logs, captioner)


def video_id_to_youtube_url(video_id):
    """Convert video ID to YouTube URL"""
    # Remove 'v_' prefix if present
    if video_id.startswith('v_'):
        video_id = video_id[2:]
    return f"https://www.youtube.com/watch?v={video_id}"

def videoAgentmain(video_id, query, captioner,yt = False):
    # this is made to run any free(natural languaged) format questions.
    # main changes :
    # delete option and answer from ann
    # edit some pre-defined prompt(formatted_question, prompt for ask_gpt_question and ask_gpt_question_step)
    # captioner can be "lavila","lavila_3rd", "moondream2"

    # if running full set, change subset to fullset
    all_cap_file = (
        "/home/intern/youngseo/a2a-tools/mcpserver/resources/captions/captions.json"
    )
    json_file_name = "egoschema_subset.json"
    
    all_caps = json.load(open(all_cap_file, "r"))
    # TODO : replace to custom captions using mcp tool.
    logs = {}

    async def video_caption():
        if video_id not in all_caps:
            all_caps[video_id] = {}
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Reload captions to check if it was added by previous attempt
                current_caps = json.load(open(all_cap_file, "r"))
                if video_id in current_caps and captioner in current_caps[video_id]:
                    logger.info(f"Caption for video {video_id} with captioner {captioner} found in file.")
                    all_caps[video_id][captioner] = current_caps[video_id][captioner]
                    return
                
                # If caption doesn't exist, try to generate it
                retry_count += 1
                print(f"Captioning attempt {retry_count}/{max_retries} for video {video_id} with captioner {captioner}")
                
                if not yt:
                    async with Client("http://0.0.0.0:8081/sse", timeout=None) as client:
                        result = await client.call_tool(
                            captioner,
                            {
                                "video_path": f"/home/intern/youngseo/a2a-tools/mcpserver/resources/vidoes/{video_id}.mp4",
                                "fps": 1,
                                "video_id": video_id,
                            },
                        )
                else:
                    async with Client("http://0.0.0.0:8081/sse", timeout=None) as client:
                        result = await client.call_tool(
                            captioner+"_yt",
                            {
                                "video_link": video_id_to_youtube_url(video_id),
                                "fps": 1,
                                "video_id": video_id,
                            },
                        )
                
                # After successful call, check if caption was actually saved
                updated_caps = json.load(open(all_cap_file, "r"))
                if video_id in updated_caps and captioner in updated_caps[video_id]:
                    logger.info(f"Caption successfully generated for video {video_id} with captioner {captioner}")
                    all_caps[video_id][captioner] = updated_caps[video_id][captioner]
                    return
                else:
                    logger.warning(f"Caption not found in file after attempt {retry_count} for video {video_id}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Caption not saved to file after {max_retries} attempts")
                        # Don't raise exception, just return to continue processing
                        return
                        
            except Exception as e:
                logger.error(f"Captioning attempt {retry_count} failed for video {video_id}: {e}")
                
                if retry_count >= max_retries:
                    logger.error(f"All {max_retries} captioning attempts failed for video {video_id}")
                    # Instead of raising exception, just return to allow processing to continue
                    return
                
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    # 비동기 함수를 순차적으로 실행
    async def run_all_questions():
        all_caps = json.load(open(all_cap_file, "r"))
        # print(all_caps)
        try:
            # Check if video caption exists before processing
            if video_id not in all_caps or captioner not in all_caps[video_id]:
                logger.warning(f"No caption available for video {video_id} with captioner {captioner}. Skipping processing.")
                return {"answer": None, "count_frame": 0}
                
            return await run_one_question_free_wrapper(
                video_id,
                query,
                all_caps[video_id][captioner],
                logs,
                captioner,
            )
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            return {"answer": None, "count_frame": 0}

    # 비동기 함수 실행
    try:
        asyncio.run(video_caption())
        return_answer = asyncio.run(run_all_questions())
    except Exception as e:
        logger.error(f"Failed to caption video {video_id} after all attempts: {e}")
        return_answer = {"answer": None, "count_frame": 0}

    if return_answer == None:
        return_answer = {"answer" : None, "count_frame" : 0}

    json.dump(logs, open(json_file_name, "w"))
    return return_answer

if __name__ == "__main__":
    #
    datas = json.load(open("/home/intern/youngseo/a2a-tools/mcpserver/resources/dataset/valid_datas.json", "r"))
    result_path = "/home/intern/youngseo/a2a/agents/videoAgent/VideoAgent/result/1k_qwen.json"
    results = json.load(open(result_path,"r"))
    invalid_videos = json.load(open("/home/intern/youngseo/a2a-tools/mcpserver/resources/dataset/invalid_clip.json", "r"))
    print(results.keys())

    for data in datas[:100]:
        #print(data["video_id"])
        if data["video_id"] in results.keys() and results[data["video_id"]]["answer"] != "GPT Error":
            # CLIP embedding for these videos are not able. try others
            continue
        elif data["video_id"] in invalid_videos:
            results[data["video_id"]] = {}
            results[data["video_id"]]["answer"] = "CLIP ERROR : This video cannot be encoded by CLIP video tool."
            results[data["video_id"]]["count_frame"] = None
            continue
        try:
            answer = videoAgentmain(
                video_id=data["video_id"],
                query=data["question"],
                captioner="qwen2_5",
                yt=True
            )
            results[data["video_id"]] = {}
            results[data["video_id"]]["answer"] = answer["answer"]
            results[data["video_id"]]["count_frame"] = answer["count_frame"]

            json.dump(results, open(result_path, "w"))
        except Exception as e:
            logger.error(f"Failed to process video {data['video_id']}: {e}")
            # Store failed result and continue
            results[data["video_id"]] = {}
            results[data["video_id"]]["answer"] = None
            results[data["video_id"]]["count_frame"] = 0
            json.dump(results, open(result_path, "w"))
            continue