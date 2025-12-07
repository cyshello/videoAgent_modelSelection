from fastmcp import FastMCP
import subprocess
import json

mcp = FastMCP("Test MCP Server")

@mcp.tool()
def clip_video():
    pass

@mcp.tool()
def clip_text():
    pass

@mcp.tool()
def internvl2(video_path: str, question: str, frame_number: int):
    # code from https://huggingface.co/OpenGVLab/InternVL2-8B
    pass

@mcp.tool()
def llavaov(video_path: str, question: str, frame_number: int):
    # code from 
    pass

@mcp.tool()
def qwen2vl(video_path: str, question: str, frame_number: int):
    # from 
    tool_python = "/home/intern/youngseo/modelSelection/videoAgent_modelSelection/vlms/qwen2/qwen2/bin/python"
    tool_script = "/home/intern/youngseo/modelSelection/videoAgent_modelSelection/vlms/qwen2/inference.py"

    
    pass

if __name__ == "__main__":
    # mcp.run(transport="sse", port=8080, host="0.0.0.0")
    mcp.run(transport="sse", port=8081, host="0.0.0.0")
