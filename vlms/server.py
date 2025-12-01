from fastmcp import FastMCP
import subprocess
import json

mcp = FastMCP("Test MCP Server")

@mcp.tool()
def internvl2():
    # code from https://huggingface.co/OpenGVLab/InternVL2-8B
    pass

@mcp.tool()
def llavaov():
    # code from 
    pass

@mcp.tool()
def qwen2vl():
    # from 
    pass

if __name__ == "__main__":
    # mcp.run(transport="sse", port=8080, host="0.0.0.0")
    mcp.run(transport="sse", port=8081, host="0.0.0.0")
