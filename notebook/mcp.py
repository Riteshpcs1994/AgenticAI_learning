import asyncio
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import datetime
import time

server = Server("multi-tool-server")

# List the tools available
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_current_time",
            description="Returns current system time",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["iso", "unix"]}
                }
            }
        ),
        types.Tool(
            name="get_weather",
            description="Simulated weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )
    ]

# Handle tool calls
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name == "get_current_time":
        fmt = arguments.get("format", "iso") if arguments else "iso"
        if fmt == "iso":
            result = datetime.datetime.now().isoformat()
        else:
            result = str(int(time.time()))
        return [types.TextContent(type="text", text=result)]

    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        # Simulate a weather response
        weather = f"Weather in {city}: Sunny, 22°C"
        return [types.TextContent(type="text", text=weather)]

    else:
        raise ValueError(f"Unknown tool: {name}")

# Run the server over stdio
async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="multi-tool-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())