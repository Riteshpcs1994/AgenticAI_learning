"""
============================================================
  Simple MCP (Model Context Protocol) Implementation
  Using OpenAI Function Calling (Tool Use Loop)
============================================================

  Tools included:
    - get_weather(city)
    - calculate(expression)
    - search_wiki(query)

  Install dependency:
    pip install openai

  Run:
    python mcp_openai.py
    or set OPENAI_API_KEY env var and run directly.
============================================================
"""

import os
import json
import math
import operator
import re
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# 1. TOOL DEFINITIONS  (sent to OpenAI as function schemas)
# ─────────────────────────────────────────────────────────────

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Tokyo'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '(17 * 4) + 99 / 3'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": "Search Wikipedia for a short summary of a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for, e.g. 'Python programming language'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────
# 2. TOOL IMPLEMENTATIONS  (mock / local — swap for real APIs)
# ─────────────────────────────────────────────────────────────

WEATHER_DB = {
    "london":   "12°C, overcast",
    "tokyo":    "22°C, sunny",
    "mumbai":   "34°C, humid and hazy",
    "new york": "8°C, cloudy",
    "paris":    "15°C, partly cloudy",
    "sydney":   "26°C, clear skies",
    "berlin":   "9°C, light rain",
    "dubai":    "38°C, sunny and dry",
}

WIKI_DB = {
    "mcp":        "Model Context Protocol (MCP) is an open standard by Anthropic for connecting AI assistants to external tools and data sources in a structured way.",
    "openai":     "OpenAI is an AI research company founded in 2015, known for GPT-4, DALL·E, and the ChatGPT product.",
    "python":     "Python is a high-level, general-purpose programming language created by Guido van Rossum, emphasizing code readability and simplicity.",
    "react":      "React is a JavaScript library for building user interfaces, originally developed by Meta (Facebook).",
    "machine":    "Machine learning is a subset of AI where systems learn from data to improve their performance without being explicitly programmed.",
    "deep":       "Deep learning is a subset of machine learning using neural networks with many layers to model complex patterns in data.",
    "llm":        "A Large Language Model (LLM) is a type of AI model trained on vast text data to understand and generate human language.",
    "langchain":  "LangChain is an open-source framework for building applications powered by large language models with tool use and memory.",
}


def get_weather(city: str) -> str:
    """Return mock weather data for a city."""
    key = city.strip().lower()
    result = WEATHER_DB.get(key, f"{hash(city) % 35 + 5}°C, variable conditions")
    return f"Weather in {city.title()}: {result}"


def calculate(expression: str) -> str:
    """
    Safely evaluate a math expression.
    Supports: +, -, *, /, **, //, %, parentheses, and common math functions.
    """
    # Allowed names for safe eval
    safe_globals = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": math.pow, "log": math.log,
        "log2": math.log2, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
        "floor": math.floor, "ceil": math.ceil,
    }
    # Basic sanity check — allow only safe characters
    if re.search(r"[a-zA-Z_]", expression.replace("sqrt","").replace("pow","")
                  .replace("log","").replace("sin","").replace("cos","")
                  .replace("tan","").replace("pi","").replace("abs","")
                  .replace("round","").replace("min","").replace("max","")
                  .replace("floor","").replace("ceil","").replace("log2","")
                  .replace("log10","").replace("e","").replace("inf","")):
        return f"Error: unsafe expression '{expression}'"
    try:
        result = eval(expression, safe_globals)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


def search_wiki(query: str) -> str:
    """Return a mock Wikipedia summary for a topic."""
    key = query.strip().lower().split()[0]
    summary = WIKI_DB.get(key)
    if summary:
        return f"Wikipedia — {query.title()}: {summary}"
    return (
        f"Wikipedia — {query.title()}: "
        f"A widely studied topic with applications across many fields. "
        f"For full details, visit https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    )


# ─────────────────────────────────────────────────────────────
# 3. TOOL DISPATCHER
# ─────────────────────────────────────────────────────────────

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate":   calculate,
    "search_wiki": search_wiki,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Look up and call the requested tool by name."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as exc:
        return f"Error calling '{name}': {exc}"


# ─────────────────────────────────────────────────────────────
# 4. CORE MCP AGENTIC LOOP
# ─────────────────────────────────────────────────────────────

def run_mcp_loop(
    user_message: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    max_turns: int = 6,
    verbose: bool = True,
) -> str:
    """
    Run the MCP tool-use loop:
      1. Send message + tool schemas to the model
      2. If model calls tools → execute them → append results
      3. Repeat until the model returns a plain text answer
      4. Return the final assistant response
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use the tools whenever they would help answer the user's question accurately."
            )
        },
        {"role": "user", "content": user_message}
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  USER: {user_message}")
        print(f"{'='*60}")

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n[Turn {turn}] Calling {model}…")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=MCP_TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # ── No tool calls → final answer ──────────────────────
        if not msg.tool_calls:
            final = msg.content or ""
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  ASSISTANT: {final}")
                print(f"{'─'*60}\n")
            return final

        # ── Tool calls → execute each one ─────────────────────
        # Append the assistant message (with tool_calls) to history
        messages.append(msg)

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if verbose:
                print(f"  🔧 TOOL CALL  → {fn_name}({fn_args})")

            result = execute_tool(fn_name, fn_args)

            if verbose:
                print(f"  ✅ TOOL RESULT → {result}")

            # Append tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Max turns exceeded
    warning = f"[Max turns ({max_turns}) reached without a final answer.]"
    if verbose:
        print(warning)
    return warning


# ─────────────────────────────────────────────────────────────
# 5. INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "What's the weather in Tokyo and London?",
    "Calculate (17 * 4) + 99 / 3",
    "Search Wikipedia for MCP and then calculate 2 ** 10",
    "What's 144 / 12, and also the weather in Paris?",
    "Search for Python and tell me the weather in Mumbai",
]


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          MCP + OpenAI  —  Tool Use Loop (Python)         ║
║                                                          ║
║  Tools: get_weather · calculate · search_wiki            ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_demo(client: OpenAI):
    """Run a set of demo queries automatically."""
    print("\n── Running demo queries ──\n")
    for q in DEMO_QUERIES:
        run_mcp_loop(q, client, verbose=True)


def run_interactive(client: OpenAI):
    """Interactive CLI loop."""
    print("\nType your question and press Enter. Type 'exit' to quit.")
    print("Type 'demo' to run preset demo queries.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "demo":
            run_demo(client)
            continue

        run_mcp_loop(user_input, client, verbose=True)


# ─────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner()
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = input("Enter your OpenAI API key (sk-…): ").strip()

    if not api_key.startswith("sk-"):
        print("⚠️  Invalid or missing API key. Exiting.")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)

    print("\nOptions:")
    print("  [1] Run demo queries")
    print("  [2] Interactive mode")
    choice = input("\nChoose (1/2): ").strip()

    if choice == "1":
        run_demo(client)
    else:
        run_interactive(client)
