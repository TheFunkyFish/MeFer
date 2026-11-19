import re
import json
from typing import List, Dict, Any, Optional


class ToolCallExtractor:
    """
    A class to extract tool calls from LLM responses.
    Supports extracting <tool_call> blocks and parsing them into structured data.
    """

    # Regex pattern to match <tool_call> blocks
    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL
    )

    def __init__(self):
        self.tool_calls: List[Dict[str, Any]] = []

    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract all <tool_call> blocks from an LLM response.

        Args:
            response: The raw LLM response string.

        Returns:
            A list of dictionaries, each representing a parsed tool call.
        """
        self.tool_calls = []
        matches = self.TOOL_CALL_PATTERN.findall(response)

        for match in matches:
            parsed = self._parse_tool_call(match)
            if parsed:
                self.tool_calls.append(parsed)

        return self.tool_calls

    def _parse_tool_call(self, raw_call: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single raw tool call string into a structured dictionary.

        Args:
            raw_call: The content inside the <tool_call> tags.

        Returns:
            A dictionary with 'name' and 'arguments' keys, or None if parsing fails.
        """
        raw_call = raw_call.strip()

        # Try to parse as JSON first
        try:
            data = json.loads(raw_call)
            if isinstance(data, dict):
                name = data.get("name", "unknown")
                arguments = data.get("arguments", {})
                return {
                    "name": name,
                    "arguments": arguments
                }
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse as key=value pairs or simple format
        # Example: name=read_file, path=/home/user/file.txt
        try:
            parts = raw_call.split(",", 1)
            if len(parts) == 2:
                name_part = parts[0].strip()
                args_part = parts[1].strip()

                # Extract name
                if "=" in name_part:
                    name = name_part.split("=", 1)[1].strip().strip("'\"")
                else:
                    name = name_part.strip()

                # Parse arguments
                arguments = self._parse_arguments(args_part)
                return {
                    "name": name,
                    "arguments": arguments
                }
        except Exception:
            pass

        # Last resort: treat the whole thing as a name with no args
        return {
            "name": raw_call.strip(),
            "arguments": {}
        }

    def _parse_arguments(self, args_str: str) -> Dict[str, Any]:
        """
        Parse an argument string into a dictionary.

        Args:
            args_str: The argument portion of the tool call.

        Returns:
            A dictionary of parsed arguments.
        """
        arguments = {}
        if not args_str:
            return arguments

        # Try JSON first
        try:
            data = json.loads(args_str)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Parse key=value pairs
        # Handle quoted values
        pattern = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\S+))")
        for match in pattern.finditer(args_str):
            key = match.group(1)
            value = match.group(2) if match.group(2) is not None else \
                    match.group(3) if match.group(3) is not None else \
                    match.group(4)
            arguments[key] = value

        return arguments

    def get_tool_calls_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all extracted tool calls with a specific name.

        Args:
            name: The tool name to filter by.

        Returns:
            A list of tool call dictionaries matching the given name.
        """
        return [tc for tc in self.tool_calls if tc["name"] == name]

    def clear(self):
        """Clear all extracted tool calls."""
        self.tool_calls = []


class ReadDirectoryTool:
    """
    Represents the read_directory tool call.
    """

    NAME = "read_directory"

    @staticmethod
    def validate(arguments: Dict[str, Any]) -> bool:
        """
        Validate the arguments for read_directory.

        Args:
            arguments: The arguments dictionary.

        Returns:
            True if valid, False otherwise.
        """
        return "path" in arguments and isinstance(arguments["path"], str)

    @staticmethod
    def execute(arguments: Dict[str, Any]) -> str:
        """
        Execute the read_directory tool.

        Args:
            arguments: The arguments dictionary containing 'path'.

        Returns:
            A string representation of the directory contents.
        """
        import os

        path = arguments["path"]

        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist."

        if not os.path.isdir(path):
            return f"Error: Path '{path}' is not a directory."

        try:
            entries = os.listdir(path)
            lines = [f"Contents of {path}:"]
            for entry in sorted(entries):
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    lines.append(f"  [DIR]  {entry}/")
                else:
                    size = os.path.getsize(full_path)
                    lines.append(f"  [FILE] {entry} ({size} bytes)")
            return "".join(lines)
        except PermissionError:
            return f"Error: Permission denied accessing '{path}'."
        except Exception as e:
            return f"Error: {str(e)}"


class ReadFileTool:
    """
    Represents the read_file tool call.
    """

    NAME = "read_file"

    @staticmethod
    def validate(arguments: Dict[str, Any]) -> bool:
        """
        Validate the arguments for read_file.

        Args:
            arguments: The arguments dictionary.

        Returns:
            True if valid, False otherwise.
        """
        return "path" in arguments and isinstance(arguments["path"], str)

    @staticmethod
    def execute(arguments: Dict[str, Any]) -> str:
        """
        Execute the read_file tool.

        Args:
            arguments: The arguments dictionary containing 'path'.

        Returns:
            The contents of the file, or an error message.
        """
        path = arguments["path"]

        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."

        if not os.path.isfile(path):
            return f"Error: Path '{path}' is not a file."

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except PermissionError:
            return f"Error: Permission denied reading '{path}'."
        except UnicodeDecodeError:
            return f"Error: File '{path}' is not a valid UTF-8 text file."
        except Exception as e:
            return f"Error: {str(e)}"


# Need to import os at module level for the tool classes
import os


class CodingHarness:
    """
    A basic coding harness that extracts and executes tool calls
    from LLM responses.
    """

    def __init__(self):
        self.extractor = ToolCallExtractor()
        self.tools = {
            ReadDirectoryTool.NAME: ReadDirectoryTool,
            ReadFileTool.NAME: ReadFileTool,
        }

    def process_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Process an LLM response, extract tool calls, and execute them.

        Args:
            response: The raw LLM response string.

        Returns:
            A list of results, each containing the tool name, arguments,
            and execution result.
        """
        tool_calls = self.extractor.extract_tool_calls(response)
        results = []

        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]

            if name not in self.tools:
                results.append({
                    "name": name,
                    "arguments": arguments,
                    "result": f"Error: Unknown tool '{name}'.",
                    "success": False
                })
                continue

            tool_class = self.tools[name]

            if not tool_class.validate(arguments):
                results.append({
                    "name": name,
                    "arguments": arguments,
                    "result": f"Error: Invalid arguments for tool '{name}'.",
                    "success": False
                })
                continue

            try:
                result = tool_class.execute(arguments)
                results.append({
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "arguments": arguments,
                    "result": f"Error: {str(e)}",
                    "success": False
                })

        return results

    def register_tool(self, name: str, tool_class: type):
        """
        Register a new tool with the harness.

        Args:
            name: The tool name.
            tool_class: The tool class with validate() and execute() methods.
        """
        self.tools[name] = tool_class

