from typing import Optional, Annotated

from pydantic import Field
from mcp.types import TextContent
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from .cli import init, cli_parser, wrap_kwargs


class SembrModel:
    def __init__(self, tokenizer, model, processor, kwargs):
        self.tokenizer = tokenizer
        self.model = model
        self.processor = processor
        self.kwargs = kwargs

    def process_text(self, text: str) -> str:
        from .inference import sembr
        return sembr(
            text, self.tokenizer, self.model, self.processor, **self.kwargs)


_sembr_model: Optional[SembrModel] = None


def get_sembr_model() -> SembrModel:
    """Get the initialized SemBr model instance."""
    global _sembr_model
    if _sembr_model is not None:
        return _sembr_model
    parser = cli_parser()
    args, _ = parser.parse_known_args()
    tokenizer, model, processor = init(args.model_name, args.bits, args.dtype)
    kwargs = wrap_kwargs(args)
    _sembr_model = SembrModel(tokenizer, model, processor, kwargs)
    return _sembr_model


mcp = FastMCP("SemBr")


@mcp.tool(
    description="Apply semantic line breaks to text",
    tags=["sembr", "semantic linebreak", "format", "string"],
)
def wrap_text(
    text: Annotated[str, Field(description="Text to wrap")],
) -> ToolResult:
    try:
        wrapped_text = get_sembr_model().process_text(text)
    except Exception as e:
        return ToolResult(
            content=[TextContent(type="text", text=f"Error processing text: {str(e)}")],
            structured_content={"success": False, "error": str(e)})
    num_lines = len(wrapped_text.splitlines())
    readable = f"Performed semantic line breaks to {num_lines} lines."
    return ToolResult(
        content=[TextContent(type="text", text=readable)],
        structured_content={"success": True, "output": wrapped_text})


@mcp.tool(
    description="Apply semantic line breaks to file",
    tags=["sembr", "semantic linebreak", "format", "file"],
)
def process_file(
    file_path: Annotated[str, Field(description="File path to process")],
) -> ToolResult:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        return ToolResult(
            content=[TextContent(type="text", text=f"Error reading file: {file_path}")],
            structured_content={"success": False, "error": str(e)})
    return wrap_text(text)


if __name__ == "__main__":
    mcp.run()
