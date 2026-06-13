from typing import Optional, Annotated

from pydantic import Field
from mcp.types import TextContent
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from .cli import init, cli_parser, rewrap_text
from .config import apply_config


class SembrModel:
    def __init__(self, tokenizer, model, default_file_type=None, config=None):
        self.tokenizer = tokenizer
        self.model = model
        self.default_file_type = default_file_type
        self.config = config or {}

    def process_text(self, text: str, file_type: Optional[str] = None) -> str:
        _, wrapped = rewrap_text(
            text,
            self.tokenizer,
            self.model,
            self.config,
            file_type=file_type or self.default_file_type)
        return wrapped


_sembr_model: Optional[SembrModel] = None


def get_sembr_model() -> SembrModel:
    """Get the initialized SemBr model instance."""
    global _sembr_model
    if _sembr_model is not None:
        return _sembr_model
    parser = cli_parser()
    args, _ = parser.parse_known_args()
    apply_config(args, args.config)
    config = vars(args)
    tokenizer, model, _ = init(
        args.model_name, args.bits, args.dtype, args.file_type,
        spaces=config['spaces'], indent_type=config['indent_type'],
        backend=args.backend, quantization=args.quantization)
    _sembr_model = SembrModel(tokenizer, model, args.file_type, config)
    return _sembr_model


mcp = FastMCP("SemBr")


@mcp.tool(
    description="Apply semantic line breaks to text",
    tags=["sembr", "semantic linebreak", "format", "string"],
)
def wrap_text(
    text: Annotated[str, Field(description="Text to wrap")],
    file_type: Annotated[Optional[str], Field(
        description=(
            "File type (latex, markdown, plaintext, etc.). "
            "Auto-detect if not provided."),
        default=None
    )] = None,
) -> ToolResult:
    try:
        wrapped_text = get_sembr_model().process_text(text, file_type)
    except Exception as e:
        return ToolResult(
            content=[TextContent(type="text", text=f"Error processing text: {str(e)}")],
            structured_content={"success": False, "error": str(e)})
    num_lines = len(wrapped_text.splitlines())
    readable = f"Performed semantic line breaks to {num_lines} lines."
    if file_type:
        readable += f" File type: {file_type}."
    return ToolResult(
        content=[TextContent(type="text", text=readable)],
        structured_content={"success": True, "output": wrapped_text, "file_type": file_type})


if __name__ == "__main__":
    mcp.run()
