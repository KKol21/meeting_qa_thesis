"""Load version-controlled prompt text bundled with the package."""

from importlib.resources import files


def load_prompt(filename: str) -> str:
    """Load a prompt while ignoring only the text file's final newline."""

    prompt = files("meeting_qa_chunking").joinpath("prompts", filename)
    return prompt.read_text(encoding="utf-8").removesuffix("\n")
