#!/usr/bin/env python

"""
Create a copy of the PAE viewer standalone HTML with the data provided
via the CLI and open it in the default browser.

The script requires a local webserver to be run with the repository root
as the webroot. You can achieve this by changing to the repository root
directory (`pae-viewer`) and using the built-in Python HTTP module:

```bash
cd /your/download/path/pae-viewer
python3 -m http.server 8000
```

This will start an HTTP server handling requests via port 8000, which
is the default for this script. However, a custom port can be set via
the `--port` option.
"""

import argparse
import json
import os
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


def resolved_path(value: str) -> Path:
    """Validate and resolve a file path from a CLI argument.

    Args:
        value: Raw string path provided on the command line.

    Returns:
        Resolved absolute :class:`~pathlib.Path`.

    Raises:
        argparse.ArgumentTypeError: If the path does not exist.
    """
    path = Path(value).resolve()

    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")

    return path


def load_pae_viewer(
    structure_path: Path,
    chain_labels: str,
    scores_path: Optional[Path] = None,
    crosslinks_path: Optional[Path] = None,
    port: int = 8000,
) -> None:
    """Create a session file and open the PAE viewer in the default browser.

    Args:
        structure_path: Path to the structure file (PDB or CIF).
        chain_labels: Semicolon-separated list of chain labels.
        scores_path: Optional path to a JSON file containing PAE scores.
        crosslinks_path: Optional path to a TSV file containing crosslinks.
        port: Port of the local HTTP file server.
    """
    data = get_session_data(structure_path, chain_labels, scores_path, crosslinks_path)
    session_path = create_session_file(structure_path.stem, data)

    webbrowser.open(
        f"localhost:{port}/{Path(*session_path.parts[-2:])}", new=0, autoraise=True
    )


def get_session_data(
    structure_path: Path,
    chain_labels: str,
    scores_path: Optional[Path] = None,
    crosslinks_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Prepare the session data dictionary from input files.

    Args:
        structure_path: Path to the structure file.
        chain_labels: Semicolon-separated chain label string.
        scores_path: Optional path to a JSON scores file.
        crosslinks_path: Optional path to a crosslinks TSV file.

    Returns:
        Dictionary suitable for JSON serialization and embedding in the
        viewer HTML.
    """
    data = {
        "structureFile": {
            "name": structure_path.name,
            "content": structure_path.read_text(),
        },
        "chainLabels": chain_labels,
    }

    if scores_path:
        data["scoresFile"] = {
            "name": scores_path.name,
            "content": scores_path.read_text(),
        }

    if crosslinks_path:
        data["crosslinksFile"] = {
            "name": crosslinks_path.name,
            "content": crosslinks_path.read_text(),
        }

    return data


def create_session_json_element(value: Any) -> str:
    """Wrap a value as an HTML ``<script type="application/json">`` element.

    Args:
        value: JSON-serializable value to embed.

    Returns:
        HTML script element string containing the JSON data.
    """
    return (
        '    <script type="application/json" id="session-data">\n'
        f"        {json.dumps(value)}\n"
        "    </script>"
    )


def create_session_file(handle: str, data: dict[str, str]) -> Path:
    """Create a timestamped HTML session file with embedded data.

    Args:
        handle: Short identifier used in the generated file name.
        data: Session data dictionary to embed in the HTML template.

    Returns:
        Path to the newly created session HTML file.
    """
    current_dir = Path(__file__).parent.resolve()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_id = os.urandom(4).hex()

    project_root = current_dir.parents[1]
    session_path = project_root / f"{timestamp}_{random_id}_{handle}.html"
    session_path.touch()
    html_path = project_root / 'index.html'

    with (
        open(html_path, "r") as template_file,
        open(session_path, "w") as session_file,
    ):
        template = template_file.read()
        template = template.replace(
            "</head>", f"\n{create_session_json_element(data)}\n</head>"
        )

        session_file.write(template)

    return session_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-s",
        "--structure",
        help="path to structure file",
        type=resolved_path,
        required=True,
    )

    parser.add_argument(
        "-r",
        "--scores",
        help="path to JSON file containing scores",
        type=resolved_path,
        required=True,
    )

    parser.add_argument(
        "-l",
        "--labels",
        help="semicolon-separated list of chain labels",
        type=str,
    )

    parser.add_argument(
        "-c",
        "--crosslinks",
        help="path to TSV containing crosslinks",
        type=resolved_path,
    )

    parser.add_argument(
        "-p",
        "--port",
        help="port of the local file server",
        type=int,
        default=8000,
    )

    args = parser.parse_args()

    load_pae_viewer(
        args.structure, args.labels, args.scores, args.crosslinks, args.port
    )
