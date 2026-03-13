#!/usr/bin/env python3
from __future__ import annotations

import ast
import io
import json
import traceback
from contextlib import redirect_stdout
from pathlib import Path


def execute_notebook(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 1

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        stdout = io.StringIO()
        outputs: list[dict[str, object]] = []

        try:
            tree = ast.parse(source, filename=str(notebook_path))
            body = tree.body
            tail_value = None

            if body and isinstance(body[-1], ast.Expr):
                expr = ast.Expression(body[-1].value)
                body = body[:-1]
                module = ast.Module(body=body, type_ignores=[])
            else:
                expr = None
                module = ast.Module(body=body, type_ignores=[])

            with redirect_stdout(stdout):
                if body:
                    exec(compile(module, str(notebook_path), "exec"), namespace)
                if expr is not None:
                    tail_value = eval(compile(expr, str(notebook_path), "eval"), namespace)

            text = stdout.getvalue()
            if text:
                outputs.append(
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": text.splitlines(True),
                    }
                )

            if tail_value is not None:
                data: dict[str, object] = {
                    "text/plain": repr(tail_value).splitlines(True) or [""]
                }
                repr_html = getattr(tail_value, "_repr_html_", None)
                if callable(repr_html):
                    html = repr_html()
                    if html is not None:
                        data["text/html"] = [html]

                outputs.append(
                    {
                        "data": data,
                        "execution_count": execution_count,
                        "metadata": {},
                        "output_type": "execute_result",
                    }
                )

            cell["outputs"] = outputs
            cell["execution_count"] = execution_count
        except Exception:
            cell["outputs"] = [
                {
                    "ename": "ExecutionError",
                    "evalue": "See traceback",
                    "output_type": "error",
                    "traceback": traceback.format_exc().splitlines(),
                }
            ]
            cell["execution_count"] = execution_count
            break

        execution_count += 1

    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Execute a simple .ipynb file and save outputs in place.")
    parser.add_argument("notebook", type=Path, help="Path to the notebook file")
    args = parser.parse_args()

    execute_notebook(args.notebook.resolve())
    print(f"Executed notebook: {args.notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
