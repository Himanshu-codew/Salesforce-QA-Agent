"""
Structured classification and deterministic section rendering for read-only
Salesforce query results (response fidelity).

A Salesforce REST ``/query`` result is a polymorphic JSON document, but every
downstream consumer currently re-infers its meaning from raw shape:

- record list   ``{"totalSize": N, "records": [{...record fields...}]}``
- bare array    ``[ {...}, ... ]``  (REST ``/recent`` fallback)
- COUNT         ``{"totalSize": 1, "records": [{"expr0": N}]}`` (or ``"count"``)
- aggregate     ``{"totalSize": N, "records": [{"expr0": N, "<group>": ...}, ...]}``
- empty         ``{"totalSize": 0, "records": []}``
- hierarchical  records contain subquery collections ``{totalSize, records}``

This module classifies results into semantic types so a final answer never
mislabels an aggregate/COUNT row as a business record ("Total: 1 record" for a
COUNT of 66), and renders answers as labeled, deterministic sections so counts
are attributed to their object and COUNT/SUM/GROUP-BY values are never reported
as record counts.

Only data-bearing listing tools are eligible (kept conservative so related
record results, schema payloads, errors, and getUserInfo keep their existing
LLM synthesis path). Single record-list / count results are intentionally left
to the existing byte-for-byte fast path while single aggregate results are
intercepted and rendered as aggregate tables (the flat formatter must never
collapse a GROUP BY into a count line).
"""
import json
import re
from dataclasses import dataclass

from .agent import (
    _aggregate_group_columns,
    _classify_expr_count,
    _is_soql_count,
    _plural,
    _render_aggregate_markdown,
    format_sf_records_as_markdown,
)

_FLAT_LIST_TOOLS = {"soqlQuery", "listRecentSobjectRecords"}


@dataclass
class ResultInfo:
    """Semantic classification of one read-only listing tool result."""

    result_type: str
    label: str
    result: str
    tool_name: str = "soqlQuery"
    soql: str = ""
    count: int | None = None


def _to_int(value) -> int:
    try:
        num = int(float(str(value).replace(",", "")))
        return num
    except (ValueError, TypeError):
        return 0


def _object_label(data, records, soql: str, tool_name: str) -> str:
    if records:
        first = records[0]
        if isinstance(first, dict) and isinstance(first.get("attributes"), dict):
            obj_type = first["attributes"].get("type")
            if obj_type:
                return str(obj_type)
    m = re.search(r"\bFROM\s+(\w+)", soql, re.IGNORECASE)
    if m:
        return m.group(1)
    return "Records"


def _has_subquery(record) -> bool:
    if not isinstance(record, dict):
        return False
    return any(
        isinstance(v, dict) and isinstance(v.get("records"), list) and "totalSize" in v
        for v in record.values()
    )


def classify_result(tool_name: str, result: str, soql: str = "") -> ResultInfo | None:
    """Classify one tool result into a semantic ResultInfo.

    Returns ``None`` when the result is not a classifiable read-only data shape
    (hierarchical records, non-listing tools, error/schema payloads, malformed
    JSON) so the caller keeps its existing LLM synthesis behavior.
    """
    if tool_name not in _FLAT_LIST_TOOLS:
        return None
    if not isinstance(result, str):
        return None
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if "records" not in data and "totalSize" not in data:
            return None
        records = data.get("records", []) or []
    else:
        return None

    label = _object_label(data, records, soql, tool_name)
    if not records:
        if tool_name == "soqlQuery" and soql and _is_soql_count(soql):
            return ResultInfo("count", label, result=result, tool_name=tool_name, soql=soql, count=0)
        return ResultInfo("empty", label, result=result, tool_name=tool_name, soql=soql)

    first = records[0]
    if isinstance(first, dict):
        # Uses THE shared expr0/count classifier from agent.py so classification
        # can never diverge from format_sf_records_as_markdown.
        agg_kind = _classify_expr_count(first, records, soql)
        if agg_kind is not None:
            if agg_kind == "aggregate":
                return ResultInfo("aggregate", label, result=result, tool_name=tool_name, soql=soql)
            value = first.get("expr0", first.get("count", 0))
            return ResultInfo("count", label, result=result, tool_name=tool_name, soql=soql, count=_to_int(value))
        if _has_subquery(first):
            return None
        # total_count-only envelope (no expr0/count on the record): a COUNT shape
        # produced by an executor/serializer that folds the value up top.
        if (
            isinstance(data, dict)
            and isinstance(data.get("total_count"), (int, float))
            and len(records) == 1
            and _to_int(data["total_count"]) > 0
            and not any(k not in ("attributes",) for k in first)
        ):
            return ResultInfo("count", label, result=result, tool_name=tool_name, soql=soql, count=_to_int(data["total_count"]))
        return ResultInfo(
            "single_record" if len(records) == 1 else "record_list",
            label,
            result=result,
            tool_name=tool_name,
            soql=soql,
        )
    return None


def _aggregate_section(info: ResultInfo) -> list[str] | None:
    """Build the labeled '### {Object} by {group...}' section + aggregate table.

    Table rendering and metric naming are delegated to the shared helpers in
    agent.py so grouped/SUM/AVG aggregates render identically everywhere.
    """
    try:
        data = json.loads(info.result)
    except (json.JSONDecodeError, TypeError):
        return None
    records = data.get("records", []) if isinstance(data, dict) else data
    if not records or not isinstance(records[0], dict):
        return None
    first = records[0]
    col_names = _aggregate_group_columns(first)
    table = _render_aggregate_markdown(records, info.soql)
    if not table:
        return None
    title = f"### {_plural(info.label)}"
    if col_names:
        title += f" by {', '.join(col_names)}"
    else:
        title += " Summary"
    return [title, "", table]


def render_sections(infos: list[ResultInfo]) -> str | None:
    """Render labeled deterministic markdown sections for classified results.

    Returns ``None`` when any result cannot be rendered deterministically, so the
    caller falls back to its existing synthesis path.
    """
    sections: list[str] = []
    for info in infos or []:
        if info.result_type == "count":
            count = info.count if info.count is not None else 0
            sections.append(f"**Total {_plural(info.label)}: {count:,}**")
            continue
        if info.result_type == "empty":
            sections.append(f"**No {_plural(info.label)} found**")
            continue
        if info.result_type in ("record_list", "single_record"):
            table = format_sf_records_as_markdown(
                info.result, tool_name=info.tool_name, soql_query=info.soql
            )
            if not table:
                return None
            sections.append(f"### {_plural(info.label)} Found\n\n{table}")
            continue
        if info.result_type == "aggregate":
            agg = _aggregate_section(info)
            if not agg:
                return None
            sections.append("\n".join(agg))
            continue
        return None
    return "\n\n---\n\n".join(sections) if sections else None