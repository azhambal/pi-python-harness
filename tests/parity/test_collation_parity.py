"""Full Unicode collation and actual ls listings against Node/Pi, with no Node needed at test time."""

import json
from pathlib import Path

import pytest

from pi_agent._collation import case_insensitive_collation_key
from pi_agent.tools import LsTool

HERE = Path(__file__).parent
CASES = json.loads((HERE / "collation_cases.json").read_text("utf-8"))
EXPECTED = {
    item["name"]: item
    for item in json.loads((HERE / "fixtures/pi_collation_results.json").read_text("utf-8"))["results"]
}


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
async def test_unicode_order_and_ls_content_match_pi(case, tmp_path):
    want = EXPECTED[case["name"]]
    assert sorted(case["names"], key=case_insensitive_collation_key) == want["ordered"]
    for name in case["names"]:
        (tmp_path / name).write_text("")
    result = await LsTool(tmp_path).execute("c", {"limit": case["limit"]} if "limit" in case else {})
    assert [block.to_json() for block in result.content] == want["content"]
