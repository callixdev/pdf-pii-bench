"""Tests for src/publish.py — the single-system generated RESULTS.md
and README block (the version that ships and backs the CI check)."""

import json
import sys

import pytest

from src import publish
from src.publish import (
    BEGIN_MARK,
    END_MARK,
    readme_block,
    results_markdown,
    updated_readme,
)
from src.scorer import Span, corpus_modes_to_dict, score_document_modes


@pytest.fixture()
def results():
    """A real results dict built through the actual scoring pipeline."""
    gt = [
        Span(0, 4, "Jane", "private_person"),
        Span(10, 24, "(800) 555-0100", "private_phone", "org"),
    ]
    out = {"system": "privacy-filter", "n_documents": 1, "date": "2026-08-12"}
    out.update(corpus_modes_to_dict(
        [score_document_modes([Span(0, 4, "Jane", "private_person")], gt)]))
    return out


README = f"""# my benchmark

built on 999 ground-truth spans.

| Ground truth | 999 char-offset spans |

## Reference results

{BEGIN_MARK}
stale content
{END_MARK}

## Layout
"""


class TestResultsMarkdown:
    def test_contains_computed_facts(self, results):
        md = results_markdown(results, n_spans=1121)
        assert "do not edit by hand" in md
        assert "August 12, 2026" in md
        assert "1,121 ground-truth spans" in md
        assert "privacy-filter" in md
        assert "Org/agency contact info is neutral" in md
        # 3 tables: micro row per mode
        assert md.count("| **micro** |") == 3

    def test_deterministic(self, results):
        assert (results_markdown(results, 1121)
                == results_markdown(results, 1121))


class TestUpdatedReadme:
    def test_block_and_counts_rewritten(self, results):
        out = updated_readme(README, results, n_spans=1121)
        assert "stale content" not in out
        assert "| privacy-filter |" in out
        assert "built on 1,121 ground-truth spans." in out
        assert "| Ground truth | 1,121 char-offset spans |" in out
        # untouched prose survives
        assert out.startswith("# my benchmark") and "## Layout" in out

    def test_idempotent(self, results):
        once = updated_readme(README, results, n_spans=1121)
        twice = updated_readme(once, results, n_spans=1121)
        assert once == twice

    def test_missing_markers_fails_loudly(self, results):
        with pytest.raises(SystemExit, match="markers"):
            updated_readme("# no markers here", results, n_spans=1121)

    def test_block_carries_its_own_markers(self, results):
        block = readme_block(results)
        assert block.startswith(BEGIN_MARK) and block.endswith(END_MARK)


class TestMainCheck:
    """The --check staleness gate that CI runs on every push."""

    @pytest.fixture()
    def paths(self, tmp_path, monkeypatch, results):
        results_path = tmp_path / "privacy-filter.json"
        results_path.write_text(json.dumps(results))
        readme_path = tmp_path / "README.md"
        readme_path.write_text(README)
        results_md_path = tmp_path / "RESULTS.md"
        monkeypatch.setattr(publish, "README_PATH", readme_path)
        monkeypatch.setattr(publish, "load_ground_truth",
                            lambda: {"doc1": [None, None, None]})
        return results_path, results_md_path, readme_path

    def _argv(self, paths, *extra):
        results_path, results_md_path, readme_path = paths
        return ["publish", "--results", str(results_path),
                "--results-md", str(results_md_path),
                "--readme", str(readme_path), *extra]

    def test_check_passes_on_fresh_files(self, paths, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", self._argv(paths))
        publish.main()
        monkeypatch.setattr(sys, "argv", self._argv(paths, "--check"))
        publish.main()  # must return normally, not exit
        assert "up to date" in capsys.readouterr().out

    def test_check_exits_1_on_stale_file(self, paths, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", self._argv(paths))
        publish.main()
        _, results_md_path, _ = paths
        results_md_path.write_text("hand-edited drift")
        monkeypatch.setattr(sys, "argv", self._argv(paths, "--check"))
        with pytest.raises(SystemExit) as exc:
            publish.main()
        assert exc.value.code == 1
        assert "stale" in capsys.readouterr().out
