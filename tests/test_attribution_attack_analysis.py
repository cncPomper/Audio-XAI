"""Tests for attribution_attack_analysis scripts added in PR."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers for importing standalone scripts that are not installable packages.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_AUXILIARY_DIR = _REPO_ROOT / "audio_xai" / "attribution_attack_analysis" / "auxiliary_scripts"
_INDIVIDUAL_DIR = _REPO_ROOT / "audio_xai" / "attribution_attack_analysis" / "individual_sample_analysis"


def _import_script(script_path: Path):
    """Import a standalone .py script as a module."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None:
        raise ImportError(f"Cannot create module spec for {script_path}")
    if spec.loader is None:
        raise ImportError(f"No loader available for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



normalize_mod = _import_script(_AUXILIARY_DIR / "normalize_attack_json_schema.py")
replace_mod = _import_script(_AUXILIARY_DIR / "replace_xshift.py")
analyze_mod = _import_script(_INDIVIDUAL_DIR / "analyze_audio_easy_hard.py")
select_mod = _import_script(_INDIVIDUAL_DIR / "select_global_easy_hard_samples.py")


# ===========================================================================
# Tests for normalize_attack_json_schema.py
# ===========================================================================


class TestNormalizeRecord:
    """Tests for normalize_record()."""

    def test_adds_cos_sim_from_alias(self):
        record = {"cos_orig_adv": 0.42}
        updates = normalize_mod.normalize_record(record, overwrite=False)
        assert record["cos_sim"] == 0.42
        assert any("cos_sim" in u for u in updates)

    def test_adds_top10_overlap_from_alias(self):
        record = {"top10_overlap_orig": 0.15}
        updates = normalize_mod.normalize_record(record, overwrite=False)
        assert record["top10_overlap"] == 0.15
        assert any("top10_overlap" in u for u in updates)

    def test_does_not_overwrite_existing_canonical_field_by_default(self):
        record = {"cos_orig_adv": 0.99, "cos_sim": 0.10}
        updates = normalize_mod.normalize_record(record, overwrite=False)
        assert record["cos_sim"] == 0.10  # not overwritten
        assert updates == []

    def test_overwrites_canonical_field_when_flag_set(self):
        record = {"cos_orig_adv": 0.99, "cos_sim": 0.10}
        updates = normalize_mod.normalize_record(record, overwrite=True)
        assert record["cos_sim"] == 0.99
        assert any("overwrote" in u for u in updates)

    def test_no_alias_key_produces_no_update(self):
        record = {"some_other_key": 1}
        updates = normalize_mod.normalize_record(record, overwrite=False)
        assert updates == []
        assert "cos_sim" not in record
        assert "top10_overlap" not in record

    def test_null_canonical_field_is_replaced(self):
        record = {"cos_orig_adv": 0.5, "cos_sim": None}
        updates = normalize_mod.normalize_record(record, overwrite=False)
        assert record["cos_sim"] == 0.5
        assert len(updates) == 1

    def test_both_aliases_processed_together(self):
        record = {"cos_orig_adv": 0.3, "top10_overlap_orig": 0.07}
        normalize_mod.normalize_record(record, overwrite=False)
        assert record["cos_sim"] == 0.3
        assert record["top10_overlap"] == 0.07

    def test_overwrite_no_change_when_values_equal(self):
        record = {"cos_orig_adv": 0.5, "cos_sim": 0.5}
        updates = normalize_mod.normalize_record(record, overwrite=True)
        # Same value → no overwrite logged
        assert updates == []


class TestNormalizeJsonData:
    """Tests for normalize_json_data()."""

    def test_single_dict(self):
        data = {"cos_orig_adv": 0.2}
        updates = normalize_mod.normalize_json_data(data, overwrite=False)
        assert data["cos_sim"] == 0.2
        assert len(updates) == 1

    def test_list_of_dicts(self):
        data = [{"cos_orig_adv": 0.1}, {"top10_overlap_orig": 0.05}]
        updates = normalize_mod.normalize_json_data(data, overwrite=False)
        assert data[0]["cos_sim"] == 0.1
        assert data[1]["top10_overlap"] == 0.05
        assert len(updates) == 2

    def test_nested_results_key(self):
        data = {"results": [{"cos_orig_adv": 0.3}]}
        updates = normalize_mod.normalize_json_data(data, overwrite=False)
        assert data["results"][0]["cos_sim"] == 0.3
        assert len(updates) == 1

    def test_nested_items_key(self):
        data = {"items": [{"top10_overlap_orig": 0.2}]}
        updates = normalize_mod.normalize_json_data(data, overwrite=False)
        assert data["items"][0]["top10_overlap"] == 0.2

    def test_nested_samples_key(self):
        data = {"samples": [{"cos_orig_adv": 0.8}]}
        updates = normalize_mod.normalize_json_data(data, overwrite=False)
        assert data["samples"][0]["cos_sim"] == 0.8

    def test_nested_data_key(self):
        data = {"data": [{"cos_orig_adv": 0.9}]}
        normalize_mod.normalize_json_data(data, overwrite=False)
        assert data["data"][0]["cos_sim"] == 0.9

    def test_non_dict_items_in_list_are_skipped(self):
        data = ["not_a_dict", 42, None]
        updates = normalize_mod.normalize_json_data(data, overwrite=False)
        assert updates == []

    def test_empty_list(self):
        data = []
        updates = normalize_mod.normalize_json_data(data, overwrite=False)
        assert updates == []


class TestIterJsonFiles:
    """Tests for iter_json_files()."""

    def test_single_json_file(self, tmp_path):
        f = tmp_path / "sample.json"
        f.write_text("{}", encoding="utf-8")
        result = normalize_mod.iter_json_files([f])
        assert f.resolve() in result

    def test_non_json_file_is_skipped(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        result = normalize_mod.iter_json_files([f])
        assert result == []

    def test_directory_returns_all_json_files(self, tmp_path):
        (tmp_path / "a.json").write_text("{}", encoding="utf-8")
        (tmp_path / "b.json").write_text("{}", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")
        result = normalize_mod.iter_json_files([tmp_path])
        stems = {p.stem for p in result}
        assert {"a", "b"} == stems

    def test_duplicate_files_are_deduplicated(self, tmp_path):
        f = tmp_path / "dup.json"
        f.write_text("{}", encoding="utf-8")
        result = normalize_mod.iter_json_files([f, f])
        assert len(result) == 1

    def test_nonexistent_path_is_ignored(self, tmp_path):
        nonexistent = tmp_path / "no_such_dir"
        result = normalize_mod.iter_json_files([nonexistent])
        assert result == []

    def test_recursive_directory_search(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.json").write_text("{}", encoding="utf-8")
        result = normalize_mod.iter_json_files([tmp_path])
        assert any("nested" in p.name for p in result)


class TestProcessFile:
    """Tests for process_file()."""

    def test_dry_run_does_not_modify_file(self, tmp_path):
        f = tmp_path / "test.json"
        original = {"cos_orig_adv": 0.5}
        f.write_text(json.dumps(original), encoding="utf-8")
        result = normalize_mod.process_file(f, dry_run=True, overwrite=False, backup=False, indent=2)
        assert result.changed is True
        content = json.loads(f.read_text())
        # File must not have been modified by dry-run
        assert "cos_sim" not in content

    def test_write_mode_modifies_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"cos_orig_adv": 0.5}), encoding="utf-8")
        normalize_mod.process_file(f, dry_run=False, overwrite=False, backup=False, indent=2)
        content = json.loads(f.read_text())
        assert content["cos_sim"] == 0.5

    def test_backup_file_created(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"cos_orig_adv": 0.7}), encoding="utf-8")
        normalize_mod.process_file(f, dry_run=False, overwrite=False, backup=True, indent=2)
        assert (tmp_path / "test.json.bak").exists()

    def test_no_backup_flag_suppresses_backup(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"cos_orig_adv": 0.7}), encoding="utf-8")
        normalize_mod.process_file(f, dry_run=False, overwrite=False, backup=False, indent=2)
        assert not (tmp_path / "test.json.bak").exists()

    def test_unchanged_file_returns_changed_false(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"pred_orig": 1}), encoding="utf-8")
        result = normalize_mod.process_file(f, dry_run=False, overwrite=False, backup=False, indent=2)
        assert result.changed is False

    def test_invalid_json_returns_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("NOT VALID JSON{{", encoding="utf-8")
        result = normalize_mod.process_file(f, dry_run=False, overwrite=False, backup=False, indent=2)
        assert result.error is not None


# ===========================================================================
# Tests for replace_xshift.py (rename_keys)
# ===========================================================================


class TestRenameKeys:
    """Tests for rename_keys()."""

    def test_renames_lrp_cos_sim(self):
        obj = {"lrp_cos_sim": 0.5}
        result = replace_mod.rename_keys(obj)
        assert "cos_sim" in result
        assert result["cos_sim"] == 0.5
        assert "lrp_cos_sim" not in result

    def test_renames_lrp_top10_overlap(self):
        obj = {"lrp_top10_overlap": 0.1}
        result = replace_mod.rename_keys(obj)
        assert result["top10_overlap"] == 0.1

    def test_unknown_keys_pass_through(self):
        obj = {"some_key": 42}
        result = replace_mod.rename_keys(obj)
        assert result["some_key"] == 42

    def test_recursive_in_nested_dict(self):
        obj = {"outer": {"lrp_cos_sim": 0.9}}
        result = replace_mod.rename_keys(obj)
        assert result["outer"]["cos_sim"] == 0.9

    def test_recursive_in_list(self):
        obj = [{"lrp_cos_sim": 0.3}, {"other": 1}]
        result = replace_mod.rename_keys(obj)
        assert result[0]["cos_sim"] == 0.3
        assert result[1]["other"] == 1

    def test_non_dict_non_list_passthrough(self):
        assert replace_mod.rename_keys(42) == 42
        assert replace_mod.rename_keys("hello") == "hello"
        assert replace_mod.rename_keys(None) is None

    def test_key_collision_raises_value_error(self):
        # Both the original key and the renamed key exist → collision.
        obj = {"lrp_cos_sim": 0.5, "cos_sim": 0.2}
        with pytest.raises(ValueError, match="Kolizja"):
            replace_mod.rename_keys(obj)

    def test_empty_dict(self):
        assert replace_mod.rename_keys({}) == {}

    def test_deeply_nested(self):
        obj = {"level1": {"level2": {"lrp_top10_overlap": 0.05}}}
        result = replace_mod.rename_keys(obj)
        assert result["level1"]["level2"]["top10_overlap"] == 0.05

    def test_values_are_unchanged(self):
        obj = {"lrp_cos_sim": [1, 2, 3]}
        result = replace_mod.rename_keys(obj)
        assert result["cos_sim"] == [1, 2, 3]


# ===========================================================================
# Tests for analyze_audio_easy_hard.py
# ===========================================================================


class TestReadNames:
    """Tests for read_names()."""

    def test_reads_simple_names(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("sample_a\nsample_b\nsample_c\n", encoding="utf-8")
        result = analyze_mod.read_names(f)
        assert result == ["sample_a", "sample_b", "sample_c"]

    def test_skips_comment_lines(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("# this is a comment\nsample_x\n", encoding="utf-8")
        result = analyze_mod.read_names(f)
        assert result == ["sample_x"]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("sample_a\n\n\nsample_b\n", encoding="utf-8")
        result = analyze_mod.read_names(f)
        assert result == ["sample_a", "sample_b"]

    def test_deduplicates_names(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("dup\ndup\nunique\n", encoding="utf-8")
        result = analyze_mod.read_names(f)
        assert result == ["dup", "unique"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("  sample_a  \n  sample_b\n", encoding="utf-8")
        result = analyze_mod.read_names(f)
        assert result == ["sample_a", "sample_b"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("", encoding="utf-8")
        result = analyze_mod.read_names(f)
        assert result == []


class TestFindAudioForName:
    """Tests for find_audio_for_name()."""

    def _make_paths(self, names: list[str], ext: str = ".wav") -> list[Path]:
        return [Path(f"/fake/{n}{ext}") for n in names]

    def test_exact_match_found(self):
        files = self._make_paths(["sample_001", "sample_002"])
        result = analyze_mod.find_audio_for_name("sample_001", files, "exact")
        assert result is not None
        assert result.stem == "sample_001"

    def test_exact_match_not_found_returns_none(self):
        files = self._make_paths(["sample_001"])
        result = analyze_mod.find_audio_for_name("sample_xyz", files, "exact")
        assert result is None

    def test_contains_match(self):
        files = self._make_paths(["my_sample_001_v2"])
        result = analyze_mod.find_audio_for_name("sample_001", files, "contains")
        assert result is not None

    def test_contains_mode_prefers_shorter_path(self):
        short = Path("/a/sample_001.wav")
        long = Path("/a/b/c/d/e/sample_001.wav")
        result = analyze_mod.find_audio_for_name("sample_001", [long, short], "contains")
        assert result == short

    def test_exact_mode_ignores_contains_match(self):
        files = self._make_paths(["prefix_sample_x"])
        result = analyze_mod.find_audio_for_name("sample_x", files, "exact")
        assert result is None

    def test_multiple_exact_matches_returns_shortest(self):
        short = Path("/x/sample_a.wav")
        long = Path("/x/y/z/sample_a.wav")
        result = analyze_mod.find_audio_for_name("sample_a", [long, short], "exact")
        assert result == short


class TestFrameSignal:
    """Tests for frame_signal()."""

    def test_basic_framing(self):
        y = np.ones(1000, dtype=np.float64)
        frames = analyze_mod.frame_signal(y, frame_len=100, hop_len=50)
        expected_frames = 1 + (1000 - 100) // 50
        assert frames.shape == (expected_frames, 100)

    def test_signal_shorter_than_frame_is_padded(self):
        y = np.ones(10, dtype=np.float64)
        frames = analyze_mod.frame_signal(y, frame_len=50, hop_len=25)
        assert frames.shape[1] == 50

    def test_output_is_copy(self):
        y = np.arange(100, dtype=np.float64)
        frames = analyze_mod.frame_signal(y, frame_len=10, hop_len=5)
        frames[0, 0] = 9999.0
        # Original y must be untouched.
        assert y[0] == 0.0

    def test_single_frame(self):
        y = np.zeros(10, dtype=np.float64)
        frames = analyze_mod.frame_signal(y, frame_len=10, hop_len=10)
        assert frames.shape[0] >= 1


class TestSafeDb:
    """Tests for safe_db()."""

    def test_positive_value(self):
        result = analyze_mod.safe_db(1.0)
        assert abs(result) < 1e-6  # 20*log10(1) == 0

    def test_zero_value_does_not_raise(self):
        # Should not raise; eps prevents log(0).
        result = analyze_mod.safe_db(0.0)
        assert result < 0  # Large negative dB

    def test_array_input(self):
        x = np.array([1.0, 0.1, 0.01])
        result = analyze_mod.safe_db(x)
        assert result.shape == x.shape
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] == pytest.approx(-20.0, abs=1e-6)
        assert result[2] == pytest.approx(-40.0, abs=1e-6)

    def test_value_below_eps_clamped(self):
        result_eps = analyze_mod.safe_db(1e-12)
        result_zero = analyze_mod.safe_db(0.0)
        assert result_eps == result_zero


class TestCohenD:
    """Tests for cohen_d()."""

    def test_identical_groups_returns_nan(self):
        x = pd.Series([1.0, 1.0, 1.0])
        y = pd.Series([1.0, 1.0, 1.0])
        result = analyze_mod.cohen_d(x, y)
        assert np.isnan(result)

    def test_known_effect_size(self):
        # Two groups with known means and std.
        x = pd.Series([2.0, 2.0, 2.0, 2.0])
        y = pd.Series([0.0, 0.0, 0.0, 0.0])
        # All values identical within each group → pooled std = 0 → nan, unless we vary values.
        x2 = pd.Series([2.0, 2.1, 1.9, 2.05])
        y2 = pd.Series([0.0, 0.1, -0.1, 0.05])
        result = analyze_mod.cohen_d(x2, y2)
        assert isinstance(result, float)
        assert result > 5  # Large effect because means differ by ~2 with small std

    def test_returns_nan_for_single_element_groups(self):
        x = pd.Series([1.0])
        y = pd.Series([2.0])
        result = analyze_mod.cohen_d(x, y)
        assert np.isnan(result)

    def test_positive_effect_when_x_greater(self):
        x = pd.Series([10.0, 10.5, 9.5, 10.2])
        y = pd.Series([5.0, 5.5, 4.5, 5.2])
        result = analyze_mod.cohen_d(x, y)
        assert result > 0

    def test_negative_effect_when_y_greater(self):
        x = pd.Series([1.0, 1.1, 0.9])
        y = pd.Series([5.0, 5.1, 4.9])
        result = analyze_mod.cohen_d(x, y)
        assert result < 0

    def test_handles_nan_in_series(self):
        x = pd.Series([1.0, np.nan, 2.0, 3.0])
        y = pd.Series([0.0, 0.5, np.nan, 0.1])
        result = analyze_mod.cohen_d(x, y)
        assert isinstance(result, float)


class TestCompareGroups:
    """Tests for compare_groups()."""

    def _make_feature_df(self):
        return pd.DataFrame(
            {
                "group": ["easy"] * 5 + ["hard"] * 5,
                "feature_a": [1.0, 1.1, 0.9, 1.2, 0.8, 5.0, 5.1, 4.9, 5.2, 4.8],
                "feature_b": [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2],
                "sample_rate": [16000] * 10,  # should be excluded
            }
        )

    def test_returns_dataframe(self):
        df = self._make_feature_df()
        result = analyze_mod.compare_groups(df)
        assert isinstance(result, pd.DataFrame)

    def test_sample_rate_excluded(self):
        df = self._make_feature_df()
        result = analyze_mod.compare_groups(df)
        assert "sample_rate" not in result["feature"].values

    def test_sorted_by_abs_cohen_d_descending(self):
        df = self._make_feature_df()
        result = analyze_mod.compare_groups(df)
        assert not result.empty
        # feature_a should have larger |d| than feature_b (well-separated groups)
        abs_d = result["abs_cohen_d"].values
        assert all(abs_d[i] >= abs_d[i + 1] for i in range(len(abs_d) - 1))

    def test_required_columns_present(self):
        df = self._make_feature_df()
        result = analyze_mod.compare_groups(df)
        for col in ["feature", "easy_mean", "hard_mean", "cohen_d_easy_minus_hard", "mannwhitney_p", "abs_cohen_d"]:
            assert col in result.columns

    def test_empty_df_when_no_overlap(self):
        # No hard group samples
        df = pd.DataFrame({"group": ["easy"] * 3, "x": [1, 2, 3]})
        result = analyze_mod.compare_groups(df)
        assert result.empty


# ===========================================================================
# Tests for select_global_easy_hard_samples.py
# ===========================================================================


def _make_base_df(n: int = 6) -> pd.DataFrame:
    """Return a minimal DataFrame satisfying add_missing_case_metrics requirements."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "stem": [f"sample_{i}" for i in range(n)],
            "model": ["AST"] * n,
            "attack": ["PGD"] * n,
            "pred_orig": rng.integers(0, 2, n),
            "pred_adv": rng.integers(0, 2, n),
            "prob_orig": rng.uniform(0, 1, n),
            "prob_adv": rng.uniform(0, 1, n),
            "cos_sim": rng.uniform(0, 1, n),
            "top10_overlap": rng.uniform(0, 1, n),
        }
    )


class TestRequireColumns:
    """Tests for require_columns()."""

    def test_passes_when_all_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        select_mod.require_columns(df, ["a", "b"])  # should not raise

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            select_mod.require_columns(df, ["a", "missing_col"])

    def test_error_includes_column_name(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="missing_col"):
            select_mod.require_columns(df, ["missing_col"])

    def test_context_appears_in_error(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="my_context"):
            select_mod.require_columns(df, ["missing_col"], context="my_context")


class TestNormalizeQualityScores:
    """Tests for normalize_quality_scores()."""

    def test_pesq_normalization(self):
        df = pd.DataFrame({"pesq": [1.0, 4.5, 2.75]})
        result = select_mod.normalize_quality_scores(df)
        assert result["pesq_norm"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["pesq_norm"].iloc[1] == pytest.approx(1.0, abs=1e-6)

    def test_stoi_normalization_is_clipped(self):
        df = pd.DataFrame({"stoi": [0.0, 1.0, 1.5, -0.1]})
        result = select_mod.normalize_quality_scores(df)
        assert result["stoi_norm"].max() <= 1.0
        assert result["stoi_norm"].min() >= 0.0

    def test_visqol_normalization(self):
        df = pd.DataFrame({"visqol": [1.0, 5.0]})
        result = select_mod.normalize_quality_scores(df)
        assert result["visqol_norm"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["visqol_norm"].iloc[1] == pytest.approx(1.0, abs=1e-6)

    def test_peaq_normalization(self):
        df = pd.DataFrame({"peaq": [-4.0, 0.0]})
        result = select_mod.normalize_quality_scores(df)
        assert result["peaq_norm"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["peaq_norm"].iloc[1] == pytest.approx(1.0, abs=1e-6)

    def test_zimtohrli_normalization(self):
        df = pd.DataFrame({"zimtohrli": [0.0, 5.0]})
        result = select_mod.normalize_quality_scores(df)
        assert result["zimtohrli_norm"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["zimtohrli_norm"].iloc[1] == pytest.approx(1.0, abs=1e-6)

    def test_quality_score_is_mean_of_norms(self):
        df = pd.DataFrame({"pesq": [1.0], "stoi": [0.0]})
        result = select_mod.normalize_quality_scores(df)
        expected = (0.0 + 0.0) / 2
        assert result["quality_score"].iloc[0] == pytest.approx(expected, abs=1e-6)

    def test_no_quality_cols_defaults_to_one(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = select_mod.normalize_quality_scores(df)
        assert (result["quality_score"] == 1.0).all()

    def test_original_df_not_modified(self):
        df = pd.DataFrame({"pesq": [2.0]})
        original_cols = list(df.columns)
        select_mod.normalize_quality_scores(df)
        assert list(df.columns) == original_cols  # original unchanged


class TestAddMissingCaseMetrics:
    """Tests for add_missing_case_metrics()."""

    def test_adds_pred_preserved_calc(self):
        df = _make_base_df()
        result = select_mod.add_missing_case_metrics(df, 0.5)
        assert "pred_preserved_calc" in result.columns
        expected = (df["pred_orig"] == df["pred_adv"]).astype(int)
        pd.testing.assert_series_equal(result["pred_preserved_calc"].reset_index(drop=True),
                                       expected.reset_index(drop=True),
                                       check_names=False)

    def test_adds_attribution_change(self):
        df = _make_base_df()
        result = select_mod.add_missing_case_metrics(df, 0.5)
        assert "attribution_change" in result.columns
        assert (result["attribution_change"] >= 0).all()
        assert (result["attribution_change"] <= 1).all()

    def test_adds_margin_safety(self):
        df = _make_base_df()
        result = select_mod.add_missing_case_metrics(df, 0.5)
        assert "margin_safety" in result.columns
        assert (result["margin_safety"] >= 0).all()
        assert (result["margin_safety"] <= 1).all()

    def test_adds_afs_stable(self):
        df = _make_base_df()
        result = select_mod.add_missing_case_metrics(df, 0.5)
        assert "afs_stable" in result.columns
        assert (result["afs_stable"] >= 0).all()

    def test_does_not_overwrite_existing_columns(self):
        df = _make_base_df()
        df["afs_stable"] = 999.0
        result = select_mod.add_missing_case_metrics(df, 0.5)
        # Already present → not recalculated
        assert (result["afs_stable"] == 999.0).all()

    def test_raises_on_missing_required_columns(self):
        df = pd.DataFrame({"pred_orig": [1, 0]})  # missing many required cols
        with pytest.raises(ValueError, match="Missing required columns"):
            select_mod.add_missing_case_metrics(df, 0.5)

    def test_quality_score_defaults_to_one_without_quality_cols(self):
        df = _make_base_df()
        result = select_mod.add_missing_case_metrics(df, 0.5)
        # No pesq/stoi/visqol → quality_score should be 1.0
        assert (result["quality_score"] == 1.0).all()

    def test_quality_score_calculated_from_pesq_stoi(self):
        df = _make_base_df()
        df["pesq"] = 2.75
        df["stoi"] = 0.90
        result = select_mod.add_missing_case_metrics(df, 0.5)
        # pesq_norm = (2.75-1)/(4.5-1) = 0.5, stoi_norm = 0.9 → quality = 0.7
        assert (result["quality_score"] > 0).all()
        assert (result["quality_score"] <= 1).all()

    def test_cos_sim_negative_uses_alternative_normalization(self):
        df = _make_base_df()
        df["cos_sim"] = -0.5  # negative cos_sim triggers alternative branch
        result = select_mod.add_missing_case_metrics(df, 0.5)
        assert "attribution_change" in result.columns
        assert (result["attribution_change"] >= 0).all()

    def test_margin_safety_clamped_to_zero_one(self):
        df = _make_base_df()
        # prob_adv very far from threshold → margin_safety should cap at 1
        df["prob_adv"] = 0.99
        result = select_mod.add_missing_case_metrics(df, 0.5)
        assert (result["margin_safety"] <= 1.0).all()


class TestMakeSampleSummary:
    """Tests for make_sample_summary()."""

    def test_returns_dataframe(self):
        df = _make_base_df()
        df = select_mod.add_missing_case_metrics(df, 0.5)
        summary = select_mod.make_sample_summary(df, "stem")
        assert isinstance(summary, pd.DataFrame)

    def test_one_row_per_sample(self):
        df = _make_base_df(n=6)
        df = select_mod.add_missing_case_metrics(df, 0.5)
        summary = select_mod.make_sample_summary(df, "stem")
        assert len(summary) == 6

    def test_afs_stable_columns_present(self):
        df = _make_base_df()
        df = select_mod.add_missing_case_metrics(df, 0.5)
        summary = select_mod.make_sample_summary(df, "stem")
        for col in ["afs_stable_mean", "afs_stable_median", "afs_stable_max", "afs_stable_min"]:
            assert col in summary.columns

    def test_multiple_rows_per_sample_are_aggregated(self):
        # Create two rows per sample (different attacks)
        df = pd.DataFrame(
            {
                "stem": ["s1", "s1", "s2", "s2"],
                "model": ["AST"] * 4,
                "attack": ["PGD", "XShift", "PGD", "XShift"],
                "pred_orig": [1, 1, 0, 0],
                "pred_adv": [1, 1, 0, 0],
                "prob_orig": [0.9, 0.9, 0.1, 0.1],
                "prob_adv": [0.85, 0.88, 0.12, 0.09],
                "cos_sim": [0.1, 0.2, 0.3, 0.4],
                "top10_overlap": [0.05, 0.1, 0.15, 0.2],
            }
        )
        df = select_mod.add_missing_case_metrics(df, 0.5)
        summary = select_mod.make_sample_summary(df, "stem")
        assert len(summary) == 2
        row_s1 = summary[summary["stem"] == "s1"].iloc[0]
        assert row_s1["n_cases"] == 2


class TestAddReasonTags:
    """Tests for add_reason_tags()."""

    def _base_row(self, **kwargs) -> pd.Series:
        defaults = {
            "pred_preserved_rate": 0.95,
            "attribution_change_mean": 0.85,
            "cos_sim_median": 0.10,
            "top10_overlap_median": 0.05,
            "quality_score_mean": 0.85,
            "margin_adv_mean": 0.30,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_easy_tag_prepended(self):
        row = self._base_row()
        result = select_mod.add_reason_tags(row, "easy")
        assert result.startswith("easy_by_high_afs_stable")

    def test_hard_tag_prepended(self):
        row = self._base_row()
        result = select_mod.add_reason_tags(row, "hard")
        assert result.startswith("hard_by_attr_stability")

    def test_prediction_preserved_tag(self):
        row = self._base_row(pred_preserved_rate=0.97)
        result = select_mod.add_reason_tags(row, "easy")
        assert "prediction_preserved" in result

    def test_prediction_often_flips_tag(self):
        row = self._base_row(pred_preserved_rate=0.50)
        result = select_mod.add_reason_tags(row, "hard")
        assert "prediction_often_flips" in result

    def test_strong_attr_change_tag(self):
        row = self._base_row(attribution_change_mean=0.90)
        result = select_mod.add_reason_tags(row, "easy")
        assert "strong_attr_change" in result

    def test_attribution_stable_tag(self):
        row = self._base_row(attribution_change_mean=0.20)
        result = select_mod.add_reason_tags(row, "hard")
        assert "attribution_stable" in result

    def test_low_cos_sim_tag(self):
        row = self._base_row(cos_sim_median=0.10)
        result = select_mod.add_reason_tags(row, "easy")
        assert "low_cos_sim" in result

    def test_high_cos_sim_tag(self):
        row = self._base_row(cos_sim_median=0.80)
        result = select_mod.add_reason_tags(row, "hard")
        assert "high_cos_sim" in result

    def test_low_top10_overlap_tag(self):
        row = self._base_row(top10_overlap_median=0.05)
        result = select_mod.add_reason_tags(row, "easy")
        assert "low_top10_overlap" in result

    def test_high_top10_overlap_tag(self):
        row = self._base_row(top10_overlap_median=0.70)
        result = select_mod.add_reason_tags(row, "hard")
        assert "high_top10_overlap" in result

    def test_high_quality_tag(self):
        row = self._base_row(quality_score_mean=0.90)
        result = select_mod.add_reason_tags(row, "easy")
        assert "high_quality" in result

    def test_low_quality_tag(self):
        row = self._base_row(quality_score_mean=0.40)
        result = select_mod.add_reason_tags(row, "hard")
        assert "low_quality" in result

    def test_safe_margin_tag(self):
        row = self._base_row(margin_adv_mean=0.30)
        result = select_mod.add_reason_tags(row, "easy")
        assert "safe_margin" in result

    def test_near_decision_boundary_tag(self):
        row = self._base_row(margin_adv_mean=0.05)
        result = select_mod.add_reason_tags(row, "hard")
        assert "near_decision_boundary" in result

    def test_tags_are_semicolon_separated(self):
        row = self._base_row()
        result = select_mod.add_reason_tags(row, "easy")
        parts = result.split(";")
        assert len(parts) >= 1

    def test_boundary_pred_preserved_rate_exactly_0_95(self):
        row = self._base_row(pred_preserved_rate=0.95)
        result = select_mod.add_reason_tags(row, "easy")
        assert "prediction_preserved" in result

    def test_boundary_pred_preserved_rate_exactly_0_75(self):
        # 0.75 is not < 0.75 → no "often_flips" tag
        row = self._base_row(pred_preserved_rate=0.75)
        result = select_mod.add_reason_tags(row, "hard")
        assert "prediction_often_flips" not in result
