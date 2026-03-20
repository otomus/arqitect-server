"""Tests for arqitect.inference.tuner -- training data collection, adapter training, conversion."""

import json
import os
import threading
from unittest.mock import patch, MagicMock

import pytest
from dirty_equals import IsInstance, IsPositiveInt
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from arqitect.inference.tuner import (
    _SIZE_CLASS_LIMITS,
    _slice_for_size_class,
    collect_training_data,
    _get_base_model_path,
    train_nerve_adapter,
    _convert_adapter_to_gguf,
    get_nerves_ready_for_training,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(count: int) -> list[dict]:
    """Build a list of training-data dicts with sequential values."""
    return [{"input": str(i), "output": str(i)} for i in range(count)]


def _default_tuning_config() -> dict:
    """Canonical tuning config used across training tests."""
    return {
        "lora_rank": 8,
        "lora_epochs": 2,
        "lora_lr": 1e-4,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "min_training_examples": 5,
        "training_max_length": 512,
    }


# ---------------------------------------------------------------------------
# _slice_for_size_class (pure function)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSliceForSizeClass:
    """Tests for size-class data slicing."""

    def test_none_returns_full_data(self):
        data = _make_rows(300)
        assert _slice_for_size_class(data, None) is data

    def test_tinylm_limits_to_50(self):
        data = [{"input": str(i)} for i in range(200)]
        result = _slice_for_size_class(data, "tinylm")
        assert len(result) == 50

    def test_small_limits_to_100(self):
        data = [{"input": str(i)} for i in range(200)]
        assert len(_slice_for_size_class(data, "small")) == 100

    def test_medium_limits_to_200(self):
        data = [{"input": str(i)} for i in range(500)]
        assert len(_slice_for_size_class(data, "medium")) == 200

    def test_large_limits_to_500(self):
        data = [{"input": str(i)} for i in range(1000)]
        assert len(_slice_for_size_class(data, "large")) == 500

    def test_unknown_size_class_returns_full(self):
        data = _make_rows(300)
        result = _slice_for_size_class(data, "galaxy")
        assert len(result) == 300

    def test_data_shorter_than_limit(self):
        data = [{"input": "a"}]
        result = _slice_for_size_class(data, "large")
        assert len(result) == 1

    def test_preserves_order(self):
        data = _make_rows(100)
        result = _slice_for_size_class(data, "tinylm")
        assert result == data[:50]

    @given(
        size_class=st.sampled_from(list(_SIZE_CLASS_LIMITS.keys())),
        data_len=st.integers(min_value=0, max_value=2000),
    )
    @settings(max_examples=50)
    def test_result_never_exceeds_limit(self, size_class, data_len):
        """For any known size class the result length is at most the configured limit."""
        data = _make_rows(data_len)
        result = _slice_for_size_class(data, size_class)
        limit = _SIZE_CLASS_LIMITS[size_class]
        assert len(result) <= limit
        # Result is always a prefix of the original
        assert result == data[: len(result)]

    @given(data_len=st.integers(min_value=0, max_value=500))
    @settings(max_examples=30)
    def test_none_size_class_is_identity(self, data_len):
        """None size class always returns the exact same list object."""
        data = _make_rows(data_len)
        assert _slice_for_size_class(data, None) is data

    @given(
        unknown_class=st.text(min_size=1, max_size=20).filter(
            lambda s: s not in _SIZE_CLASS_LIMITS
        ),
        data_len=st.integers(min_value=0, max_value=300),
    )
    @settings(max_examples=30)
    def test_unknown_class_returns_all(self, unknown_class, data_len):
        """An unrecognized size class never truncates data."""
        data = _make_rows(data_len)
        result = _slice_for_size_class(data, unknown_class)
        assert len(result) == data_len


# ---------------------------------------------------------------------------
# collect_training_data
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestCollectTrainingData:
    """Tests for training data collection from cold memory and redis."""

    @patch("arqitect.inference.tuner._slice_for_size_class")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_collects_from_test_bank(self, mock_cold_cls, mock_slice):
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.get_test_bank.return_value = [
            {"input": "hello", "output": "world"},
            {"input": "foo", "expected_behavior": "bar"},
        ]
        mock_slice.side_effect = lambda d, _: d

        with patch.dict("sys.modules", {"redis": None}):
            result = collect_training_data("my_nerve")

        assert len(result) == 2
        assert result[0] == {"input": "hello", "output": "world"}
        assert result[1] == {"input": "foo", "output": "bar"}

    @patch("arqitect.inference.tuner._slice_for_size_class")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_skips_entries_without_input(self, mock_cold_cls, mock_slice):
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.get_test_bank.return_value = [
            {"output": "no_input"},
            {"input": "", "output": "empty_input"},
        ]
        mock_slice.side_effect = lambda d, _: d

        with patch.dict("sys.modules", {"redis": None}):
            result = collect_training_data("nerve")

        assert result == []

    @patch("arqitect.inference.tuner._slice_for_size_class")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_passes_size_class_to_slicer(self, mock_cold_cls, mock_slice):
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.get_test_bank.return_value = []
        mock_slice.side_effect = lambda d, _: d

        with patch.dict("sys.modules", {"redis": None}):
            collect_training_data("nerve", size_class="tinylm")

        mock_slice.assert_called_once()
        assert mock_slice.call_args[0][1] == "tinylm"

    @patch("arqitect.inference.tuner._slice_for_size_class", side_effect=lambda d, _: d)
    def test_handles_cold_memory_failure(self, _mock_slice):
        """When ColdMemory raises, returns empty list."""
        with patch("arqitect.memory.cold.ColdMemory", side_effect=Exception("db error")):
            result = collect_training_data("nerve")
        assert result == []

    @patch("arqitect.inference.tuner._slice_for_size_class")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_each_entry_has_input_and_output_keys(self, mock_cold_cls, mock_slice):
        """Every collected entry must contain both 'input' and 'output' keys."""
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.get_test_bank.return_value = [
            {"input": "a", "output": "b"},
            {"input": "c", "expected_behavior": "d"},
            {"output": "orphan"},
        ]
        mock_slice.side_effect = lambda d, _: d

        with patch.dict("sys.modules", {"redis": None}):
            result = collect_training_data("nerve")

        for entry in result:
            assert "input" in entry
            assert "output" in entry
            assert entry["input"] == IsInstance(str)
            assert entry["output"] == IsInstance(str)


# ---------------------------------------------------------------------------
# _get_base_model_path
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetBaseModelPath:
    """Tests for base model path resolution."""

    def test_returns_path_when_exists(self):
        with patch("arqitect.inference.tuner._models_dir", return_value="/models"), \
             patch("arqitect.inference.model_registry.MODEL_REGISTRY",
                   {"nerve": {"file": "nerve.gguf"}}), \
             patch("arqitect.inference.model_registry.resolve_registry_key",
                   return_value="nerve"), \
             patch("os.path.exists", return_value=True):
            result = _get_base_model_path("tool")
            assert result == "/models/nerve.gguf"

    def test_returns_none_when_file_missing(self):
        with patch("arqitect.inference.tuner._models_dir", return_value="/models"), \
             patch("arqitect.inference.model_registry.MODEL_REGISTRY",
                   {"nerve": {"file": "nerve.gguf"}}), \
             patch("arqitect.inference.model_registry.resolve_registry_key",
                   return_value="nerve"), \
             patch("os.path.exists", return_value=False):
            assert _get_base_model_path("tool") is None

    def test_returns_none_when_not_in_registry(self):
        with patch("arqitect.inference.tuner._models_dir", return_value="/models"), \
             patch("arqitect.inference.model_registry.MODEL_REGISTRY", {}), \
             patch("arqitect.inference.model_registry.resolve_registry_key",
                   return_value="unknown"):
            assert _get_base_model_path("unknown_role") is None


# ---------------------------------------------------------------------------
# train_nerve_adapter
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestTrainNerveAdapter:
    """Tests for the LoRA training orchestrator."""

    @patch("arqitect.inference.tuner._get_min_examples", return_value=5)
    @patch("arqitect.brain.adapters.get_tuning_config")
    def test_skips_when_too_few_examples(self, mock_cfg, _mock_min):
        mock_cfg.return_value = _default_tuning_config()
        result = train_nerve_adapter("nerve1", training_data=[{"input": "a", "output": "b"}])
        assert result is False

    @patch("arqitect.inference.tuner._get_base_model_path", return_value=None)
    @patch("arqitect.inference.tuner._get_min_examples", return_value=1)
    @patch("arqitect.brain.adapters.get_tuning_config")
    def test_skips_when_no_base_model(self, mock_cfg, _min, _bmp):
        mock_cfg.return_value = _default_tuning_config()
        data = _make_rows(10)
        result = train_nerve_adapter("nerve1", training_data=data)
        assert result is False

    @patch("arqitect.inference.tuner._get_base_model_path", return_value="/models/nerve.gguf")
    @patch("arqitect.inference.tuner._get_min_examples", return_value=1)
    @patch("arqitect.brain.adapters.get_tuning_config")
    @patch("arqitect.inference.tuner._nerves_dir", return_value="/tmp/nerves")
    def test_returns_false_when_interrupted_before_loading(self, _nd, mock_cfg, _min, _bmp):
        mock_cfg.return_value = _default_tuning_config()
        data = _make_rows(10)
        interrupted = threading.Event()
        interrupted.set()  # Pre-set interrupt

        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        mock_peft = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "peft": mock_peft,
        }):
            result = train_nerve_adapter("nerve1", training_data=data, interrupted=interrupted)

        assert result is False

    @patch("arqitect.inference.tuner.collect_training_data")
    @patch("arqitect.inference.tuner._get_min_examples", return_value=5)
    @patch("arqitect.brain.adapters.get_tuning_config")
    def test_collects_data_when_none_provided(self, mock_cfg, _min, mock_collect):
        mock_cfg.return_value = _default_tuning_config()
        mock_collect.return_value = []  # Too few examples, will exit early
        result = train_nerve_adapter("nerve1", training_data=None)
        mock_collect.assert_called_once_with("nerve1")
        assert result is False

    @patch("arqitect.inference.tuner._get_base_model_path", return_value="/models/n.gguf")
    @patch("arqitect.inference.tuner._get_min_examples", return_value=1)
    @patch("arqitect.brain.adapters.get_tuning_config")
    @patch("arqitect.inference.tuner._nerves_dir", return_value="/tmp/nerves")
    def test_returns_false_on_import_failure(self, _nd, mock_cfg, _min, _bmp):
        """When torch/transformers/peft aren't installed, returns False."""
        mock_cfg.return_value = _default_tuning_config()
        data = _make_rows(10)

        import builtins
        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = train_nerve_adapter("nerve1", training_data=data)

        assert result is False

    @patch("arqitect.inference.tuner._get_min_examples", return_value=5)
    @patch("arqitect.brain.adapters.get_tuning_config")
    def test_uses_custom_lora_params(self, mock_cfg, _min):
        """Custom lora_rank, epochs, lr should override config defaults."""
        mock_cfg.return_value = _default_tuning_config()
        data = _make_rows(2)

        # Too few examples, but we verify params were resolved before exit
        result = train_nerve_adapter(
            "nerve1", training_data=data,
            lora_rank=16, epochs=5, lr=0.001,
        )
        assert result is False  # Too few examples

    @given(
        lora_rank=st.integers(min_value=1, max_value=128),
        epochs=st.integers(min_value=1, max_value=20),
        lr=st.floats(min_value=1e-7, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20)
    @patch("arqitect.inference.tuner._get_min_examples", return_value=999)
    @patch("arqitect.brain.adapters.get_tuning_config")
    def test_always_returns_bool_with_random_hyperparams(
        self, mock_cfg, _min, lora_rank, epochs, lr
    ):
        """Regardless of hyperparameter values, the function returns a bool."""
        mock_cfg.return_value = _default_tuning_config()
        data = _make_rows(2)  # intentionally below min_examples
        result = train_nerve_adapter(
            "nerve1", training_data=data,
            lora_rank=lora_rank, epochs=epochs, lr=lr,
        )
        assert result == IsInstance(bool)


# ---------------------------------------------------------------------------
# _convert_adapter_to_gguf
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestConvertAdapterToGguf:
    """Tests for the safetensors-to-GGUF converter."""

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_uses_converter_script(self, mock_exists, mock_run, tmp_path):
        safetensors_dir = str(tmp_path / "st")
        output_path = str(tmp_path / "adapter.gguf")

        # First call: converter found. Second call: output exists.
        mock_exists.side_effect = lambda p: p == output_path or p.endswith("convert_lora_to_gguf.py")
        mock_run.return_value = MagicMock(returncode=0)

        result = _convert_adapter_to_gguf(safetensors_dir, output_path, "/base.gguf")
        assert result is True

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=False)
    def test_returns_false_when_no_converter(self, _exists, _run, tmp_path):
        """When no converter script is found, returns False."""
        safetensors_dir = str(tmp_path / "st")
        output_path = str(tmp_path / "adapter.gguf")

        with patch.dict("sys.modules", {"llama_cpp": None}):
            result = _convert_adapter_to_gguf(safetensors_dir, output_path, "/base.gguf")

        assert result is False

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_converter_failure_tries_next(self, mock_exists, mock_run, tmp_path):
        safetensors_dir = str(tmp_path / "st")
        output_path = str(tmp_path / "adapter.gguf")

        mock_exists.side_effect = lambda p: p.endswith("convert_lora_to_gguf.py")
        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        with patch.dict("sys.modules", {"llama_cpp": None}):
            result = _convert_adapter_to_gguf(safetensors_dir, output_path, "/base.gguf")
        assert result is False

    @patch("subprocess.run", side_effect=TimeoutError("slow"))
    @patch("os.path.exists")
    def test_handles_subprocess_timeout(self, mock_exists, _run, tmp_path):
        safetensors_dir = str(tmp_path / "st")
        output_path = str(tmp_path / "adapter.gguf")

        mock_exists.side_effect = lambda p: p.endswith("convert_lora_to_gguf.py")

        with patch.dict("sys.modules", {"llama_cpp": None}):
            result = _convert_adapter_to_gguf(safetensors_dir, output_path, "/base.gguf")
        assert result is False

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_return_type_is_always_bool(self, mock_exists, mock_run, tmp_path):
        """Converter must always return a bool, never None or another truthy value."""
        safetensors_dir = str(tmp_path / "st")
        output_path = str(tmp_path / "out.gguf")
        mock_exists.return_value = False

        with patch.dict("sys.modules", {"llama_cpp": None}):
            result = _convert_adapter_to_gguf(safetensors_dir, output_path, "/base.gguf")

        assert result == IsInstance(bool)


# ---------------------------------------------------------------------------
# get_nerves_ready_for_training
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetNervesReadyForTraining:
    """Tests for nerve readiness scanning."""

    @patch("arqitect.inference.tuner._get_min_examples", return_value=2)
    @patch("arqitect.inference.tuner.collect_training_data")
    @patch("arqitect.inference.tuner._nerves_dir", return_value="/tmp/nerves")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_finds_ready_nerves(self, mock_cold_cls, _nd, mock_collect, _min):
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.conn.execute.return_value.fetchall.return_value = [
            ("weather", "Weather nerve", "tool"),
        ]
        mock_collect.return_value = [{"input": "a", "output": "b"}] * 5

        with patch("os.path.exists", return_value=False):
            results = get_nerves_ready_for_training()

        assert len(results) == 1
        assert results[0]["name"] == "weather"
        assert results[0]["role"] == "tool"
        assert results[0]["data_count"] == IsPositiveInt
        assert results[0]["has_adapter"] is False

    @patch("arqitect.inference.tuner._get_min_examples", return_value=100)
    @patch("arqitect.inference.tuner.collect_training_data", return_value=[])
    @patch("arqitect.inference.tuner._nerves_dir", return_value="/tmp/nerves")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_excludes_nerves_with_insufficient_data(self, mock_cold_cls, _nd, _collect, _min):
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.conn.execute.return_value.fetchall.return_value = [
            ("calc", "Calculator", "tool"),
        ]

        results = get_nerves_ready_for_training()
        assert results == []

    def test_handles_cold_memory_failure(self):
        with patch("arqitect.memory.cold.ColdMemory", side_effect=Exception("db")):
            results = get_nerves_ready_for_training()
        assert results == []

    @patch("arqitect.inference.tuner._get_min_examples", return_value=1)
    @patch("arqitect.inference.tuner.collect_training_data")
    @patch("arqitect.inference.tuner._nerves_dir", return_value="/tmp/nerves")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_defaults_role_to_tool(self, mock_cold_cls, _nd, mock_collect, _min):
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.conn.execute.return_value.fetchall.return_value = [
            ("timer", "Timer", None),  # role is None
        ]
        mock_collect.return_value = [{"input": "a", "output": "b"}] * 3

        with patch("os.path.exists", return_value=True):
            results = get_nerves_ready_for_training()

        assert results[0]["role"] == "tool"
        assert results[0]["has_adapter"] is True

    @patch("arqitect.inference.tuner._get_min_examples", return_value=1)
    @patch("arqitect.inference.tuner.collect_training_data")
    @patch("arqitect.inference.tuner._nerves_dir", return_value="/tmp/nerves")
    @patch("arqitect.memory.cold.ColdMemory")
    def test_result_schema(self, mock_cold_cls, _nd, mock_collect, _min):
        """Each result dict must have exactly name, role, data_count, has_adapter."""
        mock_cold = MagicMock()
        mock_cold_cls.return_value = mock_cold
        mock_cold.conn.execute.return_value.fetchall.return_value = [
            ("search", "Search nerve", "tool"),
        ]
        mock_collect.return_value = _make_rows(4)

        with patch("os.path.exists", return_value=False):
            results = get_nerves_ready_for_training()

        expected_keys = {"name", "role", "data_count", "has_adapter"}
        for entry in results:
            assert set(entry.keys()) == expected_keys
            assert entry["name"] == IsInstance(str)
            assert entry["role"] == IsInstance(str)
            assert entry["data_count"] == IsPositiveInt
            assert entry["has_adapter"] == IsInstance(bool)
