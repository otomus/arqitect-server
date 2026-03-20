"""Tests for arqitect.inference.engine — singleton, cosine similarity, vision delegation."""

import math
import threading
from unittest.mock import MagicMock, patch

import pytest
from dirty_equals import IsFloat
from hypothesis import given, settings, assume
from hypothesis import strategies as st

import arqitect.inference.engine as engine_mod
from arqitect.inference.engine import cosine_similarity


# ---------------------------------------------------------------------------
# Fixture: reset module-level singleton between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure every test starts with a clean singleton slot."""
    original = engine_mod._ENGINE
    engine_mod._ENGINE = None
    yield
    engine_mod._ENGINE = original


# ---------------------------------------------------------------------------
# cosine_similarity (pure function — no mocks needed)
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests for the cosine_similarity helper."""

    @pytest.mark.timeout(10)
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == IsFloat(approx=1.0)

    @pytest.mark.timeout(10)
    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == IsFloat(approx=0.0)

    @pytest.mark.timeout(10)
    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == IsFloat(approx=-1.0)

    @pytest.mark.timeout(10)
    def test_zero_vector_a(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0

    @pytest.mark.timeout(10)
    def test_zero_vector_b(self):
        assert cosine_similarity([1, 2], [0, 0]) == 0.0

    @pytest.mark.timeout(10)
    def test_both_zero(self):
        assert cosine_similarity([0, 0, 0], [0, 0, 0]) == 0.0

    @pytest.mark.timeout(10)
    def test_non_unit_vectors(self):
        result = cosine_similarity([3, 4], [6, 8])
        assert result == IsFloat(approx=1.0)

    @pytest.mark.timeout(10)
    def test_different_magnitude_same_direction(self):
        result = cosine_similarity([1, 1], [100, 100])
        assert result == IsFloat(approx=1.0)

    @pytest.mark.timeout(10)
    def test_partial_overlap(self):
        result = cosine_similarity([1, 1, 0], [1, 0, 0])
        expected = 1 / (2 ** 0.5)
        assert result == IsFloat(approx=expected)

    @pytest.mark.timeout(10)
    @given(
        a=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=50),
        b=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=50),
    )
    @settings(max_examples=200)
    def test_result_bounded_between_neg1_and_1(self, a, b):
        """Cosine similarity must always lie in [-1, 1] for non-zero vectors."""
        # Make vectors the same length by truncating to the shorter
        length = min(len(a), len(b))
        a, b = a[:length], b[:length]
        result = cosine_similarity(a, b)
        assert -1.0 - 1e-9 <= result <= 1.0 + 1e-9

    @pytest.mark.timeout(10)
    @given(
        v=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_self_similarity_is_one_or_zero(self, v):
        """A vector's cosine similarity with itself is 1.0 (or 0.0 if zero-vector)."""
        result = cosine_similarity(v, v)
        norm = sum(x * x for x in v) ** 0.5
        if norm == 0:
            assert result == 0.0
        else:
            assert result == IsFloat(approx=1.0, delta=1e-9)

    @pytest.mark.timeout(10)
    @given(
        a=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
        b=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
    )
    @settings(max_examples=200)
    def test_symmetry(self, a, b):
        """cosine_similarity(a, b) == cosine_similarity(b, a)."""
        length = min(len(a), len(b))
        a, b = a[:length], b[:length]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-12)

    @pytest.mark.timeout(10)
    @given(
        v=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=50),
        scalar=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_scale_invariance(self, v, scalar):
        """Scaling a vector should not change cosine similarity with another vector."""
        # Filter vectors whose norm is too small — underflow to zero is an
        # IEEE-754 artefact, not a contract violation worth testing here.
        norm = sum(x * x for x in v) ** 0.5
        assume(norm > 1e-150)
        scaled = [x * scalar for x in v]
        scaled_norm = sum(x * x for x in scaled) ** 0.5
        assume(scaled_norm > 1e-150)
        result = cosine_similarity(v, scaled)
        assert result == IsFloat(approx=1.0, delta=1e-6)


# ---------------------------------------------------------------------------
# GGUFEngine
# ---------------------------------------------------------------------------

class TestGGUFEngine:
    """Tests for the GGUFEngine class."""

    @pytest.mark.timeout(10)
    @patch("arqitect.inference.engine.GGUFProvider.__init__", return_value=None)
    def test_init_delegates_to_provider(self, mock_init):
        eng = engine_mod.GGUFEngine(models_dir="/tmp/models")
        mock_init.assert_called_once_with(models_dir="/tmp/models")

    @pytest.mark.timeout(10)
    @patch("arqitect.inference.engine.GGUFProvider.__init__", return_value=None)
    def test_generate_vision_delegates(self, mock_init):
        """generate_vision calls generate_vision_from_path with same args."""
        eng = engine_mod.GGUFEngine(models_dir="/tmp/m")
        eng.generate_vision_from_path = MagicMock(return_value="a cat")

        result = eng.generate_vision(
            image_path="/img.png",
            base64_data="abc",
            prompt="What?",
            max_tokens=128,
        )

        eng.generate_vision_from_path.assert_called_once_with(
            image_path="/img.png",
            base64_data="abc",
            prompt="What?",
            max_tokens=128,
        )
        assert result == "a cat"

    @pytest.mark.timeout(10)
    @patch("arqitect.inference.engine.GGUFProvider.__init__", return_value=None)
    def test_generate_vision_default_args(self, mock_init):
        eng = engine_mod.GGUFEngine(models_dir="/tmp/m")
        eng.generate_vision_from_path = MagicMock(return_value="desc")

        eng.generate_vision()

        eng.generate_vision_from_path.assert_called_once_with(
            image_path="",
            base64_data="",
            prompt="Describe this image in detail.",
            max_tokens=256,
        )


# ---------------------------------------------------------------------------
# get_engine — singleton pattern
# ---------------------------------------------------------------------------

class TestGetEngine:
    """Tests for the get_engine singleton factory."""

    @pytest.mark.timeout(10)
    @patch("arqitect.inference.engine.GGUFEngine")
    @patch("arqitect.inference.config.check_gguf_ready", return_value=(True, []))
    @patch("arqitect.inference.config.print_status_report")
    @patch("arqitect.inference.config.get_models_dir", return_value="/tmp/m")
    def test_returns_same_instance(self, _gmd, _psr, _cgr, mock_cls):
        mock_cls.return_value = MagicMock()
        mock_cls.return_value.load_from_registry = MagicMock()

        a = engine_mod.get_engine()
        b = engine_mod.get_engine()
        assert a is b
        assert mock_cls.call_count == 1

    @pytest.mark.timeout(10)
    @patch("arqitect.inference.engine.GGUFEngine")
    @patch("arqitect.inference.config.check_gguf_ready", return_value=(False, ["vision (v.gguf) from hf"]))
    @patch("arqitect.inference.config.print_status_report")
    @patch("arqitect.inference.config.print_setup_guide")
    @patch("arqitect.inference.config.get_models_dir", return_value="/tmp/m")
    def test_raises_when_models_missing(self, _gmd, _psg, _psr, _cgr, _cls):
        with pytest.raises(RuntimeError, match="GGUF models missing"):
            engine_mod.get_engine()

    @pytest.mark.timeout(10)
    @patch.dict("os.environ", {"SYNAPSE_LAZY_LOAD": "1"})
    @patch("arqitect.inference.engine.GGUFEngine")
    @patch("arqitect.inference.config.get_models_dir", return_value="/tmp/m")
    def test_lazy_mode_skips_readiness_check(self, _gmd, mock_cls):
        mock_cls.return_value = MagicMock()

        eng = engine_mod.get_engine()
        assert eng is not None
        mock_cls.return_value.load_from_registry.assert_not_called()

    @pytest.mark.timeout(10)
    @patch("arqitect.inference.engine.GGUFEngine")
    @patch("arqitect.inference.config.check_gguf_ready", return_value=(True, []))
    @patch("arqitect.inference.config.print_status_report")
    @patch("arqitect.inference.config.get_models_dir", return_value="/tmp/m")
    def test_calls_load_from_registry_when_not_lazy(self, _gmd, _psr, _cgr, mock_cls):
        instance = MagicMock()
        mock_cls.return_value = instance

        engine_mod.get_engine()
        instance.load_from_registry.assert_called_once()

    @pytest.mark.timeout(10)
    @patch("arqitect.inference.engine.GGUFEngine")
    @patch("arqitect.inference.config.check_gguf_ready", return_value=(True, []))
    @patch("arqitect.inference.config.print_status_report")
    @patch("arqitect.inference.config.get_models_dir", return_value="/models")
    def test_thread_safety(self, _gmd, _psr, _cgr, mock_cls):
        """Multiple threads racing to get_engine should produce one instance."""
        mock_cls.return_value = MagicMock()
        mock_cls.return_value.load_from_registry = MagicMock()

        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(engine_mod.get_engine())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(r) for r in results)) == 1
        assert mock_cls.call_count == 1


# ---------------------------------------------------------------------------
# get_embedder
# ---------------------------------------------------------------------------

class TestGetEmbedder:
    """Tests for the get_embedder convenience function."""

    @pytest.mark.timeout(10)
    def test_returns_embed_method(self):
        mock_engine = MagicMock()
        engine_mod._ENGINE = mock_engine

        embedder = engine_mod.get_embedder()
        assert embedder is mock_engine.embed
