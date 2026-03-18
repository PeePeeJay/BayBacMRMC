import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mrmc_baybac.model_checking import main, run_psa

# Mirror the hardcoded constants inside run_psa for case_interaction=False
PRIORS_OPTIONS = [
    "diffuse",
    "weakly informative",
    "informative",
    "frequentist",
]
GAMMA_SIM = np.arange(0.1, 0.6, 0.1)
N_READERS_SIM = [2, 4, 6]


def _fake_summary(*args, **kwargs):
    """Minimal az.summary stand-in returning mu_a and mu_b rows."""
    return pd.DataFrame(
        {"mean": [0.5, 0.2]}, index=["mu_a", "mu_b"]
    )


class TestRunPsaCaseA:
    """Tests for run_psa() with case_interaction=False."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.output_dir = str(
            tmp_path / "psa_results" / "estimates"
        )

    def _run(self, **kwargs):
        defaults = dict(
            num_sims=1,
            case_interaction=False,
            n_readers_sim=N_READERS_SIM,
            n_cases_neg=5,
            n_cases_pos=5,
            n_draws=10,
            output_dir=self.output_dir,
        )
        defaults.update(kwargs)
        with (
            patch(
                "mrmc_baybac.model_checking.run_model_inference",
                return_value=[MagicMock(), MagicMock()],
            ),
            patch(
                "mrmc_baybac.model_checking.az.summary",
                side_effect=_fake_summary,
            ),
        ):
            run_psa(**defaults)

    def test_output_dir_created(self):
        self._run()
        assert os.path.isdir(self.output_dir)

    def test_file_count(self):
        """Expect num_sims × n_readers × gamma_sim × priors files."""
        self._run()
        files = [
            f
            for f in os.listdir(self.output_dir)
            if f.endswith(".pkl")
        ]
        expected = (
            1
            * len(N_READERS_SIM)
            * len(GAMMA_SIM)
            * len(PRIORS_OPTIONS)
        )
        assert len(files) == expected

    def test_filename_format(self):
        """All files follow estimate_{sim}.{prior_idx}.{size_idx}.{gamma_idx}.pkl."""
        self._run()
        for fname in os.listdir(self.output_dir):
            if not fname.endswith(".pkl"):
                continue
            assert fname.startswith("estimate_"), fname
            stem = fname.replace("estimate_", "").replace(
                ".pkl", ""
            )
            parts = stem.split(".")
            assert len(parts) == 4, fname
            assert all(p.isdigit() for p in parts), fname

    def test_all_expected_files_present(self):
        """Every (sim, prior, size, gamma) combination is saved."""
        self._run()
        for prior_idx in range(len(PRIORS_OPTIONS)):
            for size_idx in range(len(N_READERS_SIM)):
                for gamma_idx in range(len(GAMMA_SIM)):
                    fname = f"estimate_0.{prior_idx}.{size_idx}.{gamma_idx}.pkl"
                    path = os.path.join(
                        self.output_dir, fname
                    )
                    assert os.path.isfile(
                        path
                    ), f"Missing: {fname}"

    def test_pickle_keys(self):
        """Each pickle contains all required estimate keys."""
        self._run()
        required = {
            "mu_a_neg",
            "mu_b_neg",
            "mu_a_pos",
            "mu_b_pos",
            "intercept_freq",
            "slope_freq",
            "true_params",
        }
        for fname in os.listdir(self.output_dir):
            if not fname.endswith(".pkl"):
                continue
            with open(
                os.path.join(self.output_dir, fname), "rb"
            ) as f:
                est = pickle.load(f)
            assert required.issubset(est.keys()), fname

    def test_frequentist_has_finite_intercept_and_slope(
        self,
    ):
        """Frequentist files contain finite intercept_freq and slope_freq."""
        self._run()
        freq_idx = PRIORS_OPTIONS.index("frequentist")
        for size_idx in range(len(N_READERS_SIM)):
            for gamma_idx in range(len(GAMMA_SIM)):
                fname = f"estimate_0.{freq_idx}.{size_idx}.{gamma_idx}.pkl"
                with open(
                    os.path.join(self.output_dir, fname),
                    "rb",
                ) as f:
                    est = pickle.load(f)
                assert np.isfinite(
                    est["intercept_freq"]
                ), fname
                assert np.isfinite(est["slope_freq"]), fname

    def test_bayesian_priors_have_nan_freq_fields(self):
        """Non-frequentist files store NaN for intercept_freq and slope_freq."""
        self._run()
        for prior_idx, prior in enumerate(PRIORS_OPTIONS):
            if prior == "frequentist":
                continue
            for size_idx in range(len(N_READERS_SIM)):
                for gamma_idx in range(len(GAMMA_SIM)):
                    fname = f"estimate_0.{prior_idx}.{size_idx}.{gamma_idx}.pkl"
                    with open(
                        os.path.join(
                            self.output_dir, fname
                        ),
                        "rb",
                    ) as f:
                        est = pickle.load(f)
                    assert np.isnan(
                        est["intercept_freq"]
                    ), fname
                    assert np.isnan(
                        est["slope_freq"]
                    ), fname

    def test_true_params_contains_gamma(self):
        """true_params stored in each pickle contains a 'gamma' key."""
        self._run()
        for fname in os.listdir(self.output_dir):
            if not fname.endswith(".pkl"):
                continue
            with open(
                os.path.join(self.output_dir, fname), "rb"
            ) as f:
                est = pickle.load(f)
            assert "gamma" in est["true_params"], fname

    def test_true_params_gamma_matches_sweep(self):
        """gamma in true_params corresponds to gamma_sim[gamma_idx]."""
        self._run()
        for gamma_idx, gamma_val in enumerate(GAMMA_SIM):
            for size_idx in range(len(N_READERS_SIM)):
                for prior_idx in range(len(PRIORS_OPTIONS)):
                    fname = f"estimate_0.{prior_idx}.{size_idx}.{gamma_idx}.pkl"
                    with open(
                        os.path.join(
                            self.output_dir, fname
                        ),
                        "rb",
                    ) as f:
                        est = pickle.load(f)
                    assert np.isclose(
                        est["true_params"]["gamma"],
                        gamma_val,
                    ), f"{fname}: expected gamma={gamma_val}, got {est['true_params']['gamma']}"

    def test_run_model_inference_called_for_bayesian_priors(
        self,
    ):
        """run_model_inference is called once per (size, gamma, Bayesian prior) combination."""
        n_bayesian = sum(
            1 for p in PRIORS_OPTIONS if p != "frequentist"
        )
        expected_calls = (
            1
            * len(N_READERS_SIM)
            * len(GAMMA_SIM)
            * n_bayesian
        )
        with (
            patch(
                "mrmc_baybac.model_checking.run_model_inference",
                return_value=[MagicMock(), MagicMock()],
            ) as mock_infer,
            patch(
                "mrmc_baybac.model_checking.az.summary",
                side_effect=_fake_summary,
            ),
        ):
            run_psa(
                num_sims=1,
                case_interaction=False,
                n_readers_sim=N_READERS_SIM,
                n_cases_neg=5,
                n_cases_pos=5,
                n_draws=10,
                output_dir=self.output_dir,
            )
        assert mock_infer.call_count == expected_calls

    def test_run_model_inference_called_with_correct_priors(
        self,
    ):
        """run_model_inference is called with each of the three Bayesian prior strings."""
        used_priors = set()
        original_patch = MagicMock(
            return_value=[MagicMock(), MagicMock()]
        )

        def _capture_prior(data, prior, **kwargs):
            used_priors.add(prior)
            return [MagicMock(), MagicMock()]

        with (
            patch(
                "mrmc_baybac.model_checking.run_model_inference",
                side_effect=_capture_prior,
            ),
            patch(
                "mrmc_baybac.model_checking.az.summary",
                side_effect=_fake_summary,
            ),
        ):
            run_psa(
                num_sims=1,
                case_interaction=False,
                n_readers_sim=N_READERS_SIM,
                n_cases_neg=5,
                n_cases_pos=5,
                n_draws=10,
                output_dir=self.output_dir,
            )
        expected_bayesian = {
            "diffuse",
            "weakly informative",
            "informative",
        }
        assert expected_bayesian == used_priors


class TestRunPsaCLI:
    """Tests for the CLI argument parsing in model_checking.main()."""

    def _run_cli(self, argv_extras=None):
        """Call main() with explicit argv and a mocked run_psa.

        Returns the mock so callers can inspect how run_psa was called.
        """
        with patch(
            "mrmc_baybac.model_checking.run_psa"
        ) as mock_psa:
            main(argv=argv_extras or [])
        return mock_psa

    def test_default_output_dir(self):
        """Default --output-dir is psa_results/estimates."""
        mock_psa = self._run_cli()
        assert (
            mock_psa.call_args.kwargs["output_dir"]
            == "psa_results/estimates"
        )

    def test_default_case_interaction_is_false(self):
        """case_interaction defaults to False when --case-interaction is absent."""
        mock_psa = self._run_cli()
        assert (
            mock_psa.call_args.kwargs["case_interaction"]
            is False
        )

    def test_default_mu_baseline_is_none(self):
        """mu_baseline defaults to None when --mu-baseline is absent."""
        mock_psa = self._run_cli()
        assert (
            mock_psa.call_args.kwargs["mu_baseline"] is None
        )

    def test_default_effect_size_is_none(self):
        """effect_size defaults to None when --effect-size is absent."""
        mock_psa = self._run_cli()
        assert (
            mock_psa.call_args.kwargs["effect_size"] is None
        )

    def test_default_n_readers_is_none(self):
        """n_readers_sim defaults to None when --n-readers is absent."""
        mock_psa = self._run_cli()
        assert (
            mock_psa.call_args.kwargs["n_readers_sim"]
            is None
        )

    def test_n_readers_arg(self):
        """--n-readers passes the list of ints to run_psa."""
        mock_psa = self._run_cli(
            ["--n-readers", "2", "4", "6"]
        )
        assert mock_psa.call_args.kwargs[
            "n_readers_sim"
        ] == [2, 4, 6]

    def test_output_dir_arg(self, tmp_path):
        """--output-dir is forwarded correctly."""
        mock_psa = self._run_cli(
            ["--output-dir", str(tmp_path)]
        )
        assert mock_psa.call_args.kwargs[
            "output_dir"
        ] == str(tmp_path)

    def test_case_interaction_flag(self):
        """--case-interaction sets case_interaction=True."""
        mock_psa = self._run_cli(["--case-interaction"])
        assert (
            mock_psa.call_args.kwargs["case_interaction"]
            is True
        )

    def test_mu_baseline_arg(self):
        """--mu-baseline is parsed as float and forwarded."""
        mock_psa = self._run_cli(["--mu-baseline", "0.8"])
        assert mock_psa.call_args.kwargs[
            "mu_baseline"
        ] == pytest.approx(0.8)

    def test_effect_size_arg(self):
        """--effect-size is parsed as float and forwarded."""
        mock_psa = self._run_cli(["--effect-size", "0.5"])
        assert mock_psa.call_args.kwargs[
            "effect_size"
        ] == pytest.approx(0.5)

    def test_all_args_combined(self, tmp_path):
        """All CLI args are forwarded in a single combined invocation."""
        mock_psa = self._run_cli(
            [
                "--n-readers",
                "2",
                "4",
                "--output-dir",
                "./tests/psa_output",
                "--case-interaction",
                "--mu-baseline",
                "0.7",
                "--effect-size",
                "0.2",
            ]
        )
        kwargs = mock_psa.call_args.kwargs
        assert kwargs["n_readers_sim"] == [2, 4]
        assert kwargs["output_dir"] == "./tests/psa_output"
        assert kwargs["case_interaction"] is True
        assert kwargs["mu_baseline"] == pytest.approx(0.7)
        assert kwargs["effect_size"] == pytest.approx(0.2)


@pytest.mark.slow
class TestRunPsaIntegration:
    """End-to-end test: real MCMC inference, real files written to disk.

    Run with:  pytest -m slow
    Skip with: pytest -m "not slow"  (default for fast CI)
    """

    GAMMA_SIM = np.arange(0.1, 0.6, 0.1)
    PRIORS_OPTIONS = [
        "diffuse",
        "weakly informative",
        "informative",
        "frequentist",
    ]

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.output_dir = str(tmp_path / "estimates")

    def _run(self, extra_argv=None):
        argv = [
            "--num-sims",
            "1",
            "--n-readers",
            "2",
            "--n-cases-neg",
            "5",
            "--n-cases-pos",
            "5",
            "--n-draws",
            "50",
            "--output-dir",
            self.output_dir,
        ] + (extra_argv or [])
        main(argv=argv)

    def test_output_dir_is_created(self):
        self._run()
        assert os.path.isdir(self.output_dir)

    def test_expected_file_count(self):
        """1 sim × 1 reader size × 5 gammas × 4 priors = 20 files."""
        self._run()
        files = [
            f
            for f in os.listdir(self.output_dir)
            if f.endswith(".pkl")
        ]
        assert len(files) == 1 * 1 * len(
            self.GAMMA_SIM
        ) * len(self.PRIORS_OPTIONS)

    def test_all_expected_files_present(self):
        self._run()
        for prior_idx in range(len(self.PRIORS_OPTIONS)):
            for gamma_idx in range(len(self.GAMMA_SIM)):
                fname = f"estimate_0.{prior_idx}.0.{gamma_idx}.pkl"
                assert os.path.isfile(
                    os.path.join(self.output_dir, fname)
                ), fname

    def test_bayesian_estimates_are_finite(self):
        """Bayesian priors produce finite posterior mean estimates."""
        self._run()
        freq_idx = self.PRIORS_OPTIONS.index("frequentist")
        for prior_idx in range(len(self.PRIORS_OPTIONS)):
            if prior_idx == freq_idx:
                continue
            for gamma_idx in range(len(self.GAMMA_SIM)):
                fname = f"estimate_0.{prior_idx}.0.{gamma_idx}.pkl"
                with open(
                    os.path.join(self.output_dir, fname),
                    "rb",
                ) as f:
                    est = pickle.load(f)
                for key in (
                    "mu_a_neg",
                    "mu_b_neg",
                    "mu_a_pos",
                    "mu_b_pos",
                ):
                    assert np.isfinite(
                        est[key]
                    ), f"{fname}: {key} is not finite"

    def test_frequentist_estimates_are_finite(self):
        """Frequentist OLS produces finite intercept and slope."""
        self._run()
        freq_idx = self.PRIORS_OPTIONS.index("frequentist")
        for gamma_idx in range(len(self.GAMMA_SIM)):
            fname = (
                f"estimate_0.{freq_idx}.0.{gamma_idx}.pkl"
            )
            with open(
                os.path.join(self.output_dir, fname), "rb"
            ) as f:
                est = pickle.load(f)
            assert np.isfinite(est["intercept_freq"]), fname
            assert np.isfinite(est["slope_freq"]), fname

    def test_mu_baseline_and_effect_size_stored(self):
        """CLI --mu-baseline and --effect-size are preserved in true_params."""
        self._run(
            [
                "--mu-baseline",
                "0.75",
                "--effect-size",
                "0.15",
            ]
        )
        fname = "estimate_0.0.0.0.pkl"
        with open(
            os.path.join(self.output_dir, fname), "rb"
        ) as f:
            est = pickle.load(f)
        tp = est["true_params"]
        assert np.isclose(tp["mu_baseline"], 0.75)
        assert np.isclose(tp["effect_size"], 0.15)
