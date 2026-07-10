import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_without_qwen_dependencies():
    import_script = textwrap.dedent(
        """
        import importlib
        import sys
        from importlib.abc import MetaPathFinder

        blocked_dependency_roots = {"transformers", "triton"}

        class RejectOptionalDependencies(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in blocked_dependency_roots:
                    raise AssertionError(
                        f"unexpected optional dependency import: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, RejectOptionalDependencies())

        def import_transformers():
            import transformers

        def import_triton():
            import triton

        def assert_import_is_blocked(importer, dependency, mechanism):
            try:
                importer()
            except AssertionError as exc:
                expected = f"unexpected optional dependency import: {dependency}"
                assert str(exc) == expected
            else:
                raise AssertionError(
                    f"{mechanism} did not block optional dependency: {dependency}"
                )

        assert_import_is_blocked(
            import_transformers,
            "transformers",
            "ordinary import",
        )
        assert_import_is_blocked(import_triton, "triton", "ordinary import")
        assert_import_is_blocked(
            lambda: importlib.import_module("transformers"),
            "transformers",
            "importlib.import_module",
        )
        assert_import_is_blocked(
            lambda: importlib.import_module("triton"),
            "triton",
            "importlib.import_module",
        )

        import research.kmd2_ablation as suite

        assert suite.SUITE_VERSION == "1.0.0"
        public_names = {name for name in vars(suite) if not name.startswith("_")}
        assert public_names == {"SUITE_VERSION"}
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
