"""Regression test for a real, silent production bug: ``tabpfn_extensions``
(a hard dependency, see pyproject.toml) failed to import at all with a
modern ``setuptools`` installed, because its top-level ``__init__.py``
eagerly imports its ``hpo`` submodule, which pulls in ``hyperopt``, which
still does ``import pkg_resources`` -- a module setuptools has since removed
entirely.

This isn't a rarely-hit corner: ``tabpfn_extensions.many_class.
ManyClassClassifier`` is AutoGluon core's own mechanism for handling
>10-class datasets on MITRA and every REALTABPFN-V2/V2.5/V2.6 model
(``autogluon.tabular.models.{mitra.mitra_model, tabpfnv2.tabpfnv2_5_model}``).
Without the ``setuptools<80`` pin this test guards, that native many-class
support silently hard-crashes (``ImportError`` re-raised, not a clean skip)
the moment any of those models hits a many-class dataset -- confirmed live
by reproducing the exact failure with setuptools 84.0.0 installed.
"""

import importlib
import importlib.util

import pytest

# Deliberately NOT `pytest.importorskip("tabpfn_extensions")`: that helper
# catches ImportError to decide "not installed, skip" -- which would also
# swallow the exact bug this test exists to catch (tabpfn_extensions
# installed but failing to import) and silently skip instead of failing.
# `find_spec` only checks that the package is *findable* on the path,
# without executing its (potentially broken) __init__.py, so a genuine
# "not installed" case still skips cleanly while an installed-but-broken
# case reaches the real import below and fails loudly. tabpfn_extensions is
# only pulled in by the `models` extra -- this repo's CI only ever installs
# `[dev]` (see .github/workflows/ci.yml), matching every other
# tabarena/autogluon-dependent test here (test_wrapped_models.py, etc.).
if importlib.util.find_spec("tabpfn_extensions") is None:
    pytest.skip("tabpfn_extensions not installed (needs the `models` extra)", allow_module_level=True)


def test_tabpfn_extensions_imports_cleanly():
    # A bare `import tabpfn_extensions` alone doesn't reproduce the bug --
    # the failure is specifically in the many_class submodule's import
    # chain (via the package's own eager __init__.py), so import that
    # exact path, matching what autogluon's MitraModel/TabPFNModel do.
    module = importlib.import_module("tabpfn_extensions.many_class")
    assert hasattr(module, "ManyClassClassifier")


def test_pkg_resources_is_importable():
    # The concrete missing piece: modern setuptools removes this module
    # outright, which is what actually breaks the import chain above.
    importlib.import_module("pkg_resources")
