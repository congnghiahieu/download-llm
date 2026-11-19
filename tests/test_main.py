from __future__ import annotations

import unittest

from tests.test_docker import DockerUtilsTests
from tests.test_execution import MultiExecutionTests
from tests.test_fallback import FallbackTests
from tests.test_files import FilesPartsTests
from tests.test_hf_stdlib import HfStdlibTests
from tests.test_huggingface_parse import HuggingfaceParseTests
from tests.test_phases_pull_push import PhasesPullPushTests
from tests.test_phases_pullhub_restore import PhasesPullhubRestoreTests
from tests.test_pipeline import PipelineTests
from tests.test_state_listing import StateListingTests
from tests.test_transfer import TransferTests
from tests.test_wsl_storage import WslStorageTests

__all__ = [
    "DockerUtilsTests",
    "FallbackTests",
    "FilesPartsTests",
    "HfStdlibTests",
    "HuggingfaceParseTests",
    "MultiExecutionTests",
    "PhasesPullPushTests",
    "PhasesPullhubRestoreTests",
    "PipelineTests",
    "StateListingTests",
    "TransferTests",
    "WslStorageTests",
]


if __name__ == "__main__":
    unittest.main()
