# Project Structure Summary

This document summarizes the purpose of the main directories in this project.

*   **`.github/`**: This directory contains GitHub-specific files, such as:
    *   `ISSUE_TEMPLATE/`: Templates for creating new issues.
    *   `PULL_REQUEST_TEMPLATE.md`: A template for pull requests.

*   **`docs/`**: This directory holds project documentation files. Examples include:
    *   `code-of-conduct.md`: Guidelines for community interaction.
    *   `contributing.md`: Information on how to contribute to the project.

*   **`examples/`**: This directory provides example usage of the project's functionalities. It contains Jupyter notebooks demonstrating various features, such as:
    *   `grpo_demo.ipynb`
    *   `logit_distillation.ipynb`
    *   `qlora_demo.ipynb`
    *   `qwen3_example.ipynb`

*   **`scripts/`**: This directory contains utility scripts. An example is:
    *   `setup_notebook_tpu_single_host.sh`: A shell script for setting up a notebook environment on a TPU single host.

*   **`tests/`**: This directory houses all the test code for the project. It has a hierarchical structure mirroring the `tunix` directory, with specific tests for different modules:
    *   `distillation/`: Tests for distillation functionalities.
    *   `generate/`: Tests for text generation functionalities.
    *   `rl/`: Tests for reinforcement learning components.
    *   `sft/`: Tests for supervised fine-tuning components.
    *   `test_common.py`: Common test utilities.

*   **`tunix/`**: This is the core directory containing the main source code of the project. It is organized into several submodules:
    *   `distillation/`: Source code related to model distillation.
    *   `generate/`: Source code for text generation capabilities.
    *   `models/`: Definitions of different models (e.g., Gemma, Llama3, Qwen3).
    *   `rl/`: Source code for reinforcement learning algorithms and trainers (e.g., DPO, GRPO).
    *   `sft/`: Source code for supervised fine-tuning processes.

Other important files at the root level:

*   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.
*   **`LICENSE`**: Contains the licensing information for the project.
*   **`README.md`**: Provides an overview of the project, setup instructions, and usage examples.
*   **`pyproject.toml`**: Specifies project build requirements and dependencies.

## `tunix` Library Details

The `tunix` library is the core of this project and is composed of the following submodules:

*   **`tunix/distillation/`**: This submodule focuses on knowledge distillation techniques.
    *   `distillation_trainer.py`: Contains the main trainer class for distillation.
    *   `feature_extraction/`: Includes modules for extracting features from models, such as:
        *   `pooling.py`: Implements various pooling strategies.
        *   `projection.py`: Implements feature projection layers.
        *   `sowed_module.py`: Likely related to "SOWED" or similar feature extraction methods.
    *   `strategies/`: Defines different distillation strategies, including:
        *   `attention.py`: Strategies based on attention mechanisms.
        *   `base_strategy.py`: A base class for distillation strategies.
        *   `feature_pooling.py`: Strategies utilizing feature pooling.
        *   `feature_projection.py`: Strategies utilizing feature projection.
        *   `logit.py`: Strategies based on logit matching.

*   **`tunix/generate/`**: This submodule is responsible for text generation.
    *   `beam_search.py`: Implements beam search decoding for generation.
    *   `sampler.py`: Implements various sampling methods for generation (e.g., nucleus sampling, temperature scaling).
    *   `tokenizer_adapter.py`: Adapts tokenizers for use in the generation process.
    *   `utils.py`: Utility functions for text generation.

*   **`tunix/models/`**: This submodule contains implementations and configurations for various transformer models.
    *   `gemma/`: Code specific to Gemma models, including data handling, model definition, parameters, and sampler.
    *   `gemma3/`: Code specific to Gemma3 models, including model definition and parameters.
    *   `llama3/`: Code specific to Llama3 models, including model definition and parameters.
    *   `qwen3/`: Code specific to Qwen3 models, including model definition and parameters.

*   **`tunix/rl/`**: This submodule implements reinforcement learning techniques for training models.
    *   `common.py`: Common utilities and base classes for RL.
    *   `dpo/`: Implements Direct Preference Optimization (DPO).
        *   `dpo_trainer.py`: The trainer class for DPO.
    *   `grpo/`: Implements a GRPO (Generalized Reinforcement Learning with Preference Optimization) algorithm.
        *   `grpo_helpers.py`: Helper functions for GRPO.
        *   `grpo_trainer.py`: The trainer class for GRPO.

*   **`tunix/sft/`**: This submodule is dedicated to supervised fine-tuning (SFT) of models.
    *   `checkpoint_manager.py`: Manages saving and loading model checkpoints.
    *   `inflight_throttler.py`: Likely used to manage and throttle requests or operations during training.
    *   `metrics_logger.py`: Logs metrics during the fine-tuning process.
    *   `peft_trainer.py`: A trainer class that likely incorporates Parameter-Efficient Fine-Tuning (PEFT) techniques.
    *   `profiler.py`: Tools for profiling training performance.
    *   `progress_bar.py`: Implements custom progress bars for training.
    *   `system_metrics_calculator.py`: Calculates and logs system metrics (e.g., CPU/GPU utilization).

## Testing Strategy

The project employs a comprehensive testing strategy, centered around the `tests/` directory. This directory is crucial for ensuring the reliability and correctness of the `tunix` library.

*   **Purpose of `tests/`**: The `tests/` directory contains all automated tests for the project. These tests are designed to verify the functionality of individual components (unit tests) and the interactions between them (integration tests). The goal is to catch bugs and regressions early in the development process.

*   **Relation to `tunix/`**: The structure of the `tests/` directory directly mirrors the structure of the `tunix/` library. For each submodule and often each file in `tunix/`, there is a corresponding test file or directory within `tests/`. For example:
    *   `tunix/distillation/distillation_trainer.py` is tested by `tests/distillation/distillation_trainer_test.py`.
    *   `tunix/rl/dpo/dpo_trainer.py` is tested by `tests/rl/dpo/dpo_trainer_test.py`.
    *   And so on for other components like `generate`, `sft`, and their sub-parts.

    This parallel structure makes it straightforward for developers to:
    *   Locate the tests relevant to the code they are working on.
    *   Understand how specific modules are intended to be used.
    *   Add new tests when new functionality is implemented or existing code is modified.

*   **`test_common.py`**: This file within `tests/` likely contains common utilities, helper functions, fixtures, or base test classes used across multiple test files to avoid code duplication and streamline the testing process.

By maintaining a well-organized and parallel test suite, the project aims to ensure high code quality and facilitate ongoing development and maintenance.

## Examples and Scripts

The `examples/` and `scripts/` directories serve to demonstrate the project's capabilities and assist with development and operational tasks.

*   **`examples/`**: This directory contains Jupyter notebooks that showcase how to use various features of the `tunix` library. These examples provide practical demonstrations and can serve as a starting point for users to understand and integrate the library into their own workflows.
    *   `grpo_demo.ipynb`: Demonstrates the GRPO algorithm.
    *   `logit_distillation.ipynb`: Shows how to perform logit-based distillation.
    *   `qlora_demo.ipynb`: Illustrates the use of QLoRA for efficient fine-tuning.
    *   `qwen3_example.ipynb`: Provides an example of using a Qwen3 model.

*   **`scripts/`**: This directory holds utility scripts for various purposes, such as environment setup, automation of common tasks, or deployment assistance.
    *   `setup_notebook_tpu_single_host.sh`: A shell script designed to configure a Google Cloud TPU environment for use with Jupyter notebooks, facilitating development and experimentation on TPUs.

These directories are valuable resources for both users and developers of the project. Examples help in understanding the practical application of the library, while scripts can automate and simplify common operational or setup tasks.

## Other Important Root-Level Files

Beyond the main directories, several files at the root of the project play crucial roles:

*   **`pyproject.toml`**: This file is a standard configuration file for Python projects (introduced in PEP 518). It is used by build tools like `pip` and `Poetry` to define project metadata, dependencies, and build system requirements. It centralizes the project's build configuration.

*   **`.gitignore`**: This file tells Git which files or directories to ignore in the project. This typically includes compiled code, temporary files, logs, and environment-specific configuration files that should not be committed to the version control repository.

*   **`LICENSE`**: This file contains the legal license under which the project's code is distributed. It specifies the permissions, conditions, and limitations for using, modifying, and distributing the software. It is crucial for users and contributors to understand their rights and obligations.

*   **`README.md`**: (Already mentioned in the initial summary but reiterated here for completeness of root-level files) This is the primary entry point for anyone visiting the project repository. It typically includes a project description, installation instructions, basic usage examples, contribution guidelines, and sometimes a link to more detailed documentation.
