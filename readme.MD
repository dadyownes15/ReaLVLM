

# ReaLVLM – Real‑Time One‑Shot Video Anomaly Detection with Vision–Language Reasoning

ReaLVLM pairs a self‑supervised video encoder with a frozen vision–language model to flag out‑of‑distribution events—drowning, shoplifting, traffic collisions, and more—from normal‑only training data.




## Running Tests

To run the included tests, ensure you have the necessary dependencies installed:

```bash
pip install torch torchvision decord transformers timm accelerate # Add 'hiera' if testing Hiera
```

Navigate to the project root directory (`VLLM_oneshot/`) in your terminal and run the test script directly:

```bash
python tests/run_tests.py
```

This script will execute tests for the implemented encoder types (currently JEPA and Hiera) and print the results.

**Note:** The Hiera test will be skipped automatically if the `hiera` library is not installed.# ReaLVLM
# ReaLVLM
