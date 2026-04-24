# Picture-Aliver Testing Workflow

Complete testing pipeline to verify video generation for different image types and motion styles.

## Quick Start

Run all tests:
```bash
python -m tests.testing_workflow
```

## Test Cases

| Test | Image Type | Motion | Description |
|------|------------|--------|-------------|
| portrait_subtle | Portrait | Subtle | Gentle breathing, minimal motion |
| landscape_pan | Landscape | Cinematic | Camera pan, wind in trees |
| object_rotation | Object | Auto | Slow rotation, floating |

## Verification Criteria

Each test verifies:
1. **Video Generated** - Output file exists
2. **Motion Present** - Optical flow > 0.5 intensity
3. **No Flickering** - Brightness variance < 15
4. **Structural Stability** - SSIM > 0.7 between frames

## Logging Output

```
14:32:01 | INFO     | [IMAGE_LOAD] Starting...
14:32:01 | INFO     | [IMAGE_LOAD] Loaded: tests/images/portrait.png
14:32:02 | INFO     | [VIDEO_DIFF] Initializing pipeline...
14:32:05 | INFO     | [VIDEO_DIFF] Starting generation...
14:32:45 | INFO     | [VERIFICATION] Analyzing generated video...
14:32:46 | INFO     | [Metrics] Flickering: 5.23 (OK: True)
14:32:46 | INFO     | [Metrics] Motion: 2.15 intensity, detected: True
14:32:46 | INFO     | [Metrics] Stability: 0.892 (OK: True)
14:32:46 | INFO     | [PASS] portrait_subtle_motion (45.2s)
```

## Failure Diagnosis

If a test fails, logs show:
- **Failure module**: Which component failed
- **Suggested fix**: How to resolve the issue

Example failure output:
```
[FAIL] landscape_pan
        Module: Motion Generation
        Fix: Increase motion_strength or use a more descriptive prompt.
```

## Manual Testing

Generate a single test:
```python
from tests.testing_workflow import TestingWorkflow, TestConfig

config = TestConfig(
    name="custom_test",
    description="My custom test",
    image_path="my_image.png",
    prompt="slow zoom effect",
    motion_mode="zoom",
    expected_motion="zoom",
)

workflow = TestingWorkflow(config)
result = workflow.run()
```

## Quality Metrics

The testing workflow calculates:
- **Flickering Score**: Brightness variance across frames (lower = better)
- **Motion Intensity**: Average optical flow magnitude
- **Stability Score**: SSIM between consecutive frames (higher = better)