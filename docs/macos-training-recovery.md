# macOS training failure and recovery

GUI training publishes an early recovery checkpoint after about 30 seconds, then
requests another every five minutes while training is running. Requests use the
existing project owner, training step boundary, CPU snapshot and background
project writer. Only one project write can be outstanding. Ordinary metadata
autosave remains lightweight; it cannot restore trained tensors or optimizer state.

`training_get_state.project_snapshot.recoverable_iteration` and
`recoverable_snapshot_uuid` identify the last checkpoint successfully published by
the current trainer. A failed write leaves these values and the previous checkpoint
intact. Captured-but-unpublished progress is reported separately. An iteration of
`-1` means this trainer has not published a checkpoint; a project opened for resume
can still contain its previously saved checkpoint.

If GPU work fails, training preserves the first error and settles pending save
requests. A lost or quarantined device cannot capture another checkpoint or apply
queued GPU parameter changes. A completed CPU snapshot can finish writing even
after GPU failure. Console output distinguishes failure from user cancellation
and successful completion.

The native failure notification works without rendering a Vulkan frame. On macOS
it uses an asynchronous sheet so the main loop can continue serving MCP and
settling project writes. It shows live and saved progress when available, the log
location, and instructions to restart the app and reopen the project. Recovery
uses the existing training-session loader, including optimizer and strategy state.
It cannot reconstruct iterations after the last complete checkpoint. A genuine
driver hang may still require terminating the process during shutdown.

## Shared-device lifetime rules

- External completion markers reserve a strictly increasing tensor timeline
  value, including cleanup markers after admission closes. A callback error is
  retained if marker completion also fails; the terminal device error remains
  visible.
- Physical addressable-buffer destruction holds the tensor recorder registry,
  drains tensor work, then holds the owner's submission mutex while draining its
  other queues and freeing allocations. Owner queue wrappers never acquire the
  tensor recorder lock. External callbacks already inside that boundary reuse it.
- Tensor pool trimming, native rasterizer buffer replacement and external-buffer
  destruction share this retirement contract. Cache reuse does not introduce a
  new all-queue wait. A retirement timeout retains resources with unknown
  completion and closes admission.
- macOS disables MoltenVK command prefilling before instance creation. Residency
  encoding must occur at the coordinated submission boundary, rather than while
  recording commands outside that boundary.

## Validation (2026-09-13)

The macOS application build and targeted Vulkan/progress tests were exercised on
Apple Silicon with the bundled MoltenVK:

| Check | Result |
| --- | --- |
| Vulkan tensor regressions | 11 passed, including gated external completion, callback plus completion failure, and quarantine |
| Progress reporting | 2 passed |
| Project container recovery | 69 passed; optional 24-hour simulation skipped; one benchmark disabled |
| Two-queue retirement probe | A real gated buffer write completed before retirement freed the allocation; retirement waited about 1.03 seconds for a one-second gate |
| Trainer timeout injection | Original deadline error retained; queued parameter update/save did not leave training running, paused, saving, or a request pending |
| Native notification fault injection | Rendering halted after an injected timeout; runtime-state and UI-state MCP queries continued completing with the asynchronous macOS sheet open |
| Headless checkpoint resume | Treehill MCMC resumed iteration 60 through 120 with finite model tensors and small numerical differences from uninterrupted training |
| Shared-device GUI training | Treehill MCMC completed 8,000 iterations with one million Gaussians, concurrent image loading and viewport scale changes; five periodic checkpoints and the final iteration-8,000 checkpoint published |

The 8,000-iteration run used resolution factor 4 and took about 1,279 seconds with
validation disabled. Other builds/tests ran concurrently, so this is a stability
measurement, not a controlled throughput comparison. The first small recovery
checkpoint paused capture for about 8 ms; capture cost grows with model size.
The resumed 120-step model was not bitwise identical, but tensor differences were
below those observed between fresh repeated runs. This was not a full final-quality
evaluation.

The original Metal `Invalid Resource` fault was not reproduced in the fixed run.
The completion-marker defect is independently reproduced and fixed; shared-queue
residency remains a plausible explanation for the original lifetime fault. These
checks do not establish immunity to all driver failures or replace hardware
validation, long-duration runs and storage fault testing on other platforms.
