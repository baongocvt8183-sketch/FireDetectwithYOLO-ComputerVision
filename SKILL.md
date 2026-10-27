---
name: yolo-dashboard-engineer
description: Inspect, redesign, and safely implement a fullscreen YOLO computer-vision dashboard with an expandable glassmorphism sidebar while preserving existing trained model weights and inference logic.
---

# Skill: YOLO Dashboard Engineer

## When to use
Use this skill when the task involves an existing YOLO/OpenCV application, a camera or video detection dashboard, redesigning UI panels, connecting a trained model, or improving reliability of the inference/UI integration.

## Operating principles
1. Inspect before editing. Read the repository, README, dependency files, entry points, and existing inference implementation.
2. Preserve existing trained weights and class mapping. The user may call the model file `.bt`; verify the actual extension and configured path instead of assuming it is `.pt`.
3. Do not retrain, replace, download, rename, overwrite, or delete model files unless explicitly asked.
4. Prefer minimal, reversible changes. Do not migrate frameworks merely for aesthetic reasons.
5. Separate concerns: UI state, camera lifecycle, inference, media uploads, analytics, and alerts.
6. Verify what can be verified. Never claim a real webcam or GPU test was performed when it was not available.
7. Do not fake production detection data.

## Required workflow

### 1. Reconnaissance
- Inspect project tree and runtime entry points.
- Identify Python/Node versions, dependency manager, UI framework, API/backend, and launch commands.
- Locate model initialization, weights path, class names, camera loops, video/image handling, tracking, and alerts.
- Check current tests, lint, formatting, and build commands.
- Record existing working behavior before changing it.

### 2. Plan
Provide a concise plan listing:
- Existing architecture.
- Relevant files.
- Minimal implementation strategy.
- Risks to model loading, camera resources, and inference.
Ask a question only when a missing detail genuinely blocks safe implementation.

### 3. Implement the UX
Default state:
- Fullscreen or maximized detection viewport.
- Persistent hamburger button at top-left.
- Compact status overlay.
- Optional floating controls at the bottom.

Menu-open state:
- Left drawer slides in above the viewport.
- Navigation entries for Live Detection, Image/Video, History, Analytics, Alerts, YOLO Settings, and System Settings.
- Secondary settings appear in a separate panel or accordion.
- Support Escape and an accessible close control.
- Ensure z-index, focus, and outside-click behavior are correct.

Menu-close state:
- Drawer disappears and the detection viewport uses the available window.
- Do not reset settings or restart the camera/inference pipeline because the menu changed.
- Avoid remounting the media component due to navigation state.

### 4. Integrate real detection
- Reuse existing inference functions and weights.
- Keep long-running inference off the UI thread when appropriate.
- Use one clear capture/inference lifecycle.
- Correctly release resources on stop, source changes, and application exit.
- Handle missing camera, busy camera, disconnects, invalid files, and backend errors.
- Align bounding-box coordinates with the rendered media after resizing/cropping.
- Derive FPS, latency, and counts from actual measurements.
- Clearly label mock data and keep it out of production inference paths.

### 5. Validate
- Run available tests, lint, type checks, and build.
- Check import paths, assets, responsive layout, and error states.
- Test start/stop, panel toggle, input switching, and persistence of detection state where possible.
- Report unavailable hardware tests honestly.
- Give exact run commands discovered from the project.

## Design guidance
- Glassmorphism panels with subtle blur, translucent border, soft shadow, rounded corners.
- Purple accent around `#7547FF`, neutral lavender/gray background, readable contrast.
- Camera/video is the hero; avoid permanently visible KPI-card grids.
- Controls are secondary and should not obstruct the subject.
- Support reduced motion and keyboard accessibility.
- Use responsive drawer behavior and avoid horizontal overflow.

## Safety and code quality
- Never delete or overwrite weights/data.
- Never expose secrets in logs or documentation.
- Validate uploads and avoid unsafe paths.
- Do not use broad exception swallowing.
- Avoid unneeded dependencies and global environment changes.
- Do not commit, push, deploy, or perform destructive operations without authorization.

## Completion report
State:
- What changed.
- Files changed.
- Which existing weights/config were preserved.
- Tests actually run and results.
- Exact launch commands.
- Known limitations and anything not verified.
