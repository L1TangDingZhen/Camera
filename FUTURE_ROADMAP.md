# Future Development Roadmap

**Project**: Life Tracker - Prolonged Sitting Reminder System
**Document Version**: 1.0
**Last Updated**: 2025-11-09

---

## Table of Contents

1. [Current System Architecture](#current-system-architecture)
2. [Short-term Plan (1-2 Months)](#short-term-plan-1-2-months)
3. [Mid-term Plan (3-6 Months)](#mid-term-plan-3-6-months)
4. [Long-term Plan (6+ Months)](#long-term-plan-6-months)
5. [Model Selection Rationale](#model-selection-rationale)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Timeline and Milestones](#timeline-and-milestones)

---

## Current System Architecture

### System Pipeline (First Edition)

```
Camera Input
    ↓
[1] Human Detection (YOLOv8m)
    ↓
[2] Pose Estimation (MediaPipe)
    ↓
[3] Pose Classification (SVM)
    ↓
[4] State Machine (Rule-based)
    ↓
[5] Session Tracker (Statistical)
    ↓
[6] Behavior Predictor (Statistical)
    ↓
Web Dashboard + Alerts
```

---

### Model Inventory - Current (First Edition)

| Stage | Component | Model/Algorithm | Framework | Device | Latency | Purpose |
|-------|-----------|----------------|-----------|--------|---------|---------|
| **Stage 1** | Human Detection | **YOLOv8m** | PyTorch | GPU | ~20ms | Detect person in frame |
| **Stage 2** | Pose Estimation | **MediaPipe Pose** | TensorFlow Lite | CPU | ~40ms | Extract 17 COCO keypoints |
| **Stage 3** | Pose Classification | **SVM (RBF kernel)** | scikit-learn | CPU | <1ms | Classify: sitting/standing/lying |
| **Stage 4** | State Machine | Rule-based Logic | Python | CPU | <1ms | Track state transitions |
| **Stage 5** | Session Tracking | Statistical Aggregation | NumPy | CPU | <1ms | Calculate duration stats |
| **Stage 6** | Behavior Prediction | Time-series Statistics | Python | CPU | ~3ms | Predict daily patterns |
| **Stage 7** | Web Dashboard | Flask + Chart.js | Python/JS | CPU | N/A | Visualization |

**Total Pipeline Latency**: ~65ms/frame = **15 FPS** (CPU mode)

---

### Current Limitations

| Limitation | Impact | Priority to Fix |
|------------|--------|----------------|
| **MediaPipe CPU-only** | Slow on embedded devices | 🔴 High |
| **Single-frame classification** | Cannot recognize dynamic actions | 🟡 Medium |
| **Statistical prediction** | Limited pattern learning | 🟡 Medium |
| **Static pose only** | Cannot detect: walking, exercise, cooking | 🟢 Low |
| **Single person only** | No multi-person support | 🟢 Low |

---

## Short-term Plan (1-2 Months)

### Theme: **Performance Optimization for Jetson Deployment**

---

### Milestone 1.1: Replace MediaPipe with RTMPose (GPU Acceleration)

**Objective**: Achieve 30+ FPS on Jetson Orin Nano Super @ 15W

#### Changes

| Component | Current | Upgrade To | Reason |
|-----------|---------|------------|--------|
| Pose Estimation | MediaPipe (CPU) | **RTMPose-s** (GPU) | 4x faster, GPU accelerated |
| Human Detection | YOLOv8m | **YOLOv8s** | Lighter, sufficient accuracy |
| TensorRT | Disabled | **Enabled (FP16)** | 2x faster inference |

#### Expected Performance

```
Current (PC):
  YOLOv8m (GPU): 20ms
  MediaPipe (CPU): 40ms
  Total: 60ms = 16 FPS

Optimized (Jetson Orin Nano Super @ 15W):
  YOLOv8s (TensorRT FP16): 8ms
  RTMPose-s (TensorRT FP16): 12ms
  Total: 20ms = 50 FPS ✅
```

#### Implementation Steps

1. **Install MMPose on Jetson**
   ```bash
   pip install openmim
   mim install mmcv-full
   mim install mmpose
   ```

2. **Download RTMPose-s model**
   ```bash
   mim download mmpose --config rtmpose-s_8xb256-420e_coco-256x192 \
       --dest models/
   ```

3. **Create RTMPose adapter** (`src/detectors/rtmpose_detector.py`)
   - Implement RTMPose inference
   - Output 17 COCO keypoints (same format as MediaPipe)
   - Ensure backward compatibility with existing pipeline

4. **Enable TensorRT optimization**
   - Export YOLOv8s to TensorRT engine
   - Configure RTMPose for TensorRT acceleration

5. **Benchmark on Jetson**
   - Test 15W, 25W power modes
   - Measure FPS, latency, power consumption

#### Success Criteria

- ✅ Achieve **30+ FPS** on Jetson Orin Nano Super @ 15W
- ✅ Maintain **<30ms** end-to-end latency
- ✅ No accuracy degradation (>95% pose classification accuracy)
- ✅ Power consumption **≤15W** in standard mode

---

### Milestone 1.2: Enhanced Data Collection Pipeline

**Objective**: Prepare for Transformer-based models

#### New Data Collection Format

**Current format** (`training_data/sitting_samples.json`):
```json
{
  "features": [0.12, 0.85, ...],  // 57-dim feature vector (hand-crafted)
  "label": "sitting",
  "timestamp": 1234567890.123
}
```

**New format** (`training_data_v2/sitting_sequences.json`):
```json
{
  "label": "sitting",
  "timestamp": 1234567890.123,
  "keypoints_sequence": [
    // Frame 1
    [[x1, y1, z1, vis1], [x2, y2, z2, vis2], ...],  // 17 keypoints
    // Frame 2
    [[x1, y1, z1, vis1], [x2, y2, z2, vis2], ...],
    ...
    // Frame 30
    [[x1, y1, z1, vis1], [x2, y2, z2, vis2], ...]
  ],
  "features_legacy": [...],  // Keep 57-dim for SVM compatibility
  "metadata": {
    "fps": 30,
    "duration_sec": 1.0,
    "person_id": "user_001"
  }
}
```

#### Implementation

1. **Update `collect_data.py`**
   - Add `--sequence-mode` flag
   - Save sliding windows of 30 frames
   - Maintain backward compatibility

2. **Create conversion script** (`scripts/convert_legacy_data.py`)
   - Convert old 57-dim features to new format (if possible)
   - Document data migration process

#### Benefits

- ✅ Support both SVM (single-frame) and ST-GCN (sequence) training
- ✅ Future-proof data format
- ✅ No need to re-collect static pose data

---

## Mid-term Plan (3-6 Months)

### Theme: **Transformer-based Intelligent Behavior Recognition**

---

### Milestone 2.1: Dynamic Action Recognition (ST-GCN)

**Objective**: Recognize dynamic behaviors (walking, exercise, cooking)

#### New Pipeline Component

```
YOLOv8s → RTMPose-s → Keypoint Buffer (30 frames)
                            ↓
                    ┌───────┴────────┐
                    ↓                ↓
            Single Frame         Sequence (30 frames)
                    ↓                ↓
                  SVM            ST-GCN
                    ↓                ↓
          Static Pose         Dynamic Action
        (sit/stand/lie)  (walk/exercise/cook)
```

#### ST-GCN Model Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | ST-GCN (Spatial-Temporal GCN) | Lightweight, designed for skeleton data |
| **Input** | (N, C, T, V, M) = (1, 3, 30, 17, 1) | 30 frames, 17 COCO keypoints, single person |
| **Parameters** | ~3M | Small model, fast inference |
| **FLOPs** | ~0.5 GFLOPs | Very efficient |
| **Framework** | PyTorch | Easy to integrate |
| **Pretrained** | NTU-RGB+D 60 | Transfer learning from action recognition dataset |

#### Jetson Performance Estimate

```
Optimized Pipeline (Jetson @ 15W):
  YOLOv8s (TensorRT): 8ms
  RTMPose-s (TensorRT): 12ms
  SVM (static pose): <1ms
  ST-GCN (dynamic action): 15ms
  Total: 35ms = 28 FPS ✅
```

#### Action Classes (Phase 1)

| Category | Actions | Training Data Needed |
|----------|---------|---------------------|
| **Static** (SVM) | sitting, standing, lying | ✅ Already collected |
| **Dynamic** (ST-GCN) | walking, exercising, stretching, cooking, cleaning | ⚠️ Need to collect |

#### Implementation Steps

1. **Collect dynamic action data**
   - 30 sequences × 5 actions = 150 sequences (15 minutes total)
   - Use updated `collect_data.py --sequence-mode`

2. **Train ST-GCN model**
   ```bash
   python train_stgcn.py \
       --data training_data_v2/ \
       --backbone stgcn \
       --epochs 50 \
       --batch-size 16
   ```

3. **Implement dual-path classifier** (`src/classifiers/hybrid_classifier.py`)
   ```python
   if motion_magnitude < threshold:
       prediction = svm_classifier(single_frame)  # Fast path
   else:
       prediction = stgcn_classifier(sequence_30frames)  # Slow path
   ```

4. **Benchmark on Jetson**
   - Measure latency, FPS, accuracy
   - Optimize inference with TensorRT/ONNX

#### Success Criteria

- ✅ Recognize 5+ dynamic actions with **>85%** accuracy
- ✅ Maintain **25+ FPS** on Jetson @ 15W
- ✅ Latency **<40ms**
- ✅ Seamless integration with existing SVM pipeline

---

### Milestone 2.2: Anomaly Behavior Detection

**Objective**: Detect abnormal behaviors (falling, prolonged stillness, sudden movements)

#### Approach: ST-GCN + AutoEncoder

**Training**: Only use normal behaviors
```
Normal sequences → ST-GCN Encoder → Latent Code → ST-GCN Decoder → Reconstructed sequence

Loss: MSE(original, reconstructed)
```

**Inference**: Abnormal behaviors have high reconstruction error
```
Input sequence → ST-GCN AutoEncoder → Reconstruction error

If error > threshold:
    🚨 Alert: Anomaly detected!
```

#### Anomaly Types

| Anomaly | Detection Method | Alert Type |
|---------|------------------|------------|
| **Falling** | Sudden change in hip height + high velocity | 🔴 Critical |
| **Prolonged stillness** | No movement for 30+ min (not sleeping) | 🟡 Warning |
| **Erratic movements** | High acceleration + unusual patterns | 🟡 Warning |
| **Unusual posture** | High reconstruction error | 🟢 Info |

#### Jetson Performance

```
ST-GCN AutoEncoder:
  Encoder: 10ms
  Decoder: 10ms
  Total: 20ms (only runs every 1 second)

Impact on real-time pipeline: Negligible ✅
```

#### Benefits

- ✅ **Safety**: Detect falls immediately (critical for elderly)
- ✅ **Health**: Alert unusual health states
- ✅ **No labeled anomaly data needed**: Unsupervised learning

---

### Milestone 2.3: Behavior Prediction Upgrade (Informer)

**Objective**: Improve behavior prediction accuracy using Transformer

#### Current vs Proposed

| Method | Current (Statistical) | Proposed (Informer) |
|--------|----------------------|---------------------|
| **Input** | Single time point stats | 7-day behavior sequence |
| **Output** | Simple probability | Next-hour prediction + confidence |
| **Captures** | Daily average patterns | Weekly trends, anomalies, periodicity |
| **Accuracy** | ~70% (estimated) | ~85% (target) |

#### Informer Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | Informer | Designed for long time-series forecasting |
| **Input** | (7 days × 24 hours) × 4 features | 168 time steps, 4 behavior probabilities |
| **Output** | Next 1-hour prediction | Probability distribution |
| **Parameters** | ~5M | Small model |
| **Inference Frequency** | Every 30 seconds | Low compute overhead |
| **Latency** | ~10ms | Runs asynchronously |

#### Input Features (Time-series)

```python
For each hour over past 7 days:
{
    "hour": 14,
    "day_of_week": "Monday",
    "sitting_prob": 0.85,
    "standing_prob": 0.10,
    "lying_prob": 0.05,
    "sleeping_prob": 0.00
}
```

#### Jetson Performance

```
Informer runs every 30 seconds (not every frame):
  Inference: 10ms
  Frequency: 2 times/minute
  Average overhead: 10ms × 2/60s ≈ 0.3ms/frame

Impact: Negligible ✅
```

#### Benefits

- ✅ **Better predictions**: Learn weekly patterns (weekday vs weekend)
- ✅ **Anomaly detection**: Flag unusual daily patterns
- ✅ **Personalized**: Adapts to individual routines
- ✅ **Trend analysis**: Detect long-term behavior changes

---

## Long-term Plan (6+ Months)

### Theme: **Advanced Multi-modal Intelligence**

---

### Milestone 3.1: Multi-person Interaction Recognition

**Objective**: Detect interactions between multiple people

#### Use Cases

- Family activity monitoring
- Office collaboration analysis
- Social behavior research

#### Model: ST-GCN + Attention

```
Person A keypoints (30 frames)  ──┐
                                  ├──→ ST-GCN → Cross-person Attention → Interaction
Person B keypoints (30 frames)  ──┘

Interactions: handshake, hug, conversation, argument, etc.
```

#### Jetson Performance (Estimated)

```
2-person pipeline:
  Detection + Pose: 20ms
  ST-GCN (2 persons): 30ms
  Total: 50ms = 20 FPS ⚠️

Power mode: 25W (needed for multi-person)
```

---

### Milestone 3.2: Fine-grained Activity Classification

**Objective**: Recognize 100+ detailed activities (yoga poses, exercise types, cooking steps)

#### Model: MS-G3D (Multi-Scale Graph 3D)

| Parameter | Value |
|-----------|-------|
| **Model** | MS-G3D Net |
| **Actions** | 100+ fine-grained classes |
| **Parameters** | ~10M |
| **Latency** | ~40ms on Jetson @ 25W |

#### Applications

- Smart fitness coach (evaluate exercise form)
- Rehabilitation assistance (track recovery progress)
- Cooking assistance (step-by-step guidance)

---

### Milestone 3.3: Long-term Trend Analysis (Informer v2)

**Objective**: Monthly/yearly behavior analysis

#### Features

- **30-day trend prediction**
- **Health score tracking** (sedentary hours trend)
- **Lifestyle change detection** (new job, illness, vacation)
- **Personalized recommendations**

#### Model

```
Input: 30 days × 24 hours × 4 features = (720, 4)
Output: Next 7 days prediction + trend analysis

Inference: Once per day (negligible compute)
```

---

### Milestone 3.4: Multi-modal Fusion

**Objective**: Combine vision + audio + environment sensors

#### Modalities

| Modality | Sensor | Information |
|----------|--------|-------------|
| **Vision** | Camera | Pose, actions |
| **Audio** | Microphone | Typing, phone calls, music |
| **Environment** | Sensors | Light, temperature, air quality |
| **Wearable** | Optional | Heart rate, sleep quality |

#### Benefits

- ✅ More accurate behavior context (sitting + typing = working)
- ✅ Privacy-preserving (no need to see screen)
- ✅ Holistic health monitoring

---

## Model Selection Rationale

### Why ST-GCN for Dynamic Actions?

| Alternative | ST-GCN | VideoSwin | TimeSformer | MoViNet |
|-------------|--------|-----------|-------------|---------|
| **Input** | Skeleton | RGB Video | RGB Video | RGB Video |
| **Params** | 3M | 28M | 22M | 3M |
| **Jetson Latency** | **15ms** ✅ | 80ms ❌ | 40ms ⚠️ | 25ms ⚠️ |
| **Accuracy** | High | Higher | Higher | High |
| **Privacy** | ✅ No RGB | ❌ RGB | ❌ RGB | ❌ RGB |

**Winner**: ST-GCN - Best balance of speed, accuracy, privacy

---

### Why Informer for Prediction?

| Alternative | Informer | LSTM | Temporal Fusion Transformer |
|-------------|----------|------|----------------------------|
| **Long sequence** | ✅ Efficient | ❌ Slow | ⚠️ Medium |
| **Latency** | **10ms** ✅ | 15ms | 25ms |
| **Accuracy** | High | Medium | Highest |
| **Complexity** | Medium | Low | High |

**Winner**: Informer - Designed for long time-series, efficient attention

---

## Performance Benchmarks

### Target Performance Metrics

| Stage | Target | Acceptable | Unacceptable |
|-------|--------|------------|--------------|
| **Short-term (Jetson optimization)** | 35+ FPS @ 15W | 25-35 FPS | <25 FPS |
| **Mid-term (+ ST-GCN)** | 25+ FPS @ 15W | 20-25 FPS | <20 FPS |
| **Long-term (Multi-person)** | 20+ FPS @ 25W | 15-20 FPS | <15 FPS |

### Accuracy Targets

| Component | Target Accuracy | Current |
|-----------|----------------|---------|
| **Pose Classification (SVM)** | >95% | ~95% (estimated) |
| **Dynamic Action (ST-GCN)** | >85% | N/A |
| **Anomaly Detection (AE)** | >90% recall, <5% FPR | N/A |
| **Behavior Prediction** | >80% | ~70% (statistical) |

---

## Timeline and Milestones

### Visual Timeline

```
2025-11 (Now)        2026-01         2026-04         2026-08         2027+
   │                   │               │               │               │
   ├─ First Edition ✅ │               │               │               │
   │                   │               │               │               │
   ├────── Milestone 1.1 (RTMPose) ────┤               │               │
   │       Milestone 1.2 (Data v2)     │               │               │
   │                                   │               │               │
   │                   ├──── Milestone 2.1 (ST-GCN) ───┤               │
   │                   │     Milestone 2.2 (Anomaly)   │               │
   │                   │     Milestone 2.3 (Informer)  │               │
   │                   │                               │               │
   │                   │               ├── Milestone 3.1 (Multi-person) ┤
   │                   │               │   Milestone 3.2 (Fine-grained) │
   │                   │               │   Milestone 3.3 (Trends)       │
   │                   │               │   Milestone 3.4 (Multi-modal)  │
   │                   │               │                                │
   v                   v               v                                v
First Edition    Jetson Ready    Intelligent AI               Advanced Features
```

### Detailed Schedule

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Optimization** | Nov 2025 - Jan 2026 (2 months) | RTMPose integration, Jetson deployment, 30+ FPS |
| **Phase 2: Intelligence** | Jan 2026 - Apr 2026 (3 months) | ST-GCN action recognition, Anomaly detection, Informer prediction |
| **Phase 3: Advanced** | Apr 2026 - Aug 2026 (4 months) | Multi-person, Fine-grained actions, Long-term trends |
| **Phase 4: Multi-modal** | Aug 2026+ | Audio + Environment fusion, Wearable integration |

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Jetson performance insufficient** | Low | High | Use lighter models (YOLOv8n, RTMPose-tiny), Reduce resolution |
| **ST-GCN accuracy poor** | Medium | Medium | Collect more diverse training data, Use pretrained models |
| **TensorRT conversion issues** | Medium | Medium | Use ONNX as intermediate format, Maintain PyTorch fallback |
| **Power consumption too high** | Low | Medium | Implement dynamic power modes, Skip-frame detection |

### Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Insufficient training data** | Medium | High | Data augmentation, Transfer learning from public datasets |
| **Jetson hardware unavailable** | Low | High | Continue PC development, Cloud deployment option |
| **Development time longer than planned** | Medium | Low | Prioritize P0/P1 features, Incremental releases |

---

## Success Metrics

### Quantitative Metrics

| Metric | Current | Short-term Target | Mid-term Target | Long-term Target |
|--------|---------|-------------------|-----------------|------------------|
| **FPS (Jetson @ 15W)** | 15 FPS | 35 FPS | 25 FPS | 20 FPS |
| **Latency** | 65ms | <30ms | <40ms | <50ms |
| **Power (Jetson)** | N/A | 15W | 15-20W | 20-25W |
| **Static Pose Accuracy** | 95% | 95% | 95% | 95% |
| **Dynamic Action Accuracy** | N/A | N/A | 85% | 90% |
| **Prediction Accuracy** | 70% | 70% | 80% | 85% |

### Qualitative Metrics

- ✅ User-friendly deployment (one-command setup on Jetson)
- ✅ Privacy-preserving (no RGB storage, only skeleton data)
- ✅ Real-time responsiveness (<100ms latency)
- ✅ Reliable 24/7 operation (low power, stable)

---

## Conclusion

This roadmap provides a **progressive, pragmatic approach** to evolving the Life Tracker system:

1. **Short-term**: Focus on performance optimization for embedded deployment
2. **Mid-term**: Add intelligent behavior recognition using lightweight Transformers
3. **Long-term**: Advanced features (multi-person, multi-modal, fine-grained)

**Key Principles**:
- ✅ **Incremental upgrades**: Each phase builds on previous work
- ✅ **Backward compatibility**: Preserve existing data and models
- ✅ **Jetson-first**: All features must run on target hardware
- ✅ **Privacy-preserving**: Skeleton-based, no RGB storage
- ✅ **Open-source**: Maintainable, extensible codebase

**End Goal**: A comprehensive, intelligent health monitoring system running efficiently on edge devices, providing real-time insights while preserving user privacy.

---

---

## Appendix: True Reinforcement Learning Implementation

**Added**: 2025-11-21
**Context**: Current RL agents (Ensemble & Decision) use supervised learning with RL-inspired concepts. This section documents how to evolve them into true online RL systems.

---

### Current Status: "RL" in Name Only

#### Why Current Implementation Isn't True RL

| Aspect | True RL | Current Implementation |
|--------|---------|----------------------|
| **Data Source** | Online interaction with environment | Offline pre-labeled dataset |
| **Learning** | Trial-and-error exploration | Supervised learning (gradient descent on labels) |
| **Feedback** | Reward signal (sparse, delayed) | Ground truth labels (dense, immediate) |
| **Training** | Continuous learning from interaction | One-time training, then deploy |
| **Algorithm** | Q-learning, Policy Gradient, DQN | Adam optimizer + CrossEntropy loss |

**Current agents are:**
- **RL Ensemble Agent**: Supervised learning to predict optimal weights
- **RL Decision Agent**: Supervised learning with heuristic "optimal actions"

**They borrow RL concepts** (Q-values, actions, rewards) but **train like supervised models**.

---

### Problem: Why Is True RL Hard Here?

#### The Ground Truth Problem

Unlike games or robotics, pose estimation has **no natural feedback signal**:

| Domain | Feedback Source | Cost | Suitable for Online RL |
|--------|----------------|------|----------------------|
| **Games (e.g., Go)** | Win/loss is automatic | Zero | ✅ Yes |
| **Robotics** | Physical sensors (fall/collision) | Low | ✅ Yes |
| **Self-driving** | Collision detection in simulator | Low-Medium | ✅ Yes |
| **Recommendation** | User clicks (passive collection) | Low | ✅ Yes |
| **Pose Classification** | **Human annotation required** | **High** | ❌ Difficult |

**Core issue**: The system cannot automatically know if "sitting" prediction is correct without asking the user.

---

### Three Paths to True RL

---

### Path 1: Real-time User Feedback (Not Practical)

#### Concept

```python
# Online learning with immediate user feedback

while running:
    # Agent predicts
    prediction = agent.predict(keypoints)
    display_to_user(f"Are you {prediction}? (Y/N)")

    # Wait for user feedback
    user_response = wait_for_user_input()  # Blocking!

    # Compute reward
    if user_response == 'Y':
        reward = +1
    else:
        reward = -1
        true_label = ask_user_for_correct_label()

    # Update agent online
    agent.learn(state, action, reward, next_state)
```

#### Problems

| Issue | Impact | Severity |
|-------|--------|----------|
| **Constant interruption** | User must click every few seconds | 🔴 Critical |
| **Attention drain** | Cannot do normal work | 🔴 Critical |
| **Feedback delay** | User reaction time ~1-2s | 🟡 Medium |
| **User fatigue** | Will stop responding after 5 minutes | 🔴 Critical |

**Verdict: ❌ Not feasible for real-world deployment**

---

### Path 2: Hardware Sensors (Expensive but Reliable)

#### Concept

```python
# Multi-modal sensor fusion for automatic ground truth

# Hardware setup
chair_sensor = PressureSensor()      # $50-200
bed_sensor = PressureSensor()        # $50-200
wearable = IMUSensor()               # $100-500 (smartwatch)
depth_camera = DepthCamera()         # $200-400 (optional)

# Automatic ground truth
def get_ground_truth_from_sensors():
    chair_pressure = chair_sensor.read()
    bed_pressure = bed_sensor.read()
    body_orientation = wearable.get_orientation()

    # Fusion logic
    if chair_pressure > 10kg:
        return 'sitting'
    elif bed_pressure > 20kg:
        return 'lying'
    elif body_orientation.is_upright():
        return 'standing'
    else:
        return 'unknown'

# True online RL
while running:
    prediction = agent.predict(keypoints)
    time.sleep(0.1)  # Small delay

    true_label = get_ground_truth_from_sensors()  # Automatic!

    if true_label != 'unknown':
        reward = +1 if prediction == true_label else -1
        agent.learn(state, action, reward, next_state)
```

#### Cost-Benefit Analysis

| Component | Cost | Accuracy | Notes |
|-----------|------|----------|-------|
| **Chair pressure sensor** | $50-200 | 95% | Detects sitting reliably |
| **Bed pressure sensor** | $50-200 | 95% | Detects lying reliably |
| **Smartwatch/IMU** | $100-500 | 90% | Body orientation (already owned by many) |
| **Depth camera (optional)** | $200-400 | 98% | Height estimation, occlusion handling |
| **Total** | $200-1300 | - | One-time investment |

#### Pros & Cons

**✅ Pros:**
- Automatic ground truth (no user input needed)
- Continuous learning 24/7
- Multi-modal fusion improves robustness
- Sensors work even when camera fails

**❌ Cons:**
- High initial cost ($200-1300)
- Installation complexity
- Sensor calibration needed
- May not cover all poses (e.g., standing at desk without pressure sensor)
- Maintenance (battery replacement, recalibration)

**Verdict: ⚠️ Feasible but expensive. Good for dedicated research setup.**

---

### Path 3: Offline Annotation with Active Learning (Recommended)

#### Concept: The Best of Both Worlds

User's insight: **"Save uncertain clips, let user annotate later offline"**

```python
# Smart data collection during deployment

class SmartDataCollector:
    def __init__(self):
        self.uncertainty_threshold = 0.7
        self.buffer = []
        self.annotation_queue = []

    def process_frame(self, keypoints):
        # Agent predicts
        prediction, confidence = agent.predict_with_uncertainty(keypoints)

        # High confidence: trust the model
        if confidence > self.uncertainty_threshold:
            return prediction  # No annotation needed

        # Low confidence: save for later annotation
        else:
            clip = {
                'keypoints_sequence': recent_30_frames,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': time.time(),
                'id': generate_uuid()
            }
            self.annotation_queue.append(clip)
            save_to_disk(clip, f'uncertain_clips/{clip["id"]}.json')

            return prediction  # Still output, but flagged

    def get_annotation_queue(self):
        # Return clips that need annotation
        return sorted(self.annotation_queue, key=lambda x: x['confidence'])
```

#### Offline Annotation Interface

```python
# Simple annotation tool (can be web-based or CLI)

def annotation_interface():
    clips = load_uncertain_clips()

    print(f"You have {len(clips)} uncertain clips to annotate")
    print("This will take ~5-10 minutes")

    for i, clip in enumerate(clips):
        # Replay clip
        replay_video(clip['keypoints_sequence'])

        # Show model's guess
        print(f"\nClip {i+1}/{len(clips)}")
        print(f"Model prediction: {clip['prediction']} (confidence: {clip['confidence']:.2f})")
        print("Actual label? (1=sitting, 2=standing, 3=lying, s=skip)")

        user_input = input("> ")

        if user_input == 's':
            continue

        true_label = ['sitting', 'standing', 'lying'][int(user_input)-1]

        # Save annotation
        clip['true_label'] = true_label
        clip['annotated_at'] = time.time()
        save_annotation(clip)

    print("\n✅ Annotation complete! Training updated model...")
    retrain_agent_incremental(clips)
```

#### Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Initial Training (Offline)                     │
├─────────────────────────────────────────────────────────┤
│ 1. Collect labeled data (60 min)                        │
│ 2. Train initial agent (supervised)                     │
│ 3. Deploy agent                                          │
└─────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Deployment with Smart Collection (Online)      │
├─────────────────────────────────────────────────────────┤
│ Agent runs normally, but:                                │
│ - Save uncertain clips (confidence < 0.7)               │
│ - Save environment changes (camera angle shift detected)│
│ - Target: Collect 20-50 clips per week                  │
└─────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Weekly Annotation Session (10 minutes)         │
├─────────────────────────────────────────────────────────┤
│ User annotates saved clips:                              │
│ - Watch 30-second clip replay                            │
│ - Click correct label                                    │
│ - System learns from corrections                         │
└─────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Incremental Retraining (Automatic)             │
├─────────────────────────────────────────────────────────┤
│ Every week:                                              │
│ - Retrain agent with new annotations                    │
│ - Use experience replay (old + new data)                │
│ - Deploy updated model                                   │
│ - Confidence threshold adapts over time                  │
└─────────────────────────────────────────────────────────┘
            ↓
    (Repeat Phase 2-4)
```

---

### Implementation Roadmap

#### Stage 1: Smart Data Collection (1 week)

**Goal**: Implement uncertainty-based clip saving

**Implementation**:

```bash
# 1. Add uncertainty estimation to agents
# src/classifiers/pose_classifier_dl.py
def predict_with_uncertainty(self, landmarks):
    probs = self.predict_proba(landmarks)
    max_prob = max(probs.values())
    entropy = -sum(p * np.log(p + 1e-8) for p in probs.values())

    # Multiple uncertainty metrics
    uncertainty = {
        'confidence': max_prob,
        'entropy': entropy,
        'variance': np.var(list(probs.values()))
    }

    return probs, uncertainty

# 2. Add data collector
# src/utils/active_learning_collector.py
class ActiveLearningCollector:
    def __init__(self, save_dir='uncertain_clips'):
        self.save_dir = save_dir
        self.uncertainty_threshold = 0.7
        self.buffer = deque(maxlen=90)  # 3 seconds @ 30fps

    def add_frame(self, keypoints, prediction, uncertainty):
        self.buffer.append({
            'keypoints': keypoints,
            'timestamp': time.time()
        })

        # Save clip if uncertain
        if uncertainty['confidence'] < self.uncertainty_threshold:
            self.save_uncertain_clip(prediction, uncertainty)

# 3. Integrate into main.py
from src.utils.active_learning_collector import ActiveLearningCollector

collector = ActiveLearningCollector()

while True:
    keypoints = pose_estimator.estimate(frame, bbox)
    prediction, uncertainty = classifier.predict_with_uncertainty(keypoints)

    collector.add_frame(keypoints, prediction, uncertainty)
```

**Success Criteria**:
- ✅ System saves 20-50 uncertain clips per week
- ✅ Clips include 3-second context (before/after uncertain frame)
- ✅ Minimal performance impact (<1ms overhead)

---

#### Stage 2: Annotation Interface (3 days)

**Goal**: User-friendly tool for batch annotation

**Implementation**:

```bash
# Simple CLI tool
python scripts/annotate_clips.py

# Or web interface
python scripts/annotation_server.py
# Open browser: http://localhost:5001/annotate
```

**Features**:
- Replay clip with skeleton overlay
- Keyboard shortcuts (1/2/3 for sitting/standing/lying)
- Progress bar
- Estimated time remaining
- Skip ambiguous clips

**Success Criteria**:
- ✅ User can annotate 50 clips in 10 minutes
- ✅ Annotation saves to database
- ✅ Can resume interrupted session

---

#### Stage 3: Incremental Learning (2 weeks)

**Goal**: Update agent with new annotations

**Implementation**:

```python
# scripts/retrain_incremental.py

def incremental_training(agent, new_annotations, old_data):
    # Experience replay: Mix old and new
    combined_data = combine_datasets(
        old_data,
        new_annotations,
        ratio=0.7  # 70% old, 30% new to prevent catastrophic forgetting
    )

    # Fine-tune agent (smaller learning rate)
    agent.train(
        combined_data,
        epochs=20,
        lr=0.0001,  # Lower than initial training (0.001)
        freeze_early_layers=True  # Only update final layers
    )

    # Validate performance hasn't degraded
    val_acc = agent.evaluate(validation_set)
    if val_acc > 0.9:  # Threshold
        agent.save('models/agent_v2.pt')
        print("✅ Model updated successfully")
    else:
        print("⚠️ New model worse than baseline, keeping old model")
```

**Success Criteria**:
- ✅ Agent accuracy improves after retraining
- ✅ No catastrophic forgetting (old accuracy maintained)
- ✅ Training completes in <30 minutes

---

#### Stage 4: True Online RL (Future)

**Goal**: Continuous learning with TD-learning

Once Stages 1-3 are stable, optionally evolve to true DQN:

```python
# True DQN with experience replay

class OnlineRLAgent:
    def __init__(self):
        self.dqn = DecisionDQN()
        self.target_dqn = DecisionDQN()
        self.replay_buffer = deque(maxlen=10000)
        self.epsilon = 0.1  # Exploration rate

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        else:
            q_values = self.dqn(state)
            return q_values.argmax()  # Exploit

    def learn_from_annotation(self, state, action, true_label, next_state):
        # Compute reward
        prediction = self.get_prediction_from_action(state, action)
        reward = +10 if prediction == true_label else -10

        # Store experience
        self.replay_buffer.append((state, action, reward, next_state))

        # DQN update
        if len(self.replay_buffer) > 32:
            batch = random.sample(self.replay_buffer, 32)
            self.update_dqn(batch)

    def update_dqn(self, batch):
        states, actions, rewards, next_states = zip(*batch)

        # Q-learning update
        q_values = self.dqn(states)
        next_q_values = self.target_dqn(next_states).max(1)[0]

        target_q = rewards + 0.99 * next_q_values  # Gamma = 0.99
        loss = F.mse_loss(q_values[actions], target_q)

        loss.backward()
        self.optimizer.step()
```

**Differences from Stage 3**:
- Experience replay buffer
- Exploration (ε-greedy)
- Temporal difference learning
- Target network for stability

---

### Comparison: Three Paths

| Path | Annotation Effort | Cost | Accuracy Improvement | Deployment Complexity |
|------|------------------|------|---------------------|---------------------|
| **Path 1: Real-time Feedback** | ⚠️ Continuous (impractical) | $0 | High | Low |
| **Path 2: Hardware Sensors** | ✅ Zero | $$$ ($200-1300) | High | High |
| **Path 3: Active Learning** | ✅ 10 min/week | $0 | Medium-High | Medium |

**Recommendation**: **Path 3 (Active Learning)** is the sweet spot:
- Minimal user effort (10 min/week vs continuous)
- Zero hardware cost
- Gradual improvement over time
- Practical for real-world deployment

---

### Expected Outcomes

#### After 3 Months of Active Learning

| Metric | Initial (Supervised Only) | After 3 Months |
|--------|-------------------------|----------------|
| **Accuracy (baseline environment)** | 93% | 93% (maintained) |
| **Accuracy (new environment)** | 75% | 88% ↑ |
| **Confidence in uncertain cases** | 0.55 | 0.75 ↑ |
| **Annotation effort** | 60 min (initial) | 10 min/week = 120 min total |
| **Clips annotated** | 1000 | +600 (incremental) |

#### Key Benefits

1. **Adaptation to environment changes** (moved house, new camera angle)
2. **Personalization** (learns your unique sitting/standing postures)
3. **Continuous improvement** (model gets better over time)
4. **Low effort** (10 min/week vs hours of initial collection)

---

### Implementation Priority

| Stage | Priority | Effort | Value | Timeline |
|-------|----------|--------|-------|----------|
| **Stage 1: Smart Collection** | 🔴 High | 1 week | High | Month 1 |
| **Stage 2: Annotation UI** | 🔴 High | 3 days | High | Month 1 |
| **Stage 3: Incremental Learning** | 🟡 Medium | 2 weeks | Medium | Month 2-3 |
| **Stage 4: True Online RL** | 🟢 Low | 3 weeks | Low | Future (optional) |

**Recommendation**: Implement Stages 1-2 first (minimal effort, immediate value). Stage 3 when you have collected enough annotations (50+ clips). Stage 4 is optional research direction.

---

### Conclusion

While current "RL" agents use supervised learning, they can evolve into true RL systems through:

1. **Practical short-term**: Active learning with offline annotation (Path 3)
2. **Expensive but reliable**: Hardware sensor fusion (Path 2)
3. **Research long-term**: True online DQN (Stage 4)

**User's key insight is correct**: Saving uncertain clips for later annotation is the most practical approach. It balances:
- Minimal user interruption (batch annotation vs real-time)
- Zero hardware cost
- Continuous model improvement
- Realistic for rental housing (no sensor installation needed)

This approach transforms supervised "RL" into semi-supervised active learning, which is the first step toward true online RL.

---

**Document End** | For questions or suggestions, please open an issue on GitHub.
