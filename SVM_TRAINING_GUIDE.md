# SVM Pose Classifier - User Guide

This guide explains how to use SVM machine learning models to improve pose recognition accuracy.

## Overview

The system now supports two pose classification methods:

1. **Rule-based Classification** (default): Uses fixed thresholds to determine sit/stand/lie
2. **SVM Machine Learning Classification** (recommended): Trains personalized models from your recorded data, outputs probability distributions

### Why Use SVM?

✅ **Automatically learns optimal thresholds** - No manual parameter tuning
✅ **Outputs probability distributions** - Know the model's confidence
✅ **Relative features** - Less affected by camera movement
✅ **Personalized** - Adapts to your body characteristics and environment
✅ **Lightweight and fast** - Runs on CPU, <1ms latency

---

## Quick Start

### Step 1: Record Training Data (5 minutes)

Run the data collection tool:

```bash
python collect_data.py
```

**Operation Process:**

1. Program opens camera and displays live feed
2. **Record sitting posture** (recommend 30+ seconds):
   - Stand outside camera view
   - Press `s` key
   - 5-second countdown, then automatic recording
   - Sit in chair and vary your sitting positions:
     - Upright sitting (facing camera)
     - Leaning forward (simulating writing)
     - Leaning back (resting position)
     - Turn left, turn right (sideways)
   - After 30 seconds, press `q` to stop

3. **Record standing posture** (recommend 30+ seconds):
   - Press `t` key
   - 5-second countdown, then automatic recording
   - Stand and perform various movements:
     - Face forward standing
     - Side-facing standing (left, right)
     - Close to camera (legs not visible)
     - Far from camera
   - Press `q` to stop

4. **Record lying posture** (recommend 20+ seconds):
   - Press `l` key
   - 5-second countdown, then automatic recording
   - Lie on bed:
     - Lying on back
     - Lying on side (left, right)
   - Press `q` to stop

5. Press `ESC` to exit program

**Parameter Adjustments** (optional):

```bash
# Modify countdown time (default 5 seconds)
python collect_data.py --countdown 3

# Modify suggested recording duration (default 30 seconds)
python collect_data.py --min-duration 20
```

**Data Storage Location**: JSON files in `training_data/` directory

---

### Step 2: Train SVM Model (1 minute)

Run training script:

```bash
python train_svm.py
```

The program will:
- Automatically load all samples from `training_data/`
- Use grid search to find optimal parameters
- Output accuracy reports and confusion matrix
- Save model to `models/pose_classifier_svm.pkl`

**Expected Output Example:**

```
[INFO] Loading sitting: 450 samples
[INFO] Loading standing: 380 samples
[INFO] Loading lying: 320 samples

[INFO] Total samples: 1150
[INFO] Feature dimensions: 57

[INFO] Training set: 920 samples
[INFO] Test set: 230 samples

[INFO] Using grid search to find optimal parameters...
[INFO] Best parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
[INFO] Cross-validation accuracy: 0.9457

Training accuracy: 0.9783 (97.83%)
Test accuracy: 0.9348 (93.48%)  ← Target: >90%

Test set classification report:
              precision    recall  f1-score   support
     sitting       0.95      0.94      0.94        89
    standing       0.93      0.95      0.94        77
       lying       0.93      0.92      0.92        64
```

**If accuracy <90%**: Need to record more data (re-run `collect_data.py`)

**Parameter Options**:

```bash
# Specify data directory
python train_svm.py --data-dir my_training_data

# Specify output path
python train_svm.py --output my_models/svm.pkl

# Adjust test set ratio (default 20%)
python train_svm.py --test-size 0.3

# Disable grid search (faster but may reduce accuracy)
python train_svm.py --no-grid-search
```

---

### Step 3: Run Program and See Results

```bash
python main.py --config config/config_gpu.yaml
```

**SVM model will auto-load** - no additional configuration needed!

Program will display:

```
[INFO] SVM classifier loaded: models/pose_classifier_svm.pkl
[INFO] Supported categories: ['sitting', 'standing', 'lying']
```

**In debug mode**, will display SVM probability distribution:

```
SVM Probabilities:
  Sitting: 0.85   [green progress bar]
  Standing: 0.12  [gray progress bar]
  Lying: 0.03     [gray progress bar]
```

---

## Advanced Usage

### Add More Data

If a certain pose isn't recognized accurately, re-record that pose's data:

```bash
python collect_data.py
# Only record the inaccurate pose (e.g., only press 's' to add sitting data)
# Exit and retrain
python train_svm.py
```

New data will automatically merge with existing data.

---

### View Recorded Data

```bash
ls -lh training_data/
# sitting_samples.json
# standing_samples.json
# lying_samples.json
```

Each file contains all samples for that pose.

---

### Downgrade to Rule-based Classification

If you want to temporarily not use SVM, delete or rename the model file:

```bash
mv models/pose_classifier_svm.pkl models/pose_classifier_svm.pkl.bak
```

Program will automatically downgrade to rule-based classification method.

---

### Specify Model Path in Configuration File

Edit `config/config_gpu.yaml` or `config/config_cpu.yaml`:

```yaml
behavior:
  svm_model_path: "custom_path/my_model.pkl"  # Custom model path
```

---

## Common Questions

### Q1: During data recording, program shows "Collected: 0 frames"?

**Reason**: MediaPipe cannot detect pose keypoints

**Solution**:
- Ensure sufficient lighting
- Try to have full body in frame
- Don't be too far from camera (2-4 meters optimal)

---

### Q2: Training shows "No training data found"?

**Reason**: `training_data/` directory is empty or files corrupted

**Solution**:
- Check if `collect_data.py` ran successfully
- Confirm `training_data/` directory has `.json` files
- Re-record data

---

### Q3: Test accuracy only 70-80%?

**Reason**: Insufficient data or poor data quality

**Solution**:
- Record at least 30 seconds for each pose
- Ensure diverse pose variations (don't stay still)
- Good lighting, accurate keypoint detection

---

### Q4: High CPU usage, running slow?

**Answer**: SVM classifier is very lightweight (<1ms), won't affect performance.

Bottleneck is still MediaPipe pose estimation (CPU, approx 30-70ms).

---

### Q5: After camera moves, accuracy drops?

**Explanation**: SVM uses relative features, theoretically less affected by camera movement.

But if angle changes drastically (e.g., from front to side), recommend:

1. Re-record 10-20 seconds of data at that angle
2. Retrain model

---

### Q6: Can models be exported/imported?

**Yes!** Model file `models/pose_classifier_svm.pkl` is standard sklearn format.

```bash
# Backup model
cp models/pose_classifier_svm.pkl backup/

# Restore model
cp backup/pose_classifier_svm.pkl models/
```

---

## Performance Comparison

| Method | CPU Usage | Accuracy | Camera Movement | Configuration Needed |
|------|---------|--------|-----------|---------|
| Rule-based | 0% | 80-85% | ⚠️ Needs re-tuning | ✅ None |
| **SVM (Recommended)** | **<1%** | **90-95%** | **✅ Auto-adapts** | **5-minute recording** |

---

## Technical Details

### Feature Vector (57 dimensions)

1. **3D Normalized Coordinates** (51 dims): 17 keypoints × 3D coords (x, y, z), normalized by torso length
2. **Geometric Features** (6 dims):
   - Torso angle (relative to vertical)
   - Hip-knee Z-axis difference (depth difference)
   - Hip-knee 3D distance
   - Hip height
   - Shoulder width
   - Keypoint visibility statistics

### Why Relative Features?

- **Height independent**: Normalizing by torso length, auto-adapts to different heights
- **Camera angle independent**: Uses 3D world landmarks (provided by MediaPipe)
- **Scale invariant**: When camera moves, proportional relationships stay stable

---

## Next Steps

If SVM model achieves 90%+ accuracy, congratulations! You've completed training a personalized pose classifier 🎉

Now the system will:
- ✅ Output real-time probability distributions
- ✅ Automatically select pose with highest probability
- ✅ Fall back to rule-based classification (if SVM fails)

**Enjoy more accurate prolonged sitting reminders!** 😊
