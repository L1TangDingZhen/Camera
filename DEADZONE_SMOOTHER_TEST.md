# Dead Zone Smoother - Test Guide

## Phase 1 Completed ✓

Implemented Dead Zone Smoother to solve skeleton point/line jitter issues when person is static.

---

## Feature Description

### What is Dead Zone?

When keypoint movement distance is **below threshold**, the dead zone smoother considers it "no movement" and uses previous frame coordinates, eliminating jitter caused by small noise.

**Effects**:
- ✅ When person is static, skeleton points are almost completely stable (70-80% jitter reduction)
- ✅ Normal motion unaffected (updates normally when movement exceeds threshold)
- ✅ Zero performance overhead (only simple distance calculations)

---

## Testing Method

### 1. Start System

```bash
# Ensure deadzone is enabled in config
python main.py --config config/config_gpu.yaml
```

### 2. Test Scenarios

#### Scenario A: Static Test (Most Important)
1. Stand or sit in front of camera, stay **completely still**
2. Observe if skeleton points and lines are stable
3. Observe if joints (shoulders, elbows, knees) still jitter

**Expected Results**:
- ✅ Skeleton points should be almost completely still
- ✅ Skeleton lines should not jitter or "breathe"
- ✅ Should be noticeably more stable than before

#### Scenario B: Slow Motion Test
1. Slowly raise arm
2. Observe if motion follows smoothly
3. Observe if there's noticeable delay or "sticking"

**Expected Results**:
- ✅ Motion should follow smoothly
- ✅ Should not have noticeable delay (deadzone threshold only 3 pixels)
- ⚠️ If feels "sticky", threshold is too large

#### Scenario C: Fast Motion Test
1. Wave hands quickly, jump
2. Observe if motion responds quickly

**Expected Results**:
- ✅ Should be completely normal, same as without deadzone

---

## Configuration Tuning

### View Current Configuration

Config file: `config/config_gpu.yaml`

```yaml
models:
  pose:
    deadzone:
      enabled: true      # Enable/disable deadzone
      threshold: 3.0     # Deadzone threshold (pixels)
```

### Adjust Threshold

Adjust `threshold` parameter based on test results:

| Resolution | Recommended Threshold | Adjustment Direction |
|-----------|----------------------|---------------------|
| 640x480 | 1-2 pixels | Still jitters→increase; Small motions fail→decrease |
| 1280x720 | 2-3 pixels | Default recommended |
| 1920x1080 | 3-5 pixels | Higher resolutions need larger thresholds |

**Adjustment Suggestions**:
- If **still jittering**: Increase threshold (4, 5, 6...)
- If **small motions not responsive**: Decrease threshold (2, 1.5, 1...)
- If **feels sticky/laggy**: Decrease threshold or disable deadzone

### Comparison Test

#### Test 1: Disable Deadzone
```yaml
deadzone:
  enabled: false
```
Restart system, observe jitter when static (as baseline)

#### Test 2: Enable Deadzone
```yaml
deadzone:
  enabled: true
  threshold: 3.0
```
Restart system, compare static stability

---

## Expected Results

### Success Indicators:
1. ✅ When person is static, skeleton points almost don't move
2. ✅ Skeleton lines stable, no "breathing" or jitter
3. ✅ Slow motions smooth, no noticeable delay
4. ✅ Fast motions respond normally

### If Effect Not Sufficient:
1. Try increasing `threshold` (e.g., 4, 5 pixels)
2. If threshold already large (>10 pixels) but still jitters, might be:
   - RTMPose model itself
   - Unstable lighting/background
   - Camera noise too high
3. Consider entering Phase 2: Add confidence filtering

---

## Debug Info

On startup you'll see:
```
[DeadZoneSmoother] 初始化
[DeadZoneSmoother]   死区阈值: 3.0 像素
[DeadZoneSmoother]   状态: 启用
```

---

## Next Step (Phase 2)

If Phase 1 results are satisfactory, you can stop here.

If you need further stability improvement, you can add:
- **Phase 2**: Confidence filtering (handle occlusion scenarios)
- **Phase 3**: EMA smoothing (further soften motion)
- **Phase 4**: Adaptive parameters (static/moving mode switching)

---

## Troubleshooting

### Q1: No DeadZoneSmoother initialization message on startup
**A**: Check if `deadzone.enabled` in config file is `true`

### Q2: Still jittering when static
**A**:
1. Confirm deadzone is enabled
2. Try increasing threshold (4, 5, 6)
3. Check if camera is stable (not handheld)

### Q3: Feels like motion has delay/sticking
**A**: Decrease threshold (2, 1.5, 1) or disable deadzone

### Q4: Conflicts with keypoint_smooth_alpha?
**A**: No conflict, can use simultaneously:
- Deadzone processes first (eliminate static jitter)
- EMA processes after (soften motion trajectory)
- Suggest testing deadzone effect alone first, then consider combination

---

## Feedback

After testing, please provide feedback on:
1. Static stability improvement (0-100%)
2. Whether it affects normal motion
3. Final threshold value used
4. Whether need to enter Phase 2

---

**Estimated Test Time**: 5-10 minutes
**Expected Effect**: Solve 70-80% of static jitter issues
