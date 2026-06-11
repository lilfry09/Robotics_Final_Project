# DepthVLA Complete Roadmap: From Current State to SOTA

## 📍 Current State (As of 2024-06-11)

### What Works
- ✅ RGB-only OpenVLA-OFT: 15/15 (100%) on clean, 17/30 (56.7%) on LIBERO-Plus
- ✅ Action-summary architecture: Preserves 86.7% of RGB performance
- ✅ Depth signal probe: Proves data is informative (RMSE 0.019 vs 0.031)

### What Fails
- ❌ **Normal ≈ Null**: 14/30 vs 14/30 on LIBERO-Plus
- ❌ Depth not used causally: Model ignores depth information
- ❌ Below RGB baseline: 14/30 < 17/30

---

## 🎯 Three Paths Forward

### Path A: Fix Causality First (Recommended) ⭐
**Goal**: Make depth work on current data
**Timeline**: 2-3 weeks
**Expected**: 22/30 on LIBERO-Plus (+5 tasks over RGB)

### Path B: Scale Data Simultaneously
**Goal**: More data + better training
**Timeline**: 6-8 weeks
**Expected**: 24/30 on LIBERO-Plus + RLBench 90%

### Path C: Full SOTA Push
**Goal**: Beat all baselines on multiple benchmarks
**Timeline**: 10-12 weeks
**Expected**: Top-tier publication

---

## Path A: Fix Causality (Start Here)

### Week 1: Basic Fixes
**Implement**: Depth dropout + contrastive loss
```python
# 1. Add curriculum dropout
depth_dropout = CurriculumDepthDropout(0.5, 0.2, 5000)

# 2. Add contrastive loss
loss_contrastive = relu(loss_normal - loss_null + 0.05)
loss_total = loss_action + 0.3 * loss_contrastive

# 3. Train 5k steps, same 5 tasks
# Target: Normal 14/15, Null 11/15 (gap ≥3)
```

**Files to modify**:
- `vla-scripts/finetune_depthvla.py`: Add dropout + contrastive
- `experiments/robot/libero/run_libero_eval.py`: Already has ablations

**Success criteria**:
- Normal > Null by ≥3 tasks on clean 5 tasks
- Normal ≥ 13/15 (preserve quality)

**If fail**: Increase contrastive_weight to 0.5, retry

### Week 2: Validate on LIBERO-Plus
**Evaluate**: Best checkpoint from Week 1 on 30 tasks
```bash
python run_libero_eval.py \
  --task_ids <libero_plus_30> \
  --depth_ablation_mode normal

# Target: > 17/30 (beat RGB)
# Stretch: > 20/30 (substantial gain)
```

**Success criteria**:
- Normal > 17/30 (beat RGB-only)
- Normal > Null + 3 (depth helps under perturbations)

**If fail**: Go to Week 3 (multi-scale supervision)

### Week 3: Multi-Scale Supervision (If Needed)
**Implement**: Hierarchical spatial losses
```python
# Add 3-level supervision
spatial_loss = HierarchicalSpatialLoss()
loss_total += 0.2 * spatial_loss(depth_context, spatial_targets)

# Targets: occupancy + vectors + contacts
```

**Expected**: +2-3 additional tasks (19-20/30 → 22-23/30)

**Deliverable**: Paper-ready results on LIBERO-Plus

---

## Path B: Scale Data (If Path A Insufficient)

### Week 4-5: Multi-Dataset Collection
**Datasets to add**:
1. LIBERO-90 (full suite): 90 tasks
2. RLBench (18 tasks): Standard benchmark
3. Calvin (long-horizon): Generalization test

**Data prep**:
```bash
# Download
./scripts/download_rlbench.sh
./scripts/download_calvin.sh

# Generate RGB-D
python generate_rgbd_data.py --dataset rlbench
python generate_rgbd_data.py --dataset calvin
```

**Storage**: ~200GB total

### Week 6-7: Cross-Dataset Training
**Strategy**: Mixed batch training
```python
dataset_mixer = {
    'libero': 0.4,   # 40% from LIBERO
    'rlbench': 0.3,  # 30% from RLBench
    'calvin': 0.3    # 30% from Calvin
}

# Train 30k steps on mixed data
# Expected: Better generalization
```

**Target**:
- LIBERO-Plus: 24/30
- RLBench: 88-90%
- Calvin: 80-85%

### Week 8: Multi-Benchmark Evaluation
**Full evaluation suite**:
```yaml
benchmarks:
  - libero_plus: 30 tasks
  - rlbench: 18 tasks
  - calvin: long_horizon
  - simpler_env: real_world
```

**Deliverable**: Multi-benchmark comparison table

---

## Path C: Full SOTA Push (Ambitious)

### Week 9-10: Advanced Techniques
**Implement**:
1. Temporal depth context (LSTM over history)
2. Multi-view attention fusion
3. Depth-guided action chunking
4. Hard negative mining

**Expected gains**:
- Temporal: +2-3%
- Multi-view: +2%
- Adaptive chunking: +1-2%
- Hard negatives: +2%

**Total**: +7-9% → 24-26/30 on LIBERO-Plus

### Week 11: Real-World Validation
**Hardware needed**: Physical robot (Franka, WidowX, or similar)

**Tasks**:
1. Camera calibration
2. Depth sensor setup (RealSense D435)
3. 5-10 real-world tasks
4. Compare RGB-only vs DepthVLA

**Expected**: Real-world success rate 60-80%

### Week 12: Paper Writing
**Sections**:
1. Introduction (motivation)
2. Related Work (10 papers, use RELATED_WORK_COMPARISON.md)
3. Method (curriculum dropout, contrastive, hierarchical)
4. Experiments (3 benchmarks + ablations)
5. Analysis (where depth helps, failure modes)
6. Conclusion

**Target venue**:
- CoRL 2025 (deadline ~June)
- ICRA 2026 (deadline ~September)
- RSS 2025 (deadline ~January)

---

## 📊 Expected Performance Comparison

### Path A Only (3 weeks)
| Benchmark | RGB | Previous | Target | Status |
|-----------|-----|----------|--------|--------|
| LIBERO-Plus | 17/30 | 14/30 | 22/30 | Beat RGB ✓ |
| RLBench | - | - | - | Not tested |
| Calvin | - | - | - | Not tested |

**Publication**: Workshop or short paper

### Path B (8 weeks)
| Benchmark | RGB | SOTA | Target | Status |
|-----------|-----|------|--------|--------|
| LIBERO-Plus | 17/30 | - | 24/30 | Strong result ✓ |
| RLBench | - | 88.2% | 90% | Match SOTA ✓ |
| Calvin | 65% | - | 85% | New SOTA ✓ |

**Publication**: Top conference (CoRL, ICRA)

### Path C (12 weeks)
| Benchmark | RGB | SOTA | Target | Status |
|-----------|-----|------|--------|--------|
| LIBERO-Plus | 17/30 | - | 26/30 | New SOTA ✓✓ |
| RLBench | - | 88.2% | 92% | Beat SOTA ✓✓ |
| Calvin | 65% | - | 90% | Strong SOTA ✓✓ |
| Real Robot | - | - | 75% | Real-world ✓ |

**Publication**: Top conference + journal extension

---

## 🎬 Getting Started (Today)

### Option 1: Minimal (1 command)
```bash
cd /root/autodl-tmp/openvla-oft
./run_depth_fix_experiment.sh
# Runs basic dropout + contrastive (3k steps)
# Check: Normal > Null?
```

### Option 2: Advanced (Full pipeline)
```bash
cd /root/autodl-tmp/openvla-oft
./run_advanced_training.sh
# Runs 3-stage pipeline
# Stage 1: Validation (5k)
# Stage 2: LIBERO-Plus eval
# Stage 3: Multi-dataset (if successful)
```

### Option 3: Manual (Step by step)
```bash
# 1. Read quick reference
cat QUICK_REFERENCE.md

# 2. Understand the fix
cat TRAINING_FIX_GUIDE.md | less

# 3. Implement components
cp vla-scripts/advanced_training_components.py vla-scripts/

# 4. Modify training script
# Add CurriculumDepthDropout + MultiLevelContrastiveLoss

# 5. Train
python vla-scripts/finetune_depthvla.py --depth_dropout 0.3 ...

# 6. Evaluate
python run_libero_eval.py --depth_ablation_mode normal
python run_libero_eval.py --depth_ablation_mode null
python run_libero_eval.py --depth_ablation_mode shuffle
```

---

## 🔍 Decision Tree

```
Start
  │
  ├─> Do you need results ASAP (2-3 weeks)?
  │   └─> Path A: Fix causality only
  │       Success? → Paper ready ✓
  │       Fail? → Go to Path B
  │
  ├─> Do you want strong multi-benchmark results (8 weeks)?
  │   └─> Path B: Multi-dataset training
  │       Success? → Top conference ✓
  │       Fail? → Debug, iterate
  │
  └─> Do you want SOTA + real-world (12 weeks)?
      └─> Path C: Full advanced optimization
          Success? → Top conference + journal ✓
          Fail? → Still have Path A/B results
```

---

## 📁 File Guide

### Must Read (Start Here)
1. **README_DOCS.md** - Navigation and overview
2. **QUICK_REFERENCE.md** - Quick lookup
3. **TRAINING_FIX_GUIDE.md** - Implementation details

### Deep Dive (When Needed)
4. **OPTIMIZATION_TIMELINE.md** - What we tried, what worked
5. **RELATED_WORK_COMPARISON.md** - Literature comparison
6. **ADVANCED_OPTIMIZATION.md** - Full SOTA strategy

### Implementation
7. **vla-scripts/advanced_training_components.py** - Code
8. **run_depth_fix_experiment.sh** - Basic pipeline
9. **run_advanced_training.sh** - Advanced pipeline

### Reference
10. **DATASET_COMPARISON_SUMMARY.md** - Data analysis
11. **ADVANCED_IMPLEMENTATION_SUMMARY.md** - This file

---

## ✅ Success Checklist

### Path A Success (Minimum Viable)
- [ ] Implemented dropout + contrastive
- [ ] Trained 5k steps on 5 tasks
- [ ] Normal > Null by ≥3 tasks
- [ ] Evaluated on LIBERO-Plus
- [ ] Normal > 17/30 (beat RGB)
- [ ] Wrote up results

### Path B Success (Strong Paper)
- [ ] All Path A items ✓
- [ ] Collected RLBench + Calvin data
- [ ] Trained on multi-dataset
- [ ] Evaluated on 3 benchmarks
- [ ] Beat SOTA on at least 1 benchmark
- [ ] Full ablation studies

### Path C Success (Top Publication)
- [ ] All Path B items ✓
- [ ] Implemented temporal + multi-view
- [ ] Real-world robot experiments
- [ ] Beat SOTA on multiple benchmarks
- [ ] Comprehensive analysis section
- [ ] Reproducibility materials

---

## 🚨 Common Pitfalls

### Pitfall 1: Skipping Path A
**Mistake**: "Let me collect more data first"
**Why bad**: Won't fix normal ≈ null
**Correct**: Fix training method FIRST on 5 tasks

### Pitfall 2: Not Testing Ablations
**Mistake**: Only test normal depth
**Why bad**: Can't tell if depth is used
**Correct**: Always run normal/null/shuffle

### Pitfall 3: Changing Multiple Things
**Mistake**: Add dropout + contrastive + spatial + temporal at once
**Why bad**: Can't isolate what works
**Correct**: Add one at a time, validate each

### Pitfall 4: Ignoring Failure Signals
**Mistake**: Normal ≈ null, "maybe more steps will help"
**Why bad**: Training method is broken
**Correct**: Stop, diagnose, fix method

---

## 🎓 Key Lessons

1. **Method > Data**: Fix training before scaling data
2. **Validate Early**: Test on 5 tasks before 30 tasks
3. **Ablate Everything**: Normal/null/shuffle every time
4. **Iterate Fast**: 3k steps is enough to see if method works
5. **Don't Skip Basics**: Dropout + contrastive solves 80% of the problem

---

## 📞 Final Recommendations

### If You Have 2 Weeks
→ **Path A**: Fix causality, beat RGB-only
→ **Deliverable**: Workshop paper or ArXiv

### If You Have 2 Months
→ **Path B**: Multi-dataset, multiple benchmarks
→ **Deliverable**: CoRL/ICRA paper

### If You Have 3 Months
→ **Path C**: Full SOTA push with real robot
→ **Deliverable**: Top conference + potential journal

### Start Today
```bash
# Quick validation (3 hours)
./run_depth_fix_experiment.sh

# Check if contrastive helps
grep "loss/contrastive" experiments/*/training.log

# If contrastive > 0 and decreasing → good sign
# Run full evaluation tomorrow
```

**Most important**: Take the first step today. Implementation beats planning.

🚀 **Let's make DepthVLA work!**
