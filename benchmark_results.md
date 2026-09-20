======================================================================
BENCHMARK: PKL vs ONNX × RF vs XGBoost (All 4 Models)
======================================================================

[1] Loading dataset...
    Test set: 41776 URLs (20888 legit, 20888 phishing)

[2] Loading PKL models...
    Loading RF pkl...    
    RF pkl type: Pipeline
    Loading XGBoost pkl...
    ⚠ XGB pkl failed to load: 229
    ⚠ This pkl was saved with a newer numpy version (Colab).
    ⚠ To fix: re-export the pkl from Colab with matching numpy,
    ⚠ or run the benchmark on Colab instead.
    → Skipping XGB (PKL), benchmarking remaining 3 models.

[3] Loading ONNX models...

[4] RF (Pickle) — predicting...
    Measuring latency (1000 samples)...
    Done. Total predict time: 2.8s

[5] XGBoost (Pickle) — SKIPPED (failed to load)

[6] RF (ONNX) — extracting features...
      20000/41776...
      40000/41776...
    Features: (41776, 1509), extraction: 188.7s
    Predicting (batch)...
    Measuring latency (1000 samples)...

[7] XGBoost (ONNX) — extracting features...
      20000/41776...
      40000/41776...
    Features: (41776, 160), extraction: 187.9s
    Predicting (batch)...
    Measuring latency (1000 samples)...

================================================================================
RESULTS
================================================================================

--- RF (PKL) ---
              precision    recall  f1-score   support

  Legitimate     0.9720    0.9994    0.9855     20888
    Phishing     0.9994    0.9712    0.9851     20888

    accuracy                         0.9853     41776
   macro avg     0.9857    0.9853    0.9853     41776
weighted avg     0.9857    0.9853    0.9853     41776


--- RF (ONNX) ---
              precision    recall  f1-score   support

  Legitimate     0.9736    0.9994    0.9863     20888
    Phishing     0.9994    0.9729    0.9860     20888

    accuracy                         0.9861     41776
   macro avg     0.9865    0.9861    0.9861     41776
weighted avg     0.9865    0.9861    0.9861     41776


--- XGB (ONNX) ---
              precision    recall  f1-score   support

  Legitimate     0.9829    0.9969    0.9899     20888
    Phishing     0.9969    0.9826    0.9897     20888

    accuracy                         0.9898     41776
   macro avg     0.9899    0.9898    0.9898     41776
weighted avg     0.9899    0.9898    0.9898     41776

Confusion Matrix (RF (PKL)):  TN= 20876 FP=    12 | FN=   602 TP= 20286
Confusion Matrix (RF (ONNX)):  TN= 20875 FP=    13 | FN=   566 TP= 20322
Confusion Matrix (XGB (ONNX)):  TN= 20824 FP=    64 | FN=   363 TP= 20525

================================================================================
COMPARISON TABLE (All 4 Models)
================================================================================
Metric                        RF (PKL)    RF (ONNX)   XGB (ONNX)
---------------------------------------------------------------------------
Accuracy                        0.9853       0.9861       0.9898
Precision (Phish)               0.9994       0.9994       0.9969
Recall (Phish)                  0.9712       0.9729       0.9826
F1-Score (Phish)                0.9851       0.9860       0.9897
False Positive Rate             0.0006       0.0006       0.0031
---------------------------------------------------------------------------
Avg Latency (ms)                22.788        0.044        0.054
P95 Latency (ms)                26.097        0.056        0.072
Max Latency (ms)                99.755        0.174        0.296
---------------------------------------------------------------------------
Model Size (KB)                   4345          901         2001
Process Memory (MB)              598.6
Test Samples                     41776

================================================================================
CROSS-VARIANT CHECK (RF PKL = undersampled vs RF ONNX = non-undersampled)
================================================================================
  RF variants agree on 41657/41776 predictions (99.72%)
  (differ on 119 predictions — expected, different training data)
  XGB: PKL not available — parity check skipped

================================================================================
THESIS SUCCESS CRITERIA CHECK
================================================================================
  Accuracy > 95%            RF (PKL): PASS   RF (ONNX): PASS   XGB (ONNX): PASS
  Recall > 90%              RF (PKL): PASS   RF (ONNX): PASS   XGB (ONNX): PASS
  F1-Score > 90%            RF (PKL): PASS   RF (ONNX): PASS   XGB (ONNX): PASS
  FPR < 2%                  RF (PKL): PASS   RF (ONNX): PASS   XGB (ONNX): PASS
  Avg Latency < 100ms       RF (PKL): PASS   RF (ONNX): PASS   XGB (ONNX): PASS
  Model < 2MB               RF (PKL): FAIL   RF (ONNX): PASS   XGB (ONNX): PASS

Benchmark complete.