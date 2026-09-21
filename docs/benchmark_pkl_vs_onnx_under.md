======================================================================
BENCHMARK: PKL vs ONNX × RF vs XGBoost (All 4 Models)
Protocol: common
======================================================================

[1] Loading dataset...
    Protocol: common (sampling.py) — full dataset 450176 rows
    Test set: 90036 URLs (69148 legit, 20888 phishing)

[2] Loading PKL models...
    Loading RF pkl...
    RF pkl type: Pipeline
    Loading XGBoost pkl...
    XGB pkl keys: ['model', 'scaler', 'selector_kbest', 'important_mask', 'pca', 'feature_extractor', 'best_params', 'feature_counts']

[3] Loading ONNX models...

[4] RF (Pickle) — predicting...
    Measuring latency (1000 samples, full pipeline)...
    Done. Total predict time: 5.8s

[5] XGBoost (Pickle) — extracting features...
    Feature extraction: 8.9s
    Predicting...
    Measuring latency (1000 samples, full pipeline)...
    Done. Predict time: 0.3s

[6] RF (ONNX) — extracting features...
      20000/90036...
      40000/90036...
      60000/90036...
      80000/90036...
      90036/90036...
    Features: (90036, 1509), extraction: 401.7s
    Predicting (batch)...
    Measuring latency (1000 samples, full pipeline + model-only)...

[7] XGBoost (ONNX) — extracting features...
      20000/90036...
      40000/90036...
      60000/90036...
      80000/90036...
      90036/90036...
    Features: (90036, 151), extraction: 399.5s
    Predicting (batch)...
    Measuring latency (1000 samples, full pipeline + model-only)...

================================================================================
RESULTS
================================================================================

--- RF (PKL) ---
              precision    recall  f1-score   support

  Legitimate     0.9911    0.9991    0.9951     69148
    Phishing     0.9970    0.9702    0.9834     20888

    accuracy                         0.9924     90036
   macro avg     0.9940    0.9846    0.9892     90036
weighted avg     0.9924    0.9924    0.9924     90036


--- RF (ONNX) ---
              precision    recall  f1-score   support

  Legitimate     0.9911    0.9991    0.9951     69148
    Phishing     0.9970    0.9705    0.9835     20888

    accuracy                         0.9925     90036
   macro avg     0.9940    0.9848    0.9893     90036
weighted avg     0.9925    0.9925    0.9924     90036


--- XGB (PKL) ---
              precision    recall  f1-score   support

  Legitimate     0.9966    0.9965    0.9965     69148
    Phishing     0.9883    0.9887    0.9885     20888

    accuracy                         0.9946     90036
   macro avg     0.9924    0.9926    0.9925     90036
weighted avg     0.9946    0.9946    0.9946     90036


--- XGB (ONNX) ---
              precision    recall  f1-score   support

  Legitimate     0.9950    0.9976    0.9963     69148
    Phishing     0.9919    0.9833    0.9876     20888

    accuracy                         0.9943     90036
   macro avg     0.9934    0.9905    0.9919     90036
weighted avg     0.9943    0.9943    0.9943     90036

Confusion Matrix (RF (PKL)):  TN= 69087 FP=    61 | FN=   623 TP= 20265
Confusion Matrix (RF (ONNX)):  TN= 69086 FP=    62 | FN=   617 TP= 20271
Confusion Matrix (XGB (PKL)):  TN= 68903 FP=   245 | FN=   237 TP= 20651
Confusion Matrix (XGB (ONNX)):  TN= 68980 FP=   168 | FN=   348 TP= 20540

================================================================================
COMPARISON TABLE (All 4 Models)
================================================================================
Metric                        RF (PKL)    RF (ONNX)    XGB (PKL)   XGB (ONNX)
---------------------------------------------------------------------------
Accuracy                        0.9924       0.9925       0.9946       0.9943
Precision (Phish)               0.9970       0.9970       0.9883       0.9919
Recall (Phish)                  0.9702       0.9705       0.9887       0.9833
F1-Score (Phish)                0.9834       0.9835       0.9885       0.9876
False Positive Rate             0.0009       0.0009       0.0035       0.0024
---------------------------------------------------------------------------
Avg Latency (ms)                24.630        4.988        4.237        6.856
P95 Latency (ms)                25.935        5.616        5.158        7.614
Max Latency (ms)               117.856        6.960      134.669      112.385
Avg Model-only (ms)                  -        0.048            -        0.070
---------------------------------------------------------------------------
Model Size (KB)                   4320          638         4672         1907
Process Memory (MB)             4325.6
Test Samples                     90036

================================================================================
PKL vs ONNX AGREEMENT
================================================================================
  Note: the ONNX feature path mirrors utils.js and filters empty tokens,
  while sklearn counts them ('' is in the vocabulary), so tiny divergences
  (~0.2%) are expected for XGBoost; RF is typically identical.
  RF:  90025/90036 predictions agree (99.9878%)
       (if the two files are different variants, differences are expected — see MODELS.md)
  XGB: 89790/90036 predictions agree (99.7268%)
       (small divergence expected from the empty-token emulation described above)

================================================================================
THESIS SUCCESS CRITERIA CHECK
================================================================================
  Accuracy > 95%            RF (PKL): PASS   RF (ONNX): PASS   XGB (PKL): PASS   XGB (ONNX): PASS  
  Recall > 90%              RF (PKL): PASS   RF (ONNX): PASS   XGB (PKL): PASS   XGB (ONNX): PASS  
  F1-Score > 90%            RF (PKL): PASS   RF (ONNX): PASS   XGB (PKL): PASS   XGB (ONNX): PASS  
  FPR < 2%                  RF (PKL): PASS   RF (ONNX): PASS   XGB (PKL): PASS   XGB (ONNX): PASS  
  Avg Latency < 100ms       RF (PKL): PASS   RF (ONNX): PASS   XGB (PKL): PASS   XGB (ONNX): PASS  
  Model < 2MB               RF (PKL):  n/a   RF (ONNX): PASS   XGB (PKL):  n/a   XGB (ONNX): PASS  

Benchmark complete.
