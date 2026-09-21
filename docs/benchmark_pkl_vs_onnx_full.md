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
    Feature extraction: 8.4s
    Predicting...
    Measuring latency (1000 samples, full pipeline)...
    Done. Predict time: 0.3s

[6] RF (ONNX) — extracting features...
      20000/90036...
      40000/90036...
      60000/90036...
      80000/90036...
      90036/90036...
    Features: (90036, 1509), extraction: 397.1s
    Predicting (batch)...
    Measuring latency (1000 samples, full pipeline + model-only)...

[7] XGBoost (ONNX) — extracting features...
      20000/90036...
      40000/90036...
      60000/90036...
      80000/90036...
      90036/90036...
    Features: (90036, 149), extraction: 403.9s
    Predicting (batch)...
    Measuring latency (1000 samples, full pipeline + model-only)...

================================================================================
RESULTS
================================================================================

--- RF (PKL) ---
              precision    recall  f1-score   support

  Legitimate     0.9877    1.0000    0.9938     69148
    Phishing     1.0000    0.9587    0.9789     20888

    accuracy                         0.9904     90036
   macro avg     0.9938    0.9793    0.9863     90036
weighted avg     0.9905    0.9904    0.9903     90036


--- RF (ONNX) ---
              precision    recall  f1-score   support

  Legitimate     0.9872    1.0000    0.9935     69148
    Phishing     0.9999    0.9570    0.9780     20888

    accuracy                         0.9900     90036
   macro avg     0.9935    0.9785    0.9857     90036
weighted avg     0.9901    0.9900    0.9899     90036


--- XGB (PKL) ---
              precision    recall  f1-score   support

  Legitimate     0.9954    0.9987    0.9971     69148
    Phishing     0.9957    0.9848    0.9903     20888

    accuracy                         0.9955     90036
   macro avg     0.9956    0.9918    0.9937     90036
weighted avg     0.9955    0.9955    0.9955     90036


--- XGB (ONNX) ---
              precision    recall  f1-score   support

  Legitimate     0.9946    0.9986    0.9966     69148
    Phishing     0.9953    0.9820    0.9886     20888

    accuracy                         0.9947     90036
   macro avg     0.9950    0.9903    0.9926     90036
weighted avg     0.9947    0.9947    0.9947     90036

Confusion Matrix (RF (PKL)):  TN= 69147 FP=     1 | FN=   863 TP= 20025
Confusion Matrix (RF (ONNX)):  TN= 69146 FP=     2 | FN=   899 TP= 19989
Confusion Matrix (XGB (PKL)):  TN= 69060 FP=    88 | FN=   317 TP= 20571
Confusion Matrix (XGB (ONNX)):  TN= 69052 FP=    96 | FN=   377 TP= 20511

================================================================================
COMPARISON TABLE (All 4 Models)
================================================================================
Metric                        RF (PKL)    RF (ONNX)    XGB (PKL)   XGB (ONNX)
---------------------------------------------------------------------------
Accuracy                        0.9904       0.9900       0.9955       0.9947
Precision (Phish)               1.0000       0.9999       0.9957       0.9953
Recall (Phish)                  0.9587       0.9570       0.9848       0.9820
F1-Score (Phish)                0.9789       0.9780       0.9903       0.9886
False Positive Rate             0.0000       0.0000       0.0013       0.0014
---------------------------------------------------------------------------
Avg Latency (ms)                25.118        4.953        3.867        6.614
P95 Latency (ms)                26.413        5.552        4.256        7.380
Max Latency (ms)               199.420        6.851      207.855        8.595
Avg Model-only (ms)                  -        0.045            -        0.056
---------------------------------------------------------------------------
Model Size (KB)                   7529          730         7830         1931
Process Memory (MB)             4395.4
Test Samples                     90036

================================================================================
PKL vs ONNX AGREEMENT
================================================================================
  Note: the ONNX feature path mirrors utils.js and filters empty tokens,
  while sklearn counts them ('' is in the vocabulary), so tiny divergences
  (~0.2%) are expected for XGBoost; RF is typically identical.
  RF:  89999/90036 predictions agree (99.9589%)
       (if the two files are different variants, differences are expected — see MODELS.md)
  XGB: 89888/90036 predictions agree (99.8356%)
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
