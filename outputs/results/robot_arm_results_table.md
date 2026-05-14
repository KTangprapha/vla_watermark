# VLA Watermark Evaluation Results

| Env       | Gen Method        | Trigger        | Clean SR | WM SR | Act Dev | TPR(clean) | TPR(txt-only) | TPR(vis-only) | TPR(full) | AUC(full) | COS(full) |
| --------- | ----------------- | -------------- | -------- | ----- | ------- | ---------- | ------------- | ------------- | --------- | --------- | --------- |
| robot_arm | watermark_wrapper | semantic       | 0.000    | 0.000 | —       | 0.200      | 0.100         | 0.100         | 1.000     | 0.800     | 0.028     |
| robot_arm | watermark_wrapper | neuro_symbolic | 0.000    | 0.000 | —       | 0.200      | 0.100         | 0.100         | 1.000     | 0.800     | 0.028     |
| robot_arm | stainlock         | semantic       | 0.000    | 0.000 | —       | 0.200      | 0.100         | 0.100         | 1.000     | 0.800     | 0.000     |
| robot_arm | stainlock         | neuro_symbolic | 0.000    | 0.000 | —       | 0.200      | 0.100         | 0.100         | 1.000     | 0.800     | 0.000     |
