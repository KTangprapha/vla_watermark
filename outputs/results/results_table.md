# VLA Watermark Evaluation Results

| Env    | Gen Method        | Trigger        | Clean SR | WM SR | Act Dev | TPR(clean) | TPR(txt-only) | TPR(vis-only) | TPR(full) | AUC(full) | COS(full) |
| ------ | ----------------- | -------------- | -------- | ----- | ------- | ---------- | ------------- | ------------- | --------- | --------- | --------- |
| vmas   | watermark_wrapper | semantic       | 1.000    | 1.000 | 0.103   | 0.300      | 0.200         | 0.300         | 0.700     | 0.780     | -0.015    |
| vmas   | watermark_wrapper | neuro_symbolic | 1.000    | 1.000 | 0.051   | 0.300      | 0.200         | 0.300         | 0.300     | 0.480     | -0.103    |
| libero | watermark_wrapper | semantic       | 1.000    | 1.000 | 0.024   | 0.100      | 0.100         | 0.100         | 1.000     | 0.800     | 0.054     |
| libero | watermark_wrapper | neuro_symbolic | 1.000    | 1.000 | 0.022   | 0.100      | 0.100         | 0.100         | 0.100     | 0.380     | 0.042     |
| vmas   | stainlock         | semantic       | 1.000    | 0.800 | 0.534   | 0.200      | 0.300         | 0.200         | 0.000     | 0.160     | 0.000     |
| vmas   | stainlock         | neuro_symbolic | 1.000    | 1.000 | 0.051   | 0.200      | 0.300         | 0.200         | 0.200     | 0.440     | 0.000     |
| libero | stainlock         | semantic       | 1.000    | 0.000 | 0.388   | 0.100      | 0.100         | 0.100         | 1.000     | 0.800     | 0.000     |
| libero | stainlock         | neuro_symbolic | 1.000    | 1.000 | 0.022   | 0.100      | 0.100         | 0.100         | 0.000     | 0.370     | 0.000     |
