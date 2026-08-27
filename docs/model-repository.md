# 模型仓库约定

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

模型管理支持任意用户选择的仓库根目录，推荐结构：

```text
models_repository/
├── Project_A/
│   ├── yolov8x-pose-2026-07-29/
│   │   ├── args.yaml
│   │   ├── results.csv
│   │   ├── results.png
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   └── another-run/
└── Project_B/
    └── yolov8n-obb-run/
```

每个包含 `args.yaml` 或 `weights/` 的训练结果目录会被识别为一个模型记录。支持识别的常见模型格式包括 `.pt`、`.pth`、`.onnx`、`.engine`、`.plan`、`.xml/.bin`、`.tflite` 和 `.torchscript`。

模型详情中的训练数据来源优先从训练参数、`dataset.yaml` 和 `preparation_manifest.json` 解析。缺少这些文件时，部分字段会显示未知，不会要求用户手工维护一份模型 JSON 注册表。
