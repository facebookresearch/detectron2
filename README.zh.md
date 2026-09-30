<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 是 Facebook 人工智能研究院（FAIR）推出的下一代计算机视觉算法库，提供了前沿顶尖的目标检测与图像分割算法。它是 [Detectron](https://github.com/facebookresearch/Detectron/) 与 [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/) 的下一代继承者。它为 Facebook（Meta）内部诸多计算机视觉研究项目与生产级业务应用提供了强力支持。

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>
<br>

## 深入了解 Detectron2

* 包含诸多新特性与算法能力，如全景分割（Panoptic Segmentation）、DensePose（人体姿态估计）、Cascade R-CNN、旋转边界框（Rotated Bounding Boxes）、PointRend、DeepLab、ViTDet、MViTv2 等。
* 作为基础核心库，支持在其之上构建各类[前沿研究项目（projects/）](projects/)。
* 模型可导出为 TorchScript 格式或 Caffe2 格式，便于工业级部署。
* 训练速度大幅提升（详见[基准评测 Benchmark](https://detectron2.readthedocs.io/notes/benchmarks.html)）。

欢迎阅读我们的[官方博客文章](https://ai.meta.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)查看更多示例演示。
欢迎阅读此篇[深度专访](https://ai.meta.com/blog/detectron-everingham-prize/)，了解 Detectron2 背后的研发故事。

## 安装指南

详见[安装说明文档](https://detectron2.readthedocs.io/tutorials/install.html)。

## 快速入门

详见 [Detectron2 快速入门指南](https://detectron2.readthedocs.io/tutorials/getting_started.html)，以及交互式 [Colab 教程笔记本](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) 学习基本用法。

访问我们的[官方文档中心](https://detectron2.readthedocs.org)了解更多详细内容。
并可浏览 [projects/](projects/) 目录查看基于 Detectron2 构建的各类研究子项目。

## 模型库与基线结果

我们在 [Detectron2 模型库（Model Zoo）](MODEL_ZOO.md) 中提供了丰富的基线测试结果与可供直接下载的预训练模型权重。

## 开源许可证

Detectron2 基于 [Apache 2.0 开源许可证](LICENSE) 发布。

## 引用 Detectron2

如果您在学术研究中使用了 Detectron2，或引用了 [Model Zoo](MODEL_ZOO.md) 中发布的基准测试结果，请使用以下 BibTeX 格式进行引用：

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）翻译维护，最后同步更新于 2026年09月27日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
