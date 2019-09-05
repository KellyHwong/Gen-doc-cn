# MIT Gen 文档翻译

| 待翻译文档       | 进度 |
| ---------------- | ---- |
| Gen 中的建模简介 | 22%  |

## 译文阅读指引

在 MIT-Gen 中，Gen 是 Generative Model 中的 Gen 前缀。Gen 是一个生成模型的工具包。

部分**计算机名词**选择不翻译，比如 trace。trace 意思是跟踪，作为名词意思是轨迹、踪迹。所以 Gen 提供了一组关于跟踪随机变量的 API。trace 作为术语也不翻译了。

## 教程：Gen 中的建模简介

进度 **22%**。

Gen 是概率建模和推理的多范式平台。Gen 支持多种建模和推理工作流程，包括：

- 使用蒙特卡罗，变分，EM 和随机梯度技术在生成模型中进行无监督学习和后验推理。

- 有条件推理模型的监督学习（例如监督分类和回归）。

- 混合方法，包括摊销推理/推理编译，变分自动编码器和半监督学习。

在 Gen 中，概率模型（生成模型和条件推理模型）表示为*generative functions*。Gen 提供了一种用于定义生成函数的内置建模语言（Gen 也可以扩展为支持其他建模语言，但本教程不包含此内容）。本教程介绍了 Gen 的内置建模语言的基础知识，并说明了该语言提供的几种建模灵活性，包括：

- 使用随机分支和函数抽象来表示多个模型时，哪个模型更适合的不确定性。

- 用无限数量的参数表示模型（'贝叶斯非参数'模型）。

该笔记本使用简单的通用推理算法进行后验推理，并显示了一些应用于简单模型的推理示例。笔记本还介绍了一种通过从推断参数预测新数据并将此数据与观察数据集进行比较来验证模型和推理算法的技术。

本教程不涉及*自定义推理编程*，这是 Gen 的一项关键功能，用户可以实现专门针对其概率模型的推理算法。推理编程对于有效地获得准确的后验推理很重要，将在后面的教程中介绍。此外，本教程并未详尽地介绍建模语言的所有功能 - 还有一些功能和扩展可提供此处未涉及的改进性能。
