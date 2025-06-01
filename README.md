# [**A Global Object-Oriented Dynamic Network for Low-Altitude Remote Sensing Object Detection**](http://doi.org/10.1038/s41598-025-02194-6)

Official PyTorch implementation of GOOD-Net.

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
With advancements in drone control technology, low-altitude remote sensing image processing holds significant potential for intelligent, real-time urban management. However, achieving high accuracy with deep learning algorithms remains challenging due to the stringent requirements for low computational cost, minimal parameters, and real-time performance. This study introduces the Global Object-Oriented Dynamic Network (GOOD-Net) algorithm, comprising three fundamental components: an object-oriented, dynamically adaptive backbone network; a neck network designed to optimize the utilization of global information; and a task-specific processing head augmented for detailed feature refinement. Novel module components, such as the ReSSD Block, GPSA, and DECBS, are integrated to enable fine-grained feature extraction while maintaining computational
and parameter efficiency. The efficacy of individual components in the GOOD-Net algorithm, as well as their synergistic interaction, is assessed through ablation experiments. Evaluation conducted on the VisDrone dataset demonstrates substantial enhancements. Furthermore, experiments assessing robustness and deployment on edge devices validate the algorithm’s scalability and practical applicability. Visualization methods further highlight the algorithm’s performance advantages. This research presents a scalable object detection framework adaptable to various application scenarios and contributes a novel design paradigm for efficient deep learning-based object detection.
</details>
## **Revised Note:**

GOOD-Net was previously designated MED-Net.

## Installation

```
pip install -e .
```

## Training

```
python train.py
```

## Acknowledgement

The code base is built with [Ultralytics](https://github.com/ultralytics/ultralytics).

We sincerely thank all those who contributed to this study.

## Citation

```
<<<<<<< HEAD
@Article{10.1038/s41598-025-02194-6,
=======
﻿@Article{10.1038/s41598-025-02194-6,
>>>>>>> b3b5afa7c866ce4bd39d1a3af5e17f81fd65b98b
author={Tang, Daoze and Tang, Shuyun and Wang, Yalin and Guan, Shaoyun and Jin, Yining},
title={A global object-oriented dynamic network for low-altitude remote sensing object detection},
journal={Scientific Reports},
year={2025},
month={May},
day={30},
volume={15},
number={1},
pages={19071},
issn={2045-2322},
doi={10.1038/s41598-025-02194-6},
url={https://doi.org/10.1038/s41598-025-02194-6}
<<<<<<< HEAD
=======
}
>>>>>>> b3b5afa7c866ce4bd39d1a3af5e17f81fd65b98b
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tdzdele/GOOD-Net&type=Date)](https://www.star-history.com/#Tdzdele/GOOD-Net&Date)
