# LVLM Evaluation Dataset for Tunnel Geology

* [中文](./readme-ZH.md)

<image src="./images/Geo-LVLM.png" width="200" height="200" />

## Evaluation Dataset
* **Common Scene**: Sourced from the [CogVLM-SFT-311K](https://github.com/THUDM/CogVLM/blob/main/dataset.md) dataset.
* **Complex Reasoning**: Tasks that require multi-faceted analysis of the excavation face, integrating geological, structural, and operational data.
* **Single Feature Judgment**: Focused on specific, isolated aspects of the tunnel construction process, such as identifying rock types or evaluating structural integrity.
* **Support Params QAs**: Question-Answer pairs related to support parameters in tunnel construction, aiding in the understanding of critical engineering decisions.
* **Tunnel Knowledge QAs**: Question-Answer pairs based on tunnel-related knowledge, facilitating deeper insights into construction methodologies and safety protocols.

### 1. Dataset Composition
- **Images**: The dataset comprises 281 high-resolution images captured from active tunnel construction sites. These images provide a comprehensive visual record of various stages and elements of the construction process, offering valuable insights into real-world scenarios.
- **Knowledge**: Included are 197 meticulously curated pieces of tunnel-related knowledge, encompassing technical specifications, safety guidelines, construction methodologies, and other pertinent information. This knowledge base is essential for informed decision-making and advanced analysis.

### 2. Vision Data Classification
- **Single Feature Judgment**: This category involves tasks that concentrate on a singular, specific aspect of the excavation face exposure during tunnel construction. For instance, identifying the type of rock or assessing the condition of a particular structural component. 
- **Complex Reasoning**: This category entails more sophisticated analyses that require the integration of multiple data points. Examples include identifying potential risks and conducting comprehensive assessments of surrounding rock conditions. 
### 3. Tunnel Knowledge
The knowledge base is derived from the **Tunnel Design-Construction Code**, ensuring that all information is aligned with industry standards and best practices. 

### 4.Comments
* Some of the data is generated using ChatGPT and may contain errors. Please do not use it directly in actual production.
* This dataset is designed to provide support for the evaluation and research of VLLM in the field of tunnel geology.

## Citation
```bibtex
@misc{sungatetop,
      title={LVLM Evaluation Dataset for Tunnel Geology}, 
      author={Baolin Chen and Jiawei Xie},
      howpublished={\url{https://github.com/sungatetop/geology-lvlm.git}},
      year={2025},
}
```

