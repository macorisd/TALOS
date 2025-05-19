# TALOS: A Modular and Automatic System for Open-Vocabulary Semantic Instance Segmentation

**TALOS** is a modular and extensible Computer Vision pipeline for performing **semantic instance segmentation** using an **open vocabulary** for semantic categories. Unlike conventional approaches (e.g., Detectron2) that are limited to a fixed set of object categories seen during training (such as those in the COCO dataset), TALOS identifies and segments object instances belonging to uncommon and diverse categories.

Many open-vocabulary detection and segmentation models require user inputs for semantic categories, which is impractical for automated applications like mobile robotics. TALOS addresses this limitation by automatically extracting semantic labels from images using large-scale models and then locating and segmenting the objects in the image.

The system is composed of three sequential stages: **Tagging**, **Location**, and **Segmentation**, each of which can be independently configured with state-of-the-art models.

[📄 Read the paper (Spanish PDF)](./docs/paper/talos_paper.pdf)

*This project is currently under development. Please check the **develop** branch for the latest updates.*

---

## Pipeline overview

TALOS takes an arbitrary number of **RGB images** as input and produces **instance-level segmentations** for each image, that include binary masks for each object instance, along with their corresponding bounding boxes and semantic labels.

The pipeline is designed to be modular, allowing for easy integration of new models and components. The three main stages of the pipeline are as follows:

### 1. Tagging
- **Description**: Extracts object category labels using large-scale models (LVLMs and/or LLMs).
- **Input**: RGB image.
- **Output**: List of semantic object categories (textual labels)
- **Tagging methods**:
  - **Direct Tagging**: Uses a Large Vision-Language Model (LVLM) to extract labels directly. A smaller and more specific model like RAM++ is suitable for this Tagging method too, but this is not as flexible as the LVLM approach.
  - **Tagging via LVLM Image Description and LLM Keyword Extraction**: Uses a LVLM to generate a description of the image, which is then processed by an LLM to extract keywords of the object categories that are present in the image description.

### 2. Location
- **Description**: Locates objects described by the category tags using a visual grounding model.
- **Inputs**:
  - RGB image
  - List of object labels (output from the Tagging stage).
- **Output**: List of bounding boxes with label and confidence.

### 3. Segmentation
- **Description**: Produces accurate instance segmentation masks using category-agnostic segmentation models.
- **Inputs**:
  - RGB image.
  - Located bounding boxes for each detected object (output from the Location stage).
- **Outputs**:
  - Binary masks, one per object instance.
  - Detections JSON that includes the semantic label, bounding box coordinates, location confidence score for each instance and mask ID.

---

## Integrated technologies and models

### Tagging
- **Direct Tagging**: 
  - Qwen
  - Gemma 3
  - MiniCPM
  - Recognize Anything Plus Model (RAM++)

- **Tagging via LVLM Image Description and LLM Keyword Extraction**:
  - **LVLM Image Description**:
    - Qwen
    - LLaVA
  - **LLM Keyword Extraction**:
    - DeepSeek

### Location
- Grounding DINO

### Segmentation
- Segment Anything Model 2 (SAM2)


---

## Installation and Usage

TODO

Consideraciones especiales:
- Añadir src del proyecto a PYTHONPATH.
- Si el modelo es de Ollama, hacer pull primero.
- Si el modelo es de HuggingFace y requiere token, hay que añadirlo a la variable de entorno `HUGGINGFACE_TOKEN`.
- Particularidades de cada modelo:
  - Ollama: hacer pull primero (LLaVA, DeepSeek)
  - HuggingFace: si requiere token, hay que añadirlo a la variable de entorno `HUGGINGFACE_TOKEN` (Gemma 3)
  - MiniCPM: requiere una versión muy concreta de transformers.

- Recomendaciones de qué incluir en el config.json de .vscode


---

## Pipeline evaluation and results

TODO (tabla)


---

## 🤝 How to contribute

Contributions are welcome!

- Fork the repository.
- Create a new branch: git checkout -b feature/my-feature.
- Make your changes.
- Push to your fork: git push origin feature/my-feature.
- Submit a pull request with a clear description of your changes.

All pull requests will be reviewed and require approval before being merged into the main branch.


--- 

## Citation

If you use TALOS in your research, please cite this repository as follows:

```bibtex
@misc{decena2025talos,
  author       = {Decena-Gimenez, Macoris},
  title        = {TALOS: A Modular and Automatic System for Open-Vocabulary Semantic Instance Segmentation},
  year         = {2025},
  howpublished = {\url{https://github.com/macorisd/TALOS}}
}
```

---


## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](./LICENSE) file for details.


---

## Author's note

Hi! I'm Maco 👋​

This project is being developed as part of my Bachelor's Thesis in Software Engineering, as a member of the MAPIR research group at the University of Málaga, Spain. It would not have been possible without the trust and support of my professors Javier and Raúl. Thank you for believing in me!


---

## Contact

If you have any questions, suggestions, or feedback, please reach out to me!

- Linkedin: [macorisd](https://www.linkedin.com/in/macorisd/)
- Email: [macorisd@gmail.com](mailto:macorisd@gmail.com)
