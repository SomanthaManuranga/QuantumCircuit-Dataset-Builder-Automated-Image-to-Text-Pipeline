This project is a sophisticated data engineering and machine learning pipeline designed to bridge the gap between general-purpose image-to-text models and specialized technical domains.

Below is a structured GitHub README template you can use for the repository.

📌 Project Overview

State-of-the-art image-to-text models (like those trained on COCO) exhibit near-zero accuracy when interpreting specialized technical diagrams such as Quantum Circuits. This project addresses the lack of domain-specific training data by providing an automated pipeline to compile a high-quality dataset of quantum circuit images and metadata directly from arXiv "quant-ph" publications.

The pipeline automates the extraction, classification, and labeling of 250+ publication-quality quantum circuit images, enabling the fine-tuning of deep learning models for quantum research.

🚀 Key Features

Multi-Source Extraction: Combines HTML parsing (for clean metadata) and LaTeX source/ZIP files (for high-fidelity image retrieval).

SciBERT-Powered Classification: Uses a fine-tuned SciBERT model to distinguish between quantum circuits and non-quantum figures with 96% accuracy.

Automated Metadata Labeling: Extracts quantum gates (X, Y, Z, CNOT, etc.) and algorithms using regex-based natural language processing.

JSON-Linked Dataset: Every image is paired with a structured JSON file containing the arXiv ID, page number, figure number, gate list, and a cleaned descriptive caption.

🛠️ Methodology

The project evolved through three iterations to overcome the limitations of technical PDF parsing:

PDF-Heuristic (PyMuPDF): Failed due to inconsistent vector/raster rendering in research papers.

TeX-Source Parsing: Reliable for images but difficult for structured metadata scraping.

HTML-Based Hybrid (Final Approach): Utilizes arXiv’s HTML versions and Beautiful Soup for reliable caption-metadata linkage, combined with ZIP file extraction for original image files.

The Pipeline Flow:

Scrape: Programmatically download 1,000+ recent "quant-ph" papers.

Extract: Use Beautiful Soup to grab figure captions from HTML versions.

Classify: A SciBERT classifier identifies "Quantum Circuit" figures based on caption text.

Enrich: Parse captions for gates and specific algorithms (Grover’s, Shor’s, VQE, etc.).

Retrieve: Map classified figures back to the source ZIP files to extract the original PNG/PDF image.

📊 Dataset Structure

The final output consists of images paired with JSON metadata:

code
JSON
download
content_copy
expand_less
{
  "arxiv_number": "2502.19970",
  "figure_number": 11,
  "quantum_gates": ["RZZ", "CNOT", "H"],
  "quantum_problem": ["Feature Map", "ZZ Feature Map"],
  "description": "Circuit diagram for the 4-qubit ZZ feature map.",
  "text_positions": [184294, 184342]
}
📈 Results

Classification Accuracy: SciBERT achieved an F1-score of 0.9612, significantly outperforming standard BERT for scientific terminology.

Extraction Accuracy: Gate extraction and algorithm identification achieved an estimated 80–90% accuracy on standard quantum papers.

Final Dataset: 250 high-quality images ready for multimodal model training.

💻 Technical Stack

Languages: Python

Libraries: Beautiful Soup 4 (HTML Scraping), PyMuPDF (PDF Processing), Transformers/HuggingFace (SciBERT), Pandas, Regex.

Models: SciBERT (Scientific Bidirectional Encoder Representations from Transformers).

📝 References

PDFFigures 2.0: Mining Figures from Research Papers (Clark & Divvala).

DocVQA: A Dataset for VQA on Document Images (Mathew et al.).

SciBERT: A Pretrained Language Model for Scientific Text (Beltagy et al.).

Author

Somantha Manuranga
Master of Artificial Intelligence in Industrial Applications
OTH Amberg-Weiden
