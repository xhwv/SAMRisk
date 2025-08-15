Overview
SAM-Risk is a novel framework for glioma survival risk prediction that leverages a customized SAM-Med3D model with multi-view representation fusion and clinical knowledge-based age-grade stratified loss. This framework effectively integrates multimodal MRI, handcrafted radiomics (HCR), and clinical features to improve prognostic accuracy, outperforming several state-of-the-art survival risk prediction methods.

Gliomas are the most common malignant brain tumors in adults, and accurate survival risk prediction is crucial for personalized treatment. While molecular biomarkers offer prognostic value, their invasive and costly nature limits widespread use. SAM-Risk addresses this by utilizing non-invasive MRI data, radiomics, and clinical features, combined with advanced deep learning techniques, to achieve robust prognosis prediction.
Key Contributions
Multi-view Early Fusion Module: Transforms 1D handcrafted radiomics and clinical features into 3D representations, fusing them with multimodal MRIs at multiple scales to explore complementary information between views.
Customized SAM-Med3D: Fine-tunes the SAM-Med3D foundational model using LoRA (Low-Rank Adaptation) and a disparity function to focus on spatially significant features, with a feature refinement module to explore inter-channel relationships.
Age-Grade Stratified Loss: Incorporates clinical prior knowledge based on the RTOG 9802 standard, designing a loss function that accounts for survival risk differences between high-grade and low-grade gliomas across different age groups.
Datasets
SAM-Risk is validated on two publicly available datasets:

UCSF-PDGM Dataset
Access: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
BraTS2020 Dataset
Access: https://www.med.upenn.edu/cbica/brats2020/data.html
Requirements
To run SAM-Risk, the following dependencies are required:

torch>=1.10.0
monai>=0.9.0
torchvision>=0.11.0
scipy>=1.7.0
numpy>=1.21.0
pickle5>=0.0.11

Install dependencies using pip:

pip install torch>=1.10.0 monai>=0.9.0 torchvision>=0.11.0 scipy>=1.7.0 numpy>=1.21.0 pickle5>=0.0.11

Usage

Data Preparation
Download the UCSF-PDGM or BraTS2020 dataset from the provided links.
Organize the dataset into the appropriate directory structure (see dataset documentation for details on file formatting).
Ensure the dataset includes multimodal MRI scans, handcrafted radiomics features, and clinical features (e.g., age, gender, tumor grade).
Training
To train SAM-Risk on the UCSF-PDGM dataset:
python main.py --dataset ucsf-pdgm --batch-size 8 

To train on the BraTS2020 dataset:
python main.py --dataset brats2020 --batch-size 8 

