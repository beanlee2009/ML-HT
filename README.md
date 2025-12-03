# ML-HT
This repo is for our paper "Explainable multimodal prediction model of hematologic toxicities to optimize neoadjuvant chemoradiotherapy for locally advanced rectal cancer：A multicenter retrospective study".
Terms:
Hematologic Toxicity（HT）; Neoadjuvant chemoradiotherapy (NACRT); local advanced rectal cancer (LARC)

HT endpoints：
HT grade was evaluted according to the National Cancer Institute Common Terminology Criteria for Adverse Events (CTCAE, version 5.0). The white blood cell count (WBC), hemoglobin count (HGB), absolute neutrophil count (ANC), platelet count (PLT) before NACRT were recorded as baseline values. The greatest grade toxicity during NACRT was recorded. The leukopenia is defined if WBC<3×109/L, while the neutropenia, anemia and thrombocytopenia, are defined if ANC <1.5×109/L, HGB <100g/L, and PLT <75×109/L, respectively.

Clinical factors：
Clinical factors includes the basic information of an individual patient, such as age, gender, smoking history, and the prescription of NACRT. Baseline of hematological parameters were collected from the initial blood inspection before NACRT, such as blood urea nitrogen (BUN), carcinoembryonic antigen (CEA).

PBM delineation：
The pelvic bone marrow (PBM) was divided into three subsites: (1) lumbosacral spine, extending from the superior border of the L5 vertebra to include the entire sacrum. (2) ilium, extending from the iliac crests to the superior border of the femoral heads; and (3) lower pelvis, extending from the femoral heads to the inferior border of the ischial tuberosities. Hence, three volumes of interest (VOIs) were included, i.e. BM of lumbosacral spine (LS), BM of ilium (IL), BM of lower pelvis (LP).

Radiomic features：
The radiomic features were extracted from 19 base images, including the original image, 10 Laplacian of Gaussian (LoG) filtered images and 8 wavelet filtered images. These extracted features constitute together into three main types: first-order, shape and texture features. In total, there are 432 first-order, 720 shape-based and 1024 texture features (includes: gray level co-occurrence matrix (GLCM), gray level dependence matrix (GLDM), gray level size zone matrix (GLSZM), gray level run length matrix (GLRLM), and neighboring gray tone difference matrix (NGTDM)), of all the 4 ROIs together. Radiomic features were extracted by the common-used Pyradiomics (version 3.0.1) package in Python.

Dosimetric features:
Dosimetric features were collected from dose-volume parameters (DVPs) in the radiotherapy plans. As for DVPs of a specific VOI, Vx means the relative volume that received a dose greater than x-Gy, and Dy means the absolute received dose greater than y% volume of interest. In addition, some special parameters Dmax, Dmin, Dmean, D95 and D98 were also included.

Radiomic and dosimetric score：
A least absolute shrinkage and seletion operator (LASSO) was applied to select the most representative features of dosimetric and radiomic features to conduct Rad_score and Dose_score by linear regression seperately. The linear models of Rad_score and Dose_score can be expressed as score=∑_i〖a+w_i×〗 f_i, where a is the intercept value, w_i is the weight of the feature f_i.
