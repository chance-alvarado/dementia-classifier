# dementia-classifier

Application of common supervised learning algorithms to predict dementia.

**View the properly rendered notebook with *nbviewer* [here](https://nbviewer.jupyter.org/github/chance-alvarado/dementia-classifier/blob/master/dementia_classifier.ipynb).**

---

## Introduction

Dementia is a blanket term describing multiple symptoms of cognitive decline. While common among aging individuals, dementia is not a normal part of the aging process. Being able to identify and subsequently diagnosis patients earlier into the onset of dementia allows healthcare professionals to better apply intervention techniques.

---

## Emphasis

The notebook emphasizes the following skills:
  * Creating and merging DataFrames with different columnar data using _Pandas_.
  * Dealing with missing and ambiguous data using _Pandas_.
  * Produce useful and visually interesting plots using _Matplotlib_ and _Seaborn_.
  * Applying common predictive models to real world data using _Scikit-learn_.
  * Analyze the effectiveness of these models through common statistical plots created with _Matplotlib_ and _Seaborn_.

---

## Prerequisites

This repository is written using [Python](https://www.python.org/) v3.7.4 and [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/) v6.0.3. 

The following packages are recommended for proper function:

Requirement | Version
------------|--------
[Pandas](https://pandas.pydata.org/) | 1.0.1
[Matplotlib](https://matplotlib.org/) | 3.1.3
[Numpy](https://numpy.org/) | 1.18.1
[Seaborn](https://seaborn.pydata.org/) | 0.10.0
[Scikit-learn](https://scikit-learn.org/) | 0.22.1

Installation instructions for these packages can be found in their respective documentation.

---

## Data

The data used in this analysis is divided among two csv files:

- `oasis_cross_sectional`
  -  This datasets contains a cross-sectional collection of 416 subjects and their respective medical attributes.
  
- `oasis_longitduinal`
  - This datasets contains a longitudinal collection of 150 subjects and their respective medical attributes. For this analysis only the first visit observation of each subject will be used. 

This data has been collected by the Open Access of Imaging Studies (OASIS) project. More information about this project can be found [here](http://www.oasis-brains.org/). 

- Acknowledgments: 
  - OASIS: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382
  - OASIS: Longitudinal: Principal Investigators: D. Marcus, R, Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382

---

## Cloning

Clone this repository to your computer [here](https://github.com/chance-alvarado/dementia-classifier/).

---

## Author

- **Chance Alvarado** 
    - [LinkedIn](https://www.linkedin.com/in/chance-alvarado/)
    - [GitHub](https://github.com/chance-alvarado/)

---

## License

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
