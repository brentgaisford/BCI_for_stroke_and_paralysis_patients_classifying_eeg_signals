# Using neural networks to improve the accuracy of EEG classification

### Contents:
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Original Study Details](#original-study)
- [Notebooks and scripts created](#Notebooks-and-scripts-created)
- [Dependencies](#dependencies)
- [Sources](#Sources)

### Problem Statement

The goal of this project is to replicate, and with luck, improve on the results of a scientific study which sought to create a binary classifier for EEG recordings from nine study participants with significant central nervous system damage (either spinal cord injuries or large strokes). The participants each performed a series of trials, completing five different mental tasks forty times each while EEG recordings were taken, and then returning to make another session of EEG recordings doing the same tasks several days later. The goal of the original study was to identify which two mental tasks created the most differentiable signal for a binary classifier and optimize the accuracy of the classifier, using data from session 1 as a training set to predict which mental tasks the participants were performing on day 2. 

I will seek to expand on those results by examining alternative preprocessing of the EEG data and using neural networks to improve model accuracy. Additionally, with an eye toward creating software in future BCI devices, I will not be doing any manual dropping of trials or providing other human input into the data. Instead choices to discard input must be made algorithmically. Additionally, I will not be using any data from day 2 to retrain the model - although that certainly harmed model accuracy.

### Executive Summary

The original study was able to achieve 82% average accuracy when discriminating between the two most-discernable trial types on day 2. Their most discernable trial pairing was word association vs imagined hand movements.

In my modeling, I was able to achieve XXX



A number of next steps are proposed to improve model accuracy:

1. Utilize Riemannian geometry based methods. In the last few years, such methods have often outperformed CSP-based methods at EEG classification.
2. Use the first few trials of session 2 to rebias the models, as the original study authors did. In the long term, I think BCI devices will be far more successful if frequent retraining sessions are not required, but a minimum level of accuracy is also required to make such devices into useful tools for those with disabilities. Short daily (or more frequenct) re-training sessions appear to be the current best approach to making the technology accurate enough on an ongoing basis to be broadly useful.  

### Original Study

The original study was conducted by Reinhold Scherer, Josef Faller, Elisabeth V. C. Friedrich, Eloy Opisso, Ursula Costa, Andrea Kübler, and Gernot R. Müller-Putz and published in May 2015.

The study article can be found here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4436356/

Additionally, the full dataset of EEG recordings collected in the study can be found on this page, listed under "13. Individual imagery (004-2015'": http://bnci-horizon-2020.eu/database/data-sets

The five tasks the trial participants were asked to complete were mental word association, mental subtraction, spatial navigation, right hand motor imagery, and feet motor imagery. EEG recordings were taken with 30 electrode channels using standard 10-20 placement and sampled at 256Hz.

### Notebooks and Scripts Created

X notebooks and Y scripts were created in this project and are included in this repository. This is the suggested order to replicate the project:

1. Download the original study data from http://bnci-horizon-2020.eu/database/data-sets and save in the 'data' folder
2. data_ingester.py
    - This script is called in every notebook throughout the project
    - Extracts the data from the individual .mat files, transforms, and loads into MNE Python raw objects
3. Preprocessing grid search and CNN modeling
    - Grid searching for ideal preprocessing settings to improve signal quality
    - Visualizations of a number of those settings in action
    - Creating level 1 CNN models
4. Selecting CSP parameters and modeling
    - Grid searching for ideal CSP parameters
    - Creating level 1 LDA and neural networks using CSP data
5. Ensemble modeling
    - Combining CNN and CSP models into final, customized model for each participant

### Dependencies

File to replicate conda environment saved in repo at requirements.txt. Packages used:

- 

### Sources

1. The original study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4436356/#pone.0123727
2. The dataset, listed under "13. Individual imagery (004-2015'": http://bnci-horizon-2020.eu/database/data-sets
3. MNE Python's documentation: https://mne.tools/stable/index.html
4. The winning entry to an EEG classification on challenge: https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge
5. This medium article on using CNNs to interpret EEG data: https://justlv.medium.com/using-ai-to-read-your-thoughts-with-keras-and-an-eeg-sensor-167ace32e84a
6. This study on using Riemann geometry and rebiasing models to improve performance: https://www.frontiersin.org/articles/10.3389/fnhum.2021.635653/full
