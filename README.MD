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

I will seek to expand on those results by examining alternative preprocessing of the EEG data and using a number of models, combined . Additionally, with an eye toward creating software in future BCI devices, I will not be doing any manual dropping of trials or providing other human input into the data. Instead choices to discard input must be made algorithmically. Additionally, I will not be using any data from day 2 to retrain the model - although that certainly harmed model accuracy.

### Executive Summary

The original study was able to achieve 75% average accuracy when discriminating between the two most-discernable trial types on day 2. Their most discernable trial pairing was word association vs imagined hand movements.

After some early testing on general models with lackluster results, I decided to create individualized models for each subject. I used an ensemble approach, and created four level 1 models for each subject, which then feed into an ensemble neural network to make the final predictions on the unseen data from day 2.

First, I created a tool to grid search for the best preprocessing parameters for each subject (e.g., high and lowpass filtering, channels to drop, etc) using MNE for python to perform those signal processing steps.

Then, I created four models for each individual, all of which were designed to perform at approximately 75-80% accuracy on the training data from session 1 to avoid overfitting. EEG readings from the same person have relatively large shifts between days (and even within the same day), so given the challenge of making predictions on day 2 data using only data from day 1, it was important to avoid too much bias in the models. Given the number of models I was creating, I spun up an AWS instance to be able to create multiple models at a single time.

Finally, I created an ensemble neural network for each subject which took in the outputs of the four L1 models and used them to come to a final prediction for each sample. 

Unforunately, my ensemble model with all four L1 models still ended up far too tuned toward session 1, and only performed at 55% percent accuracy, with the distribution of true positive vs true negative predictions for each subject showing that the model had been skewed significantly by the shift in EEG readings in session 2 - particularly for the subjects whose LDA models had performed the best in training and testing on session 1 data. I achieved an accuracy of only 55% on the unseen data from session 2 at this point.

A number of next steps are proposed to improve model accuracy:

1. Utilize Riemannian geometry based methods. In the last few years, such methods have often outperformed CSP-based methods at EEG classification.
2. Use the first few trials of session 2 to rebias the models, as the original study authors did. In the long term, I think BCI devices will be far more successful if frequent retraining sessions are not required, but a minimum level of accuracy is also required to make such devices into useful tools for those with disabilities. Short daily (or more frequenct) re-training sessions appear to be the current best approach to making the technology accurate enough on an ongoing basis to be broadly useful.
3. Improve the accuracy of the CNN models by resampling the data from session 1 at different time periods to get the models to see more variability
4. Try to use generalized CNN models instead of individually created CNN models
5. Rerun analysis with fewer dropped channels

### Original Study

The original study was conducted by Reinhold Scherer, Josef Faller, Elisabeth V. C. Friedrich, Eloy Opisso, Ursula Costa, Andrea Kübler, and Gernot R. Müller-Putz and published in May 2015.

The study article can be found here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4436356/

Additionally, the full dataset of EEG recordings collected in the study can be found on this page, listed under "13. Individual imagery (004-2015'": http://bnci-horizon-2020.eu/database/data-sets

The five tasks the trial participants were asked to complete were mental word association, mental subtraction, spatial navigation, right hand motor imagery, and feet motor imagery. EEG recordings were taken with 30 electrode channels using standard 10-20 placement and sampled at 256Hz.

### Notebooks and Scripts Created

5 notebooks and 1 script were created in this project and are included in this repository. This is the suggested order to replicate the project:

1. Download the original study data from http://bnci-horizon-2020.eu/database/data-sets and save in the 'data' folder
2. data_ingester.py
    - This script is called in every notebook throughout the project
    - Extracts the data from the individual .mat files, transforms, and loads into MNE Python raw objects
3. 1_Preprocessing grid search and CNN modeling - MOST UP TO DATE VERSION ON MY LOCAL MACHINE - BREAKS HALFWAY THROUGH
    - Grid searching for ideal preprocessing settings to improve signal quality
    - Visualizations of a number of those settings in action
4. 2_Selecting_CSP_parameters_and_LDA_modeling.ipynb
    - Finalizing grid search for ideal CSP parameters
    - Creating level 1 LDA and neural networks using CSP data
5. 3_individual_CNN_models.ipynb
    - Creating individualized CNN models for each subject
    - Selecting two models for each individual that did not appear likely to overfit
6. 4_NN_from_CSP_components
    - Using CSP transformed data to train shallow neural networks
    - Selecting one model for each individual, again trying to avoid overfitting
7. 5_Ensemble modeling - MOST UP TO DATE VERSION ON LOCAL MACHINE - GET IT READY TO TURN IN
    - Running all four L1 models for each individual
    - Feed those L1 models into an ensembly neural network

### Dependencies

File to replicate conda environment saved in repo at requirements.txt.

Alternatively, all libraries can be installed via pip. Conda tends to have problems with mne.

- MNE
- pandas
- numpy
- scipy
- scikit-learn
- tensorflow/keras
- matplotlib
- And several libraries included in base python
    - itertools, IPython.utils, ast

### Sources

1. The original study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4436356/#pone.0123727
2. The dataset, listed under "13. Individual imagery (004-2015'": http://bnci-horizon-2020.eu/database/data-sets
3. MNE Python's documentation: https://mne.tools/stable/index.html
4. The winning entry to an EEG classification on challenge: https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge
5. This medium article on using CNNs to interpret EEG data: https://justlv.medium.com/using-ai-to-read-your-thoughts-with-keras-and-an-eeg-sensor-167ace32e84a
6. This study on using Riemann geometry and rebiasing models to improve performance: https://www.frontiersin.org/articles/10.3389/fnhum.2021.635653/full
