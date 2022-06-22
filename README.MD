|WHAT'S GOING WELL/STRUGGLES|DEVELOPMENT PLAN|INSTRUCTOR FEEDBACK|
|---------------------------|----------------|-------------------|
|Selected, installed, and started using python package for managing EEG data, called MNE|N/A||
|Got all my data downloaded and can open it in MNE|N/A||
|Tagged data properly to turn it into variables & events|N/A||
|Have some early visualizations, but the data is too jumpy to be comprehensible|Been reading about methods for filtering EEG data - or smoothing / averaging, see final bullet point||
|Haven't quite cracked the code on MNEs built in scalers|Need to spend more time experimenting with scaling the data, but could also scale the data with sci kit learn outside of MNE||
|Need to reduce the number of data points per trial (256 Hertz times ten second trials is 2,560 samples for each of 30 channels, too many for deep learning I understand)|Read about this in EEG, but my first thought is to take the average of each 10 readings for each channel||
