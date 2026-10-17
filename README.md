# AcousticTestingTools
A suite of programs to perform and post-process loudspeaker measurements. 

The repository consists of multiple programs in two directories:

**measurement_tools folder:**  
------------------------------------------------------------------------------------------------------
This folder contains the source code for two programs related to data acquisition and data measurement.  
* Probe_audio - this program provides information about the audio devices available on the local machine  
* DaDooRunRun - this is a response measurement program intended to measure the response of a DUT when given a test stimulus. Multiple data collection runs can be performed and averaged (e.g. time synchronous averaging) to improve data S/N. The stimulus is a windowed and filtered log-chirp with user-defined sweep rate. A pause between measurements of a user-defined mean duration, plus 10% random variation to reduce reinforcement of room echo. It produces raw measurement and stimulus data files that must be post-processed into the system impulse response and then into frequency domain response data using the tools in the post processing tools folder. Data is saved to WAV files in the FLOAT32 format.  

The source code has been built into executables that are available for Windows and Linux in the latest project release. See the RELEASES page to download them.

**post_processing_tools folder:**  
------------------------------------------------------------------------------------------------------
This folder contains the source code for several Python programs that are use to post-process raw measurement data into frequency domain data and produce FRD files.  
* H1mpulResper - converts raw measurement data produced by DaDooRunRun into the impulse response via an H1 estimator.   
* Time2Freq and NoGuiTime2Freq - converts windowed time domain impulse response data into frequency response data (SPL, phase). The resulting frequency response is saved to file in the familiar FRD file format.  
* GPDP - Groundplane Measurement Data Processing for data collected using the Lift and Tuck and Measure, Move-Move, Measure Techniques.  
* WAVplot - Plotting utility for up to 4 WAV files. Can simultaneously zoom all plots and save them to file as PNG or PDF files.  
* ImpAlign - A tool for measurements made with the Measure, Move-Move, Measure Technique. Aligns the impulse response produced in measurement 2 with the impulse response from measurement 1, fixing any small alignment errors.  

**Installation Instructions (Python and required modules)**  
------------------------------------------------------------------------------------------------------
The code for all of the data post-processing programs remains uncompiled. To run these programs you must install Python on your machine including the following modules:
* pandas  
* numpy  
* matplotlib  
* scipy

Instructions for installing Python and the required modules under Windows and Ubuntu Linux are provided in the docs folder.
