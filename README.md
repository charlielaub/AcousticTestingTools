# AcousticTestingTools
A suite of programs to collect and post-process loudspeaker measurements. 

The repository consists of multiple programs in two separate archives. See the latest release for download information.

**Measurement Tools:**  
------------------------------------------------------------------------------------------------------
Programs related to data acquisition and data measurement:  
* ProbeAudio - this program provides information about the audio devices available on the local machine  
* DaDooRunRun - this is a response measurement program intended to measure the response of a DUT when given a test stimulus. Multiple data collection runs can be performed and averaged (e.g. time synchronous averaging) to improve data S/N. The stimulus is a windowed and filtered log-chirp with user-defined sweep rate. It produces raw measurement and stimulus data files that must be post-processed into the system impulse response and then into frequency domain response data using the tools in the post processing tools folder. Data is saved to WAV files in the FLOAT32 format.
* DaDooImdImd - this program collects data under a multi-tone stimulus that can be post-processed into the intermodulation distortion profile for the DUT.

The source code for DaDooRunRun, DaDooImdImd, and ProbeAudio has been built into binary executable files for Windows and Linux. These may be found in the respective folders in the latest project release.  

**Data Post-processing Tools:**  
------------------------------------------------------------------------------------------------------
Python programs used to post-process raw measurement data into frequency domain data and produce FRD files and other info.  
* Time2Freq - converts windowed time domain impulse response data into frequency response data (SPL, phase) and harmonic distortion using the Farina method.
* H1_Time2Freq - converts windowed H1 estimated impulse response into  into frequency response data (SPL, phase). Can be run in an automated mode for batch processing multiple files.
* IMDanalysis - computes the intermodulation distortion response from the data produced by DaDooImdImd
* Fimpulse2CSD - generates the cumulative spectral decay as waterfall and heatmap plots from the time domain response produced by the Farina method.

**Plotting Utilities:**  
------------------------------------------------------------------------------------------------------
* WAVplot - Plotting utility for up to 4 WAV files. Can simultaneously zoom all plots and save them to file as PNG or PDF files.
* FRDplot - Plotting utility for any number of frequency responses from FRD files, into the same plot.
* BandKplot - Plot a single frequency response from an FRD file to a B&K style response chart, with a fixed 50dB y-axis span.

**Groundplane Measurement Data Processing Tools**  
------------------------------------------------------------------------------------------------------
These tools are intended to process data obtained using the techniques outlined in my whitepaper "New Approaches to Groundplane Measurements" that can be found in my file bank:
https://github.com/charlielaub/file_bank/tree/main/Groundplane%20Measurement%20Techniques
* GPDP - Groundplane Measurement Data Processing for data collected using the Lift and Tuck and Measure, Move-Move, Measure Techniques.  
* ImpAlign - A tool for measurements made with the Measure, Move-Move, Measure Technique. Aligns the impulse response produced in measurement 2 with the impulse response from measurement 1, fixing any small alignment errors.  

