# Template-Bank Pipeline for Compact Binary Pulsar Searches

Template-Bank Pipeline for Compact Binary Pulsar Searches
This repo is a wrapper to the Template-Bank Pipeline, a powerful tool to search for compact binary pulsars. This pipeline is based on the works of Messenger (2008), Harry (2009), Knispel (2011), Allen (2013), and Balakrishnan (2021). It is designed to find short orbital period binaries that may not be detected with acceleration or jerk searches, typically used as a second step after an acceleration search.

Features


Searches for compact binary pulsars coherently across 3 Keplerian parameters for circular orbit binaries and 5 Keplerian parameters for elliptical orbit binaries.

This is a C++ Cuda GPU Pulsar Search Pipeline and you will need access to Nvidia GPUs to use this. This pipeline has borrowed a lot of code from Peasoup written by Ewan Barr (MPIfR). You can find the original repo here that does an 1-D acceleration search. https://github.com/ewanbarr/peasoup

This repo is a work in progress. Eventhough the entire code works, I still need to add support for segmented searches and slurm commands to split the jobs in a compute node. The original repo which is well tested can be found here: https://github.com/vishnubk/5D_Peasoup

Getting Started

Prerequisites
Docker: Make sure you have Docker installed on your machine. If not, download and install it from Docker's official website.

Alternatively you can use singularity. You can download a compiled image using:

singularity pull docker://vishnubk/template_bank_software:latest

Edit the template_bank_search.cfg with your singularity image path, and your binary parameter search range.

Copy code
./run_template_bank_pipeline.sh [input_parameters]

Acknowledgments


Ewan Barr for Peasoup,
Messenger (2008),
Harry (2009),
Knispel (2011),
Allen (2013),
Balakrishnan (2021)






