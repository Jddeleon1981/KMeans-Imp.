# KMeans-Imp.
This project contains a lecture notebook meant to introduce how to use my k-means implmentation as well as the math behind it. On top of that it includes a julia file named methods that contains all of the behind the scenes functions that makes the implmentation work. This was all made for the Final Project in Math 157 WI23 at UCSD.

## Table of Contents

- [Project Description](#project-description)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Description

This project is a beginner-level data science/machine learning project that demonstrates the use of Julia for analyzing and modeling data. The project allows for the user to cluster data among 2 seperate features to try and unveil any hidden patterns in the data. It's important to note that the project is still limited in some senses in that the user is restricted to only clustering among two features and the main distance function used in this implmentation is the Euclidean Distance. On top of clustering, this package also allows for us to measure how appropiate this algorithm might even be in that case with quality checks like the elbow plot and Silhouette score.

## File Descriptions

- `FinalLecture.ipynb`: This file contains the final lecture notebook for the project, which covers the basics of data exploration, visualization, preprocessing, and modeling using Julia.
- `methods.jl`: This file contains the code for the machine learning models used in the notebook.

## Usage

To run the notebook, you will need to install Julia and the necessary packages. The FinalLecture notebook has all additional packages included, so if any particular one isn't added to your particular enviroment you can uncomment and run that initial cell again.

The methods file is mainly used to supplement the notebook file so we don't need to worry about running that file individually.

## Contributing

Contributions to this project are welcome. If you find a bug or have a feature request, please open an issue or submit a pull request.

