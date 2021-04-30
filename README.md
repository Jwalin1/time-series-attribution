# time-series-attribution

**Type:**
Student Project SS21

**Topci:**
Understanding the applicability of attribution methods on time-series.

**Description:**  
The amount of existing time-series classification tasks increases every day, and deep learning approaches have shown impressive results among different classification tasks [1]. However, there is a limited amount of deep neural networks used in real-world problems due to the lack of their interpretability. The goal of this project is to transfer attribution methods used in the image domain and test their performance on time-series classification tasks. This requires to understand and modify the existing methods and use appropriate measurements to validate their performance.
[1] Bagnall, A., Lines, J., Bostrom, A., Large, J., & Keogh, E. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data Mining and Knowledge Discovery, 31(3), 606-660.

==================================================

**Instructions:**
- I will highly appreciate if you use [Boards](https://git.opendfki.de/mercier/time-series-attribution/-/boards) in the repository. Your tasks are already added to **"To Do"**.
- Before you start working on a specific task, please move it to **"Doing"**. And once a task is done, please move it to **"Closed"**.
- As soon as a task is finished, push the code to the repository with self-explanatory details in the commit. Please do **not** push work in progress to the repository.
- Do **not** push data files (i.e. dataset, embeddings, models etc.) to the repository as it is only intended for source files. When you want to share data files, use this [Cloud directory](https://cloud.dfki.de/owncloud/index.php/s/N6sAe7DJRPPksP6)
- Document the code along with your implementation because it becomes very difficult to do it at the end.
- Create a **"src"** folder for the code, **"images"** folder for plots, and a **"statistics"** folder for evaluation files. This structure will help you to organize the files within the repository.
- Note down your findings in this readme. Use the sections below that are created for each Milestone. Only summarize the finding. For deatails you can create individual files and link them (e.g. M1_readme.md).

**Dataset sources:**  
Huge Collection of time-series datasets (UCR / UEA): www.timeseriesclassification.com/  
Most famous time-series dataset archive (UCR): www.cs.ucr.edu/~eamonn/time_series_data_2018/  
Snythetic Anomaly Detection dataset: www.bit.ly/2UNk0Lo


**Suggested datasets:**  
- Character Trajectories: Back-projection is very interpretable. (Main focus)
- Synthetic Anomaly Dectection: Classification task is interpretable.
- FordA: Wide spread uni-variate datset.
- Electric Devices: Provides huge discrepancy between the classes.
- Daily and Sport Activites: Real-world data that is complex.



==================================================

# Project plan

## Milestone 1 : [Generic Expeiment Setup](https://git.opendfki.de/mercier/time-series-attribution/-/milestones/1)

## Milestone 2 : [Attribution Methods](https://git.opendfki.de/mercier/time-series-attribution/-/milestones/2)

## Milestone 3 : [Approach Evaluation](https://git.opendfki.de/mercier/time-series-attribution/-/milestones/3)

## Milestone 4 : [Impact of Randomization](https://git.opendfki.de/mercier/time-series-attribution/-/milestones/4)

## Milestone 5 : [Deployment](https://git.opendfki.de/mercier/time-series-attribution/-/milestones/5)
