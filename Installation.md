Installation
=============================
 
 
 - Download and Install Anaconda3
 - Create anaconda environement from anaconda-cloud
 - Open anaconda-navigator and lunch jupyter notebook
 
 -Step 1: Download and Install Anaconda3
-----------------------------------------
Download anaconda3 distribution compatible with your OS (Linux, windows, mac)
from their [website](https://www.continuum.io/downloads)
 
  
 -Step 2: Create anaconda environement from anaconda-cloud
-----------------------------------------------------------
In order to create an anaconda environement please follow the instruction below:

Open terminal console and type

```
#Username and password is required for this step to connect anaconda-cloud
anaconda login  

#create env in your local machine for windows
conda env create wail/syncpy 

#create env in your local machine for linux /mac
conda env create wail/syncpylinux
```


 

 -Step 3: Open anaconda-navigator and lunch jupyter notebook
----------------------------------------------------------
Now you have installed an environement that have all dependencies required for SyncPy to work( if the instruction above were successful). 
What remains is to test on a real example using jupyther notebook.

First open Anaconda-navigator, Choose SyncPy environement and then lunch jupyter notebook:

![Anaconda](/img/2.png)

