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

# Activate env and install syncpy package
 source activate syncpylinux		 #for linux/mac
 activate  syncpy 				#for windows

# installing syncpy package in anaconda environement
 pip install syncpy


```


 

 -Step 3: Open anaconda-navigator and lunch jupyter notebook
----------------------------------------------------------
Now you have installed an environement that contains all the dependencies required for SyncPy to work. 
What remains is to test on a real example using jupyther notebook.

Open Anaconda-navigator ->Choose SyncPy environement ->lunch jupyter notebook:

![Anaconda](/img/2.png)

