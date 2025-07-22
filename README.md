This is the repository for the 'Virtual Acoustics in Immersive Audio' workshop to be held at CCRMA, Stanford University in July 2025, taught by Orchisama Das (KCL) 
and Gloria Dal Santo (Aalto University). We will upload assignments and code here.

### Installation
- Fork the repo to your github account (optional).
- Clone the repository to your local machine using `git clone https://github.com/YOUR-USERNAME/VAIAWorkshop25Public.git`
- Create a virtual environment with `python3.10 -m venv .venv`. For Linux/Mac users, activate the virtual environment with `source .venv/bin/activate`. For Windows users, activate virtual environment with `C:\> .venv\Scripts\activate.bat` on the command prompt, or `C:\> .venv\Scripts\Activate.ps1` on PowerShell(not tested).
 - If you don't have python 3.10, you can install it with homebrew : `brew install python@3.10`
- Install the repository with `pip install -e .` `pyproject.toml` contains all dependendies. Installation with pyproject.toml requires pip > 21.3. Upgrade pip with `pip install --upgrade pip`.
- In week 2, we will ask you to generate and save spatial audio data in the [SOFA format](https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)). We will load these files into the [SPARTA plugin suite](https://github.com/leomccormack/SPARTA/releases/tag/v1.7.1) and auralize them in a DAW. For one of the assignments, we will use the [3DTI Spatialiser VST plugin](https://github.com/3DTune-In/3dti_AudioToolkit/releases). Make sure you have SPARTA and 3DTI installed, along with any DAW of your choice (we recommend using [Reaper](https://www.reaper.fm/)). 

### Structure
- Code is in the [src](src/) folder. We will ask you to complete the functions in the scripts in the this folder. Do not add any test code to this folder.
	- The code for week 1 is in the `room_acoustics` folder
	- The code for week 2 is in the `spatial_audio` folder
	- Reusable functions across both weeks are in `utils.py`
- All test code should be in `jupyter notebooks` and placed in the [notebooks](notebooks/) folder (NOT in `src`). To open a jupyter notebook, go to the terminal, activate virtual environment, and write `jupyter notebook &`. This will open notebooks in your browser.
	- You can call functions in the `src` folder from your notebooks. For example, to use the `t60_estimator` function in `src/room_acoustics/analysis.py`, you can write `from room_acoustics.analysis import t60_estimator` in your notebook.
	- When requesting assistance after the lecture hours, please submit plots and wav files along with your notebooks. For plotting we will use the `matplotlib` library and for reading/writing wav files we will use the `soundfile` library. 
- The instructions for assignments are in the [assignments](assignments/) folder. The assignments span a week each, but are divided in several parts.
- We recommend you download the data needed for the assignments in thed [data](data/) folder. The link to the online folder containing the data has been shared with you.

### Instructions for daily pulls

Solutions we will release should be in the `main` branch. Your code for daily assignments should be in different branches with names `day#` (# stands for number).

#### Forked repo

- The default branch name is `main`. Each day, create your own branch with `git checkout -b day#`.
- Commit the changes you make (`git add -u` + `git commit -m <commit-message>`). 
- We will release the solutions to the assignments daily. To keep up with those, configure git to sync your fork with the original repository.
	- First add the original repository as upstream, `git remote add upstream https://github.com/orchidas/VAIAWorkshop25Public.git`
	- Check out the main branch with `git checkout main`. You can do daily pulls of the solutions with `git pull upstream main`.
	- OPTIONAL -  To merge upstream `main` with your branch, pull the main branch, checkout your branch, and run `git merge main`. You will have to fix merge conflicts manually. **We recommend doing this only if you are comfortable with git.**


#### Cloned repo without fork

- The default branch name is `main`. Each day, create your own branch with `git checkout -b day#` for each day.
- Commit the changes you make (`git add -u` + `git commit -m <commit-message>`). 
- Check out the main branch with `git checkout main` and pull the daily solutions to your main branch with `git pull origin main`.


