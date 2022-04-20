# Pyreidolia

## How to install the package

### 1 - Create an environment for the project

It is always a best practice to have a dedicated python environment for every project. Since `pyreidolia` has a lot of package dependencies, it runs best in its own virtualenv. An environment is a sandbox where we can play and install pacakges without breaking anything in the base environment. 

To create a fresh environment:
 - `conda create -n pyreidolia python numpy pandas matplotlib scipy tqdm numba`
 - `conda activate pyreidolia`
 - Install the pyreidolia package as explained below

You can also clone the `cloudseg`, but be sure that your hardware and drivers are the same (cuda, etc. ) - if not create a fresh environment as above:
  - save the `cloudseg.yml` in the `C:\WINDOWS\system32` (or in any other folder but then provide the full path `[full_path]/cloudseg.yml` in the command line).
  - Open the anaconda command/powershell prompt
  - Create env with following command: `conda env create -f cloudseg.yml`  
  - Installation of packages will proceed
  - Activate the environment `conda activate cloudseg`, you should then see `(cloudseg) PS C:\WINDOWS\system32>` instead of  `(base) PS C:\WINDOWS\system32>` (move from base to the new environment)

### 2 - Install the package
By installing the package, you will be able to import the different functions and classes in the package and sub-packages as:<br>
`from pyreidolia.plot import set_my_plt_style, plot_cloud, plot_rnd_cloud, draw_label_only`

You can use the setup.py to install the package

   - [clone the repo](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository) (says at `C:/Users/pyreidolia`)
   - in a command prompt: `> conda activate cloudseg`  
   - in the same command prompt: `> cd C:/Users/pyreidolia`
   - install using pip  `pip install .`  (pip will look for the `setup.py` and will proceed)

If some changes are made to the package (new version, etc.), one needs to reinstall it

## Explanations and Instructions

This repository contains the files necessary for the initialization of a red thread project as part of your [DataScientest] training (https://datascientest.com/).

It mainly contains this README.md file and an application template [Streamlit] (https://streamlit.io/).

**README**

The README.md file is a central part of any git repository. It allows you to present your project, its objectives, as well as explain how to install and launch the project, or even contribute to it.

You will therefore need to modify various sections of this README.md to include the necessary information.

- Complete ** in English ** the sections (`## Presentation` and` ## Installation` `## Streamlit App`) by following the instructions in these sections.
- Delete this section (`## Explanations and Instructions`)

**Application Streamlit**

An application template [Streamlit] (https://streamlit.io/) is available in the [`streamlit_app`] (streamlit_app) folder. You can use this template to highlight your project.)

## Presentation

Complete this section **in English** with a brief description of your project, the context (including a link to the DataScientest journey), and the objectives.

You can also add a brief presentation of the team members with links to your respective networks (GitHub and / or LinkedIn for example).

**Exemple :**

This repository contains the code for our project **PROJECT_NAME**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to **...**

This project was developed by the following team :

- John Doe ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Martin Dupont ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
conda env create -f cloudseg.yml
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

You can also run the Streamlit app in a [Docker](https://www.docker.com/) container. To do so, you will first need to build the Docker image :

```shell
cd streamlit_app
docker build -t streamlit-app .
```

You can then run the container using :

```shell
docker run --name streamlit-app -p 8501:8501 streamlit-app
```

And again, the app should then be available at [localhost:8501](http://localhost:8501).
