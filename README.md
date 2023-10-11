# Welcome on Be Stars Ha Line Classfication

**Automatic classification of Be Stars Ha Line profil with Fastai.**

Please, note that this project is experimental and on progress.

The objective of this algorithm is to use Deep Learning on a set of Be star spectra in order to derive an automatic classification of the line profiles. The Halpha line profiles described here are based on the coding implemented by V. Desnoux (see document "codage_profil.pdf").


**Authors :** M. Le Lain (Model Training) / V. Desnoux (Codification) / F. Cochard (Proposition)

**Licence :** ![https://www.gnu.org/licenses/quick-guide-gplv3.html](https://img.shields.io/badge/Licence-GPLv3-orange.svg?style=flat )

**Infos et contact :** [Stellartrip.net](https://stellartrip.net)

**Version :** 1.0 - Update 08-01-2021

**Licence GPLv3 Notice :** [Quick guide](https://www.gnu.org/licenses/quick-guide-gplv3.html)



    Classification automatique des étoiles Be selon le profil de raie Halpha
    Copyright (C) 2021  M. Le Lain (Model Training) - V. Desnoux (Codification) - F. Cochard (Proposition)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see https://www.gnu.org/licenses/.




## Introduction

This Notebook trains a Halpha line profile classification model from Be star spectra. The data used to train this model come exclusively from the [Base BeSS](http://basebe.obspm.fr/basebe/Accueil.php?flag_lang=fr)<sup>1</sup>. A special thanks to all the contributors of this base and to the people who maintain it should be noted here, for the possible use of all these data.


<p align="center">
  <img width="15%" height="15%" src="http://basebe.obspm.fr/basebe/Images/LogoBeSS.jpg"/>
</p>

The retrieval of data to build a dataset is explained in another document (link).

The process of preparing the data for training, as well as the training steps, are described before each run cell in this notebook.



The training of this model is done with the library [Fast.ai](https://www.fast.ai/)<sup>2</sup>, based on [PyTorch](https://pytorch.org/) <sup>3</sup>. 


<p align="center">
  <img width="40%" height="40%" src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" />  <img width="30%" height="30%" src="https://miro.medium.com/max/1200/1*PQTzNNvBlmjW0Eca-nw14g.png" />
</p>

## Download Dataset
You can download the dataset for test training process here (approx. 80Mo) :
[Donwload dataset](https://e.pcloud.link/publink/show?code=XZk6JXZs24TDASz2yXOgIQ62sC8sYfwYs0k)


## Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matthieulel/be-stars-ai-classification/blob/main/training/DL_Be_classif_training.ipynb)


## Production 


Launch App : [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://be-stars-ai-classification-d6yl9qvdybbamw4mfsyv56.streamlit.app/)


