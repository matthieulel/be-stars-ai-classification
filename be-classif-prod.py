#fastai imports
from fastai.vision.all import *
from fastai.vision.widgets import *

#Maths and visualisation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
#from pyts.datasets import load_gunpoint

# astronomy import
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from astropy.io import fits
from astropy.units import Quantity
from astropy.visualization import quantity_support
quantity_support()  # for getting units on the axes below  
from astropy.time import Time
from astropy import units as u
from astropy.table import Table
from astropy import units as u
import astropy.wcs as fitswcs #wcs
from specutils import Spectrum1D, SpectralRegion #spectrum1D (specutils)
from astropy.wcs import WCS

import os




#methods
def generate_spec1Ds_bess(fits_file_path):
    """
    This function take a FITS file for Generate two Spec1D with SpecUtils,
    entire spectrum and zoom spectrum on Halpha (6525 <> 6595 A)

    Parameters
    ----------
    fits_file_path : str
        The path for FITS file spectrum.

    Returns
    -------
    spec1D
      The spec1D object of the entire spectrum.
    spec1D
      the spec1D object of the Halpha spectrum.

    """
    

    f = fits.open(fits_file_path)

    #create global spectrum
    evt_data = Table(f[1].data)
    spec1D_global = Spectrum1D(spectral_axis=evt_data['WAVE'] * u.AA, flux=evt_data['FLUX'] * u.Jy)

    #create spectral region for Halpha line zoom (6525 <=> 6595)
    sr =  SpectralRegion(6525*u.AA, 6595*u.AA)

    #create a new spectrum of the selected region for plot
    sub_spectrum = extract_region(spec1D_global, sr)
    spec1D_Ha = Spectrum1D(flux=sub_spectrum.flux,spectral_axis=sub_spectrum.spectral_axis)
    f.close()
    return spec1D_global, spec1D_Ha 




#methods
def generate_spec1Ds(fits_file_path, spec_type):
    """
    This function take a FITS file for Generate two Spec1D with SpecUtils,
    entire spectrum and zoom spectrum on Halpha (6525 <> 6595 A)

    Parameters
    ----------
    fits_file_path : str
        The path for FITS file spectrum.

    Returns
    -------
    spec1D
      The spec1D object of the entire spectrum.
    spec1D
      the spec1D object of the Halpha spectrum.

    """
    
    f = fits.open(fits_file_path)

    if spec_type == 'bess':
        #create bess spectrum
        evt_data = Table(f[1].data)
        header = f[1].header
        spec1D_global = Spectrum1D(spectral_axis=evt_data['WAVE'] * u.AA, flux=evt_data['FLUX'] * u.Jy)


    if spec_type == 'personnal':
        specdata = f[0].data
        header = f[0].header

        #make WCS object
        wcs_data = fitswcs.WCS(header={'CDELT1': header['CDELT1'], 'CRVAL1': header['CRVAL1'],
                                        'CUNIT1': header['CUNIT1'], 'CTYPE1': header['CTYPE1'],
                                        'CRPIX1': header['CRPIX1']})

        #set flux units
        flux= specdata * u.Jy

        spec1D_global = Spectrum1D(wcs=wcs_data, flux=flux)






    #create spectral region for Halpha line zoom (6525 <=> 6595)
    sr =  SpectralRegion(6525*u.AA, 6595*u.AA)

    #create a new spectrum of the selected region for plot
    sub_spectrum = extract_region(spec1D_global, sr)
    spec1D_Ha = Spectrum1D(flux=sub_spectrum.flux,spectral_axis=sub_spectrum.spectral_axis)
    f.close()
    return spec1D_global, spec1D_Ha, header



def spec_plots(full_spec1D, region_spec1D):
    """
    Generate and show a two plots with Matplotlib from full and zoom Spec1D

    Parameters
    ----------
    full_spec1D : spec1D
        The spec1D object full spectrum to plot
    full_spec1D : spec1D
        The spec1D object Halpha zoom spectrum to plot

    """
    #create each plot 
    fig, axs = plt.subplots(2, 1, figsize=(16,9))

    #Global
    axs[0].plot(full_spec1D.spectral_axis, full_spec1D.flux)
    axs[0].set_ylabel('Flux')
    axs[0].set_title("Global Spectrum")

    #Halpha Zoom
    axs[1].plot(region_spec1D.spectral_axis, region_spec1D.flux)
    axs[1].set_ylabel('Flux')
    axs[1].set_title("Halpha Crop")

    #Plot
    fig.tight_layout()
    plt.show()
    fig.savefig('tmp/spec_plots.png', dpi=100, bbox_inches='tight')
    plt.close()


def generate_GAF(spec_to_gaf, field_type, png_path):
    """
    Generate a GAF graph in .png and record it in the path given in parameters.

    Parameters
    ----------
    spec_to_gaf : spec1D
        The halpha zoom spec1D object
    filed_type : str
        The type of graph : difference or sum
    png_path : Path
        The path for png records

    """
    X_spec = np.array([spec_to_gaf.flux, spec_to_gaf.spectral_axis])

    #record png from specArray in folder png_path
    gaf = GramianAngularField(method=field_type)
    X_gaf = gaf.fit_transform(X_spec)

    # Show the images for the first time series
    fig_gaf = plt.figure(figsize=(5,5))
    ax_gaf = fig_gaf.add_subplot()
    plt.axis('off')
    #ax_gaf.set_title('Plot title')
    #plt.colorbar(orientation='vertical')

    #Generate plot image
    im_gaf = plt.imshow(X_gaf[0], cmap='viridis', origin='lower')
    #Save fig
    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.close()
 


#------- Start Streamlit configuration here -------#

import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import plotly.figure_factory as ff
import numpy as np


#delete old files
if os.path.isfile('tmp/gip.png'):
    os.remove('tmp/gip.png')
if os.path.isfile('tmp/spec_plots.png'):
    os.remove('tmp/spec_plots.png')

#load model
learn_inf = load_learner('export.pkl', cpu=True)


#title
st.title('Be Stars Classification')
st.header('Halpha line profil detection app')

st.write('------------------------------------------')


#sidebar
col_sb1, col_sb2, col_sb3 = st.sidebar.columns([1,3,1])
with col_sb2:
    bess_logo = Image.open('img/public/bess.jpg')
    st.image(bess_logo,  use_column_width=True)


#sb_title = st.sidebar.title('Hi !')
sb_welcome = st.sidebar.header('Hi ! Welcome on Be Stars Classification Application.')


sb_welcome = st.sidebar.write('This app is based on a [ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html) detection model trained with fastai from BeSS spectrum dataset\
    , V. Desnoux Line Codification proposal, F. Cochard idea, and M. Le Lain technical implementation.')

st.sidebar.write('The dataset used for trained the model is approximately 5000 spectrums. Thanks to all the BeSS contributors\
     without whom this work would not have been possible.')




option = st.sidebar.selectbox(
    'Want to watch metrics of the training ?',
     ('', 'Result Panel', 'Matrix Confusion', 'Learning rate find'))

if option == 'Result Panel':
    image = Image.open('img/public/result.png')
    st.sidebar.image(image, use_column_width=True)
if option == 'Matrix Confusion':
    image = Image.open('img/public/matrix.png')
    st.sidebar.image(image, use_column_width=True)
if option == 'Learning rate find':
    image = Image.open('img/public/lr.png')
    st.sidebar.image(image, use_column_width=True)


st.sidebar.write('You can find more information about [BeSS here](http://basebe.obspm.fr/basebe) \
    and [line codification here](http://basebe.obspm.fr/basebe)')

st.sidebar.write('This app is a prototype, please if you find bugs, errors, or if you have any idea \
    for make best experience, please contact us here : https://stellartrip.net/contact')

st.write('---------------------------------')

st.sidebar.write('Powered by')

col_sb_bt1, col_sb_bt2 = st.sidebar.columns([2,1])
with col_sb_bt1:
    fastai_logo = Image.open('img/public/fastai.png')
    st.image(fastai_logo,  use_column_width=True)
with col_sb_bt2:
    stream_logo = Image.open('img/public/streamlit.png')
    st.image(stream_logo,  use_column_width=True)




left_column, right_column = st.columns(2)

with left_column:
    rb_type = st.radio("1 - Select Personnal or BeSS (VOTable) spectrum",
                ('Personnal', 'BeSS'))
    if rb_type == 'BeSS':
        spec_type = 'bess'
        st.info("You have selected a BeSS spectrum from VOTable request.")
    else:
        spec_type = 'personnal'
        st.info("You have selected a personnal spectrum (i.e. processed by ISIS, Demetra, VSpec, SpcAudace).")

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    #st.subheader("2")
    uploaded_file = st.file_uploader("2 - Choose a file")
    

if uploaded_file is not None:

    if spec_type == 'bess':
        specs1D = generate_spec1Ds(uploaded_file, 'bess')

    elif spec_type == 'personnal':
        specs1D = generate_spec1Ds(uploaded_file, 'personnal')

    #generate gadf for preds and show
    generate_GAF(specs1D[1], 'difference', 'tmp/gip.png')
    img = PILImage.create('tmp/gip.png')

    #preds
    pred,pred_idx,probs = learn_inf.predict(img)
    #st.write(pred)
    #st.write(pred_idx.item())
    #st.write((round(probs[pred_idx.item()].item(),4))*100)
    #get accuracy
    pred_acc = round(probs[pred_idx.item()].item(),4)*100   

    #show values and images
    st.write('------------------------------------------')
    col_pred, col_gadf, col_explanation = st.columns(3)
    with col_pred:
        st.subheader("Prediction")
        if pred_acc > 90:
            st.success(f"Line profil : {pred}")
            st.success(f"Accuracy : {pred_acc}")
            st.success('Seems a good preds.')
        elif pred_acc >70 and pred_acc < 90:
            st.warning(f"Line profil : {pred}")
            st.warning(f"Accuracy : {pred_acc}")
            st.warning('Need verification.')
        else :
            st.error(f"Line profil : {pred}")
            st.error(f"Accuracy : {pred_acc}")
            st.error('Seems a bad preds.')

    with col_gadf:
        st.subheader("GADF Plot")
        st.image(img, use_column_width=True)

    with col_explanation:
        st.subheader("Line profil explanation")
        if os.path.isfile('img/lines_explanation/'+pred+'.png'):
            image = Image.open('img/lines_explanation/'+pred+'.png')
            st.image(image, use_column_width=True)
        elif os.path.isfile('img/lines/'+pred+'.png'):
            image = Image.open('profil_img/'+pred+'.png')
            st.image(image, use_column_width=True)
        else:
            st.write('No image explanation yet sorry.')
  

    st.write('------------------------------------')
    #ha spec
    x_ha = specs1D[1].spectral_axis
    y_ha = specs1D[1].flux
    fig_ha = go.Figure(data=go.Scatter(x=x_ha, y=y_ha))
    config = {'displayModeBar': True, 'responsive': True}



    col_globalspec, col_zoomha= st.columns([2,1])

    with col_globalspec:
        st.subheader("Halpha Zoom")
        st.plotly_chart(fig_ha, use_container_width=True, config = config)

    with col_zoomha:
        st.subheader("Header infos")
        st.write(f"Filename : {uploaded_file.name}")
        df = pd.DataFrame(specs1D[2].values(),specs1D[2].keys())
        st.dataframe(df)
    #    st.subheader("Header infos")
    #    key_val = st.multiselect(
    #        'Select keys for show values (max 7).',
    #        ['OBJECT', 'DATE-OBS', 'OBSERVER', 'BSS-INST'],
    #        ['OBJECT', 'DATE-OBS'])


    #    if len(key_val) < 8:
    #        for i in key_val:
    #            st.info(specs1D[2][i]) 
    #    else:
    #        st.warning('Please, select only 3 items for keep good presentation')


    st.subheader("Global Spectrum")
    #global spec
    x_global = specs1D[0].spectral_axis
    y_global = specs1D[0].flux
    fig_global = go.Figure(data=go.Scatter(x=x_global, y=y_global))
    st.plotly_chart(fig_global, use_container_width=True)
    

    
    st.write('------------------------------------------')
    st.warning('Please note that the line profile detected here is a prediction made by \
        a deep learning algorithm trained on approximatly 5000 spectrums from BeSS database.\
             Always verifiy manually the line profil if you need to integrate them in your treatment processes.')
   

    st.info('by : [M. Le Lain](https://stellartrip.net) | source : [Github - Classif Be](https://github.com/matthieulel/be-stars-classif)')


    #d = np.array([[specs1D[2].keys], [specs1D[2].values]]),columns=['a', 'b']
    del img, pred, pred_acc, pred_idx, probs
   


    
st.write('------------------------------------')



#st.balloons()
#st.success("Test success")
#st.error("Test error")
#st.warning("Test warning")

#with st.spinner('Work in progress...'):
#    time.sleep(1)
#st.success('Done!')

#with st.echo():
#    st.write('Code test and execution')

#stop execution in progress
#st.stop()


#genre = st.radio("Select BeSS or personnal spectrum",
#                ('BeSS', 'Personnal'))
     
#if genre == 'BeSS':
#    st.write('BeSS spectrum selected')
#else:
#    st.write("Personnal spectrum selected")
