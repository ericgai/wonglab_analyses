import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def read_stats(dir,ignore=['Overall','Volume_Img=1','BoundingBoxAA_Length','BoundingBoxOO_Length'],verbose=False,grouped_surfaces=False,grouped_images=False):
    '''
    Read surface statistics from a directory. 
    Parameters:
        dir: str, relative path to directory that contains statistics files
        ignore: iterable of str, Imaris outputs statistics files with a common prefix and suffixes indicating the statistic contained. Files with suffixes in ignore will not be read.
        verbose: bool, provide additional details for processes.
        grouped_surfaces: bool, toggle to True if each .csv file contains statistics from more than one surface object. The dataframe returned will have a column 'Surpass Object' that corresponds to the surface object name of each surface. 
        grouped_images: bool, toggle to True if each .csv file containts statistics from more than one image. The dataframe returned will have a column 'Original Image Name' that corresponds to the image from which each surface is derived.
    '''
    files=os.listdir(dir)
    prefix=os.path.commonprefix(files)
    if verbose:
        print(f'Extracted file prefix as {prefix}')
    truncated_files=[os.path.splitext(file[len(prefix):])[0] for file in files]
    ignore=ignore.copy()
    if 'Position_X' in truncated_files:
        ignore.append('Position')
    cols=[]
    for i,file in enumerate(files):
        title=truncated_files[i]
        if title in ignore:
            if verbose:
                print(f'Ignorning {title}')
            continue
        title=title.removesuffix('_Img=1')
        if verbose:
            print(f'Reading in {title}')
        try:
            input_df=pd.read_csv(dir+r"\\"+file,header=2)
        except Exception as e:
            print(f'Could not import {title}')
            print(e)
        if 'OriginalID' in input_df.columns:
            sort_cols=['OriginalID']
        elif 'ID' in input_df.columns:
            sort_cols=['ID']
        else:
            sort_cols=[]
            print(f'No ID column found for {title}')
        if grouped_surfaces:
            sort_cols.insert(0,'Surpass Object')
        if grouped_images:
            sort_cols.insert(0,'Original Image Name')
        input_df=input_df.sort_values(by=sort_cols)
        if verbose:
            print(f'Sorting dataframe by {sort_cols}')
        if title=="Position":
            for dimension in ['X','Y','Z']:
                cols.append(input_df[title+' '+dimension].rename(title+'_'+dimension))
        else:
            cols.append(input_df.iloc[:,0].rename(title).reset_index(drop=True))
    if grouped_surfaces:
        cols.append(input_df['Surpass Object'])
    if grouped_images:
        cols.append(input_df['Original Image Name'])
    try:
        cols.append(input_df['OriginalID'])
    except KeyError:
        cols.append(input_df['ID'])
    df=pd.concat(cols,axis=1)
    return df

def assign_channels(df, channel_dict):
    '''
    Assign channel names to a dataframe of Imaris statistics.
    Parameters:
        df: DataFrame, object containing Imaris statistics
        channel_dict: dict-like, with channel numbers as keys and channel names as values
    '''
    new_label={}
    for label in df.columns:
        idx=label.find('Ch=')
        if idx<0:
            continue
        i=idx+3
        while (i<len(label) and label[i].isdigit()):
            i+=1
        new_label[label]=label[:idx]+channel_dict[int(label[idx+3:i])]
    return df.rename(columns=new_label)

def plot_kde(kde_object,xmin,xmax,ymin,ymax,xsteps=100,ysteps=100,cmap='winter'):
    '''
    Plot a kde density in two dimensions
    Parameters:
        kde_object: scipy kde density
        xmin,xmax,ymin,ymax: x and y ranges to plot
        xsteps, ysteps: number of x and y steps to plot
    '''
    X,Y=np.meshgrid(np.linspace(xmin,xmax,xsteps),np.linspace(ymin,ymax,ysteps))
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde_grid=np.reshape(kde_object(positions),(xsteps,-1)).T
    plt.imshow(np.rot90(kde_grid),extent=[xmin,xmax,ymin,ymax],cmap=cmap)
    return None

def filter_df(df,column,criteria):
    '''
    Filter a dataframe based on some criteria applied to a column
    Parameters:
        df: DataFrame, containing data to apply filter
        column: str, title of column to filter
        criteria: Callable [ ... , bool], filtering criteria to apply to each row
    '''
    return df[df[column].map(criteria)]

def logicle_xform(arr):
    pass

def add_jitter(arr,jitter=1):
    '''
    Add jitter to an array of data for ease of visualization
    Parameters:
        arr: np.ndarray, array to which jitter is to be added
        jitter: float, average amount of jitter to be added
    '''
    return arr+np.random.normal(loc=0.,scale=jitter,size=arr.shape)

def purity_thresholds(arr1,arr2):
    '''
    Hyperparameter scan for a threshold classification model based on one parameter assuming equal prior of two classes. 
    Parameters:
        arr1: 1D iterable, containing data points from class 1
        arr2: 1D iterable, containing data points from class 2
    Outputs:
        thresholds: ordered list of all values from both classes
        ratios: accuracy of model at each value of threshold
    '''
    thresholds=[]
    ratios=[]
    arr1=np.sort(arr1) 
    arr2=np.sort(arr2)
    i1=0
    i2=0
    while i1<len(arr1) and i2<len(arr2):
        if arr1[i1]<arr2[i2]:
            process_arr1,process_arr2=True,False
        elif arr1[i1]==arr2[i2]:
            process_arr1,process_arr2=True,True
        else:
            process_arr1,process_arr2=False,True
        if process_arr1:
            if not process_arr2:    
                thresholds.append(arr1[i1])
            i1+=np.searchsorted(arr1[i1:],arr1[i1],side='right')
        if process_arr2:
            thresholds.append(arr2[i2])
            i2+=np.searchsorted(arr2[i2:],arr2[i2],side='right')
        # ratios.append(i1/len(arr1)/(i1/len(arr1)+i2/len(arr2)))
        ratios.append(i1/len(arr1)+(len(arr2)-i2)/len(arr2))
    assert len(thresholds)==len(ratios)
    return thresholds, ratios

def bin_norm_heatmap(x,y,ax=None,bins=10,**kwargs):
    if ax is None:
        ax = plt.gca()    
    hist,xedges,yedges=np.histogram2d(x,y,bins=bins)
    norm_hist2d=hist.T/np.sum(hist,axis=1)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, norm_hist2d,**kwargs)

def bin_wise_boxplot(x,y,ax=None,bins=10,hist_range=None,include_violin=True,**kwargs):
    if ax is None:
        ax = plt.gca()    
    bin_edges=np.histogram_bin_edges(x,bins=bins,range=hist_range)
    x_bin_indices=pd.Series(np.digitize(x,bin_edges))
    slicer_dict=x_bin_indices.groupby(x_bin_indices).indices
    plot_values=[]
    y=np.array(y)
    for i in range(1,len(bin_edges)):
        # print(slicer_dict[i])
        try:
            plot_values.append(y[slicer_dict[i]])
        except KeyError:
            plot_values.append([])
    ax.boxplot(plot_values,positions=(bin_edges[:-1]+bin_edges[1:])/2,manage_ticks=False,widths=(bin_edges[:-1]-bin_edges[1:])*0.5)
    if include_violin:
        int_val=lambda tuple,frac:min(tuple)+abs(tuple[0]-tuple[1])*frac
        ax.violinplot(x,positions=[int_val(ax.get_ylim(),0.05)],vert=False,side='high')

def proportion_true(arr):
    return arr.sum()/len(arr)

# np.random.seed(0)
# x=np.random.normal(size=10000,scale=5)
# y=np.random.exponential(size=10000)
# bin_wise_boxplot(x,y)
# plt.show()
