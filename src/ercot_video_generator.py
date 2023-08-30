import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import chain
import cv2
from matplotlib.path import Path
import math
import scipy.interpolate
import matplotlib.colors as colors
from scipy.signal import medfilt
import gc
from tqdm import tqdm
import sys
matplotlib.use('Agg')


#Generates the backround (graph and lines on left side of screen) for moving dot
def backround_generator(x_points, y_points, med_vals):
    plt.clf()
    height = 9 
    width = 7
    width_height = (width, height) 
    fig = plt.figure(figsize=width_height)
    ax = fig.add_axes([0.22, 0.15, 0.7, 0.7])
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    for arr in y_points:                            #plots every column in the data frame (each column is its own line)
        plt.plot(x_points, arr)
    plt.plot(x_points, med_vals, "r-" )             #plots the calculated median
    fig.canvas.draw()
    saved_bg = fig.canvas.copy_from_bbox(fig.bbox)
    (dot,) = plt.plot([], [], 'o', color='r', markersize=5, animated=True)  #make a dot object so it can be redrawn for every frame
    
    return (dot,saved_bg, fig)

#redraws the graph only for things that changed (the moving median dot)
def line_plot(x_points, median, index, d, bg, sf):
    #print(sf.canvas.get_width_height())
    sf.canvas.restore_region(bg)
    d.set_xdata(x_points[index])        #redraw median point (dot)
    d.set_ydata(median)
    sf.draw_artist(d)
   
    sf.canvas.blit(sf.bbox)         #blit only redraws pieces that have changed
    sf.canvas.flush_events()        #flushes the previous GUI event
    img = figure2cv2img2(sf)
    return img

#median is the median value across all busses at a given x (median of y vals for a given x)
def find_median(x_arr, y_arr, column_size):
    median = []
    for i in range((len(x_arr))):
        tmp = []
        for j in range(column_size):
            tmp.append(y_arr[j + (i * column_size)])
        median.append(sorted(tmp)[int(len(tmp)//2)])        #calculation to find median
    return median

def calc_distance(cor1, cor2):
    '''
    calc_distance: Calculate to the distance between two points
    (Float,Float), (Float,Float) -> Float

    Return:
    the Euclidean distance of the two points
    '''
    return math.sqrt(((cor1[0] - cor2[0]) ** 2 + (cor1[1] - cor2[1]) ** 2))

def find_nearest_neighbor(cor, points):
    '''
    find_nearest_neighbor: find the nearest neighbor of a certain cor and return its index in points
    (Float,Float), [(Float,Float)] -> Int

    Return:
    the index of the nearest neighbor in the list.

    Args:
    cor: coordinates that need to be find the neighbor.
    points: a list of coordinates

    '''
    min_d = calc_distance(cor, points[0])
    min_index = 0
    for i, p in enumerate(points):
        cur_d = calc_distance(cor, p)
        if cur_d < min_d:
            min_d = cur_d
            min_index = i
    return min_index

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

try: 
    cli_area = eval(open(sys.argv[4]).read())
except:
    print("clip area file doesn't exist")

lat_range = (20, 40)                                    #range of vals for ERCOT 
lon_range = (-110, -90)

corners = []
for i in lon_range:                                     #generate corners of grid
    for j in range(lat_range[0], lat_range[1], 2):
        corners.append((i, j))
for i in range(lon_range[0], lon_range[1], 2):
    for j in lat_range:
        corners.append((i, j))
x_n = 500
y_n = 500
xi = np.linspace(lon_range[0], lon_range[1], x_n)       # 500 evenly spaced points for the lat and lon range
yi = np.linspace(lat_range[0], lat_range[1], y_n)
draw_fdr_x = []
draw_fdr_y = []

utk_logo = cv2.imread('utk-logo.jpg')                   #logos at the bottom right
oak_logo = cv2.imread('oak-logo.jpg')

utk_logo = cv2.resize(utk_logo, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
oak_logo = cv2.resize(oak_logo, (utk_logo.shape[1], utk_logo.shape[0]), interpolation=cv2.INTER_AREA)


#converts a matplotlib figure into an openCV image 
def figure2cv2img2(fig):
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img

#builds the color map on the right side of the screen
def build_map(idx, osc_data, num_elements, min_y, max_y, ercot_x_vals, ercot_y_vals, multitude, latitude):
    gc.collect()
    #store oscillation data for each respective bus location (for each given time)
    oscilations = []
    
    #at a given row in the oscillation data access the given oscillation value for each bus and store it in the array
    for i in osc_data[(idx*num_elements):((idx*num_elements) + num_elements)]:     #oscilation data across a row
        oscilations.append(i)

    points = []
    for i in range (len(latitude)):                                                 #bus locations(x,y), oscilation val at that location
        points.append((multitude[i], latitude[i], oscilations[i]))


    inner_x, inner_y, inner_z = [], [], []
    inner_points = []
    vertices = [(x, y) for x, y in zip(ercot_x_vals, ercot_y_vals)]

    #filter out the points that aren't inside of the ercot outline
    clip_path = Path(vertices)                             
    for (i,j) in zip(multitude, latitude):
        if clip_path.contains_point((i,j)):
            inner_x.append(i)
            inner_y.append(j)
            t = next(line for line in points if line[0] == i and line[1] == j)[2]
        
            inner_z.append(t)
            inner_points.append((i,j,t))

   
    
    for i in range(len(inner_x)):
        draw_fdr_x.append(inner_x[i])
        draw_fdr_y.append(inner_y[i])

    for corner in corners:
        n_nei = find_nearest_neighbor(corner, inner_points) #using find nearest neighbor interp method
        inner_x.append(corner[0])
        inner_y.append(corner[1])
        inner_z.append(inner_z[n_nei])

    
    zi = scipy.interpolate.griddata((inner_x, inner_y), inner_z, tuple(np.meshgrid(xi, yi)), method='linear')       #preform interpolation 

    #limit the range of values in zi
    min_freq = min_y
    max_freq = max_y
    for i in range(len(zi)):
        zi[i] = np.clip(zi[i], min_freq, max_freq, out=zi[i])
    
    
    for cor in cli_area:           
        zi[cor[1]][cor[0]] = np.nan


    #------Graph settings
    size_ = (9,9)
    fig = plt.figure(figsize=size_, frameon=True)
    ax = fig.add_axes([0.1, 0.15, 0.7, 0.7])

    
    ax.set_xlim([-106, -92.5])
    ax.set_ylim([25,36])
    ax.set_xlim(auto=False)
    ax.set_ylim(auto=False)

    bgx = np.arange(lon_range[0], lon_range[1] + 1)
    bgy = np.linspace(lat_range[0], lat_range[1] + 1, lon_range[1] - lon_range[0] + 1)
    bgz = np.zeros((lon_range[1] - lon_range[0] + 1, lon_range[1] - lon_range[0] + 1))
    ax.contourf(bgx, bgy, bgz, cmap='Blues', alpha=0.5, zorder=0)
    plt.plot(ercot_x_vals, ercot_y_vals, c='black', linewidth=3.0, zorder=10)        #texas outline


    color_range = np.linspace(min_freq, max_freq, 50, endpoint=True)
    cmap = plt.get_cmap('gist_rainbow_r')                                           #color range
    new_cmap = truncate_colormap(cmap, 0.2, 1)
    CS = ax.contourf(xi, yi, zi, cmap=new_cmap, vmin=min_freq, vmax=max_freq, levels=color_range, zorder=5)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.03, 0.7])
    cb = fig.colorbar(CS, cax=cbar_ax, orientation='vertical')
    cb.ax.axhline((sum(oscilations) / len(oscilations) - min_freq) / (max_freq - min_freq), c='k', linewidth=8.0) 

    ax.scatter(draw_fdr_x, draw_fdr_y, marker='^', c='w', edgecolors='black', s=40, zorder=10)  #bus locations

    #sliding colorbar on right of screen
    color_bar_ylabs = []
    for f in np.linspace(min_freq, max_freq, 11):
        color_bar_ylabs.append('{:.3f}'.format(f))
    cb.set_ticks(np.linspace(min_freq, max_freq, 11))
    cb.ax.set_yticklabels(color_bar_ylabs)
    cb.ax.plot([0,1], [med]*2, 'k', linewidth=10, alpha=.7)

    ax.set_title('ERCOT Data Display\n Frequency(HZ): {:.4f}'.format((med), fontsize=15))

    fig.canvas.draw()
    img = figure2cv2img2(fig)
    
    plt.clf()
    plt.cla()
    plt.close('all')
    return img
   

if len(sys.argv) < 4:
    print("Not enough arguements. Enter filenames in order: 1.Oscillation data, 2. Bus Locations, 3. ERCOT Grid, 4.Ercot Clip Area")


x_vals_data = []
y_vals_data = []
y_vals = []
latitude = []           #bus location y 
multitude = []          #bus location x
ercot_x_vals = []       #outline values
ercot_y_vals = []

try:
    raw_data = pd.read_excel(sys.argv[1], 'Sheet1', header=2)
except:
    print("oscillation data file doesn't exist")
    sys.exit(1)

column_size = len(raw_data.columns) - 1

#get data for plotting ERCOT map and Bus:
try:
    bus_data = pd.read_excel(sys.argv[2], 'Sheet1', header=0)
except:
    print("bus location file doesn't exist")
    sys.exit(1)

latitude = bus_data['Latitude'].values
multitude = bus_data['multitude'].values

#Reads in Ercot outline
try:
    bus_data = pd.read_csv(sys.argv[3], delimiter=',0' , engine="python")
except:
    print("Ercot grid file doesn't exist")
ERCOTpoints_arr = bus_data.columns

for val in ERCOTpoints_arr:
    val = val.replace(",", " ")
    tempList = val.split()
    x = float(tempList[0])
    y = float(tempList[1])
    ercot_x_vals.append(x)  #x values of outline
    ercot_y_vals.append(y)  #y values of outline


#put 'time' values into x_vals_data & oscillation values into y_vals_data
for column in raw_data:
    if(column == 'Time'): 
        time_data = raw_data['Time']
        x_vals_data = raw_data['Time']
    else:
        column_obj = raw_data[column]
        y_vals_data.append(column_obj.values)
        
max_y = 0
min_y = 9999

data_frame_osc = raw_data.loc[:, raw_data.columns != 'Time']           #orient Y values to be put in list by row not col (easier to find median)
for i in data_frame_osc.stack():
    y_vals.append(i)
    if i > max_y:
        max_y = i
    if i < min_y:
        min_y = i


med_values = find_median(x_vals_data, y_vals, column_size)
size = (1600,900)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_file = "movie4.avi"                                          #edit this to change output file name

#openCV settings
video_writer = cv2.VideoWriter(output_file, fourcc, 85, size)
d,bg,sf = backround_generator(x_vals_data, y_vals_data, med_values)

for idx, med in enumerate(tqdm(med_values)):
    graph_img = line_plot(x_vals_data, med, idx, d, bg, sf)
    map_img = build_map(idx, y_vals, column_size, min_y, max_y, ercot_x_vals, ercot_y_vals, multitude, latitude)

    frame = cv2.hconcat([graph_img, map_img])
    frame[800:800 + utk_logo.shape[0], 800:800 + utk_logo.shape[1]] = utk_logo          #logos
    frame[800:800 + oak_logo.shape[0], 1050:1050 + oak_logo.shape[1]] = oak_logo
    
    video_writer.write(frame)
    cv2.imshow('frame', frame)
    gc.collect()
   

    
   
video_writer.release()
cv2.destroyAllWindows()
