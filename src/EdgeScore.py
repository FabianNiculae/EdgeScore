from shapely import geometry
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point

df = pd.read_csv('latLong.csv')
df['type'] = 'traffic_signal'
# print(df)
crs = {'init': 'epsg:4326'}

# Create a geoseries holding the single polygon. Coordinates in counter-clockwise order
# inner area coordinates
# pointList = [(52.22669847133732, 6.909191401494679), (52.226981059290274, 6.8774742547709415),
#              (52.21373281512692, 6.876090233822996), (52.21352081109679, 6.908499391020707)]
# inner area with the ring
pointList = [(52.22961055924491, 6.871450950146954), (52.209587383986594, 6.871970831975787),
             (52.21010536924198, 6.912358143978945), (52.2305321929256, 6.911700676109128)]
# pointList = [(52.24240418192129, 6.834810366620807), (52.19246686392612, 6.8415464729689734), (52.19381725631812, 6.913218225340269), (52.23412893937712, 6.929363495090334)]
from area import area

obj = {'type': 'Polygon', 'coordinates': [
    [[52.24240418192129, 6.834810366620807], [52.19246686392612, 6.8415464729689734],
     [52.19381725631812, 6.913218225340269], [52.23412893937712, 6.929363495090334]]]}

area_m2 = area(obj)

area_km2 = area_m2 / 1e+6
print('area m2:' + str(area_m2))
print('area km2:' + str(area_km2))
poly = geometry.Polygon(pointList)
spoly = gpd.GeoSeries([poly], crs=crs)

# Create geodataframe of points
# Converting lat/long to cartesian
import numpy as np


def get_cartesian(lat=None, lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371  # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z


print(get_cartesian(52.22961055924491, 6.871450950146954))
geometry = [geometry.Point(xy) for xy in zip(df.latitude, df.longitude)]
dfpoints = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

# Create a subset dataframe of points within the polygon
subset = dfpoints[dfpoints.within(spoly.geometry.iloc[0])]
# subset1 = dfpoints[dfpoints.within(spoly.geometry.iloc[0])]
# print('Number of points within polygon: ', subset.shape[0], subset)

import random

# Defining the randomization generator
# inner area
# poly = Polygon([(52.22669847133732, 6.909191401494679), (52.226981059290274, 6.8774742547709415),
#                 (52.21373281512692, 6.876090233822996), (52.21352081109679, 6.908499391020707)])
# inner area with the ring
poly = Polygon([(52.22961055924491, 6.871450950146954), (52.209587383986594, 6.871970831975787),
                (52.21010536924198, 6.912358143978945), (52.2305321929256, 6.911700676109128)])


def polygon_random_points(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)
    return points


# Choose the number of points desired. This example uses 20 points.
points = polygon_random_points(poly, 1000)
# Printing the results.
data = {'latitude': [],
        'longitude': [],
        'type': ''}
list = ["User", "Sensor"]
dg = pd.DataFrame(data)
# print(dg)
numerator = 0
for p in points:
    # print(p.x,",",p.y)
    item = random.choice(list)

    numerator += 1
    if numerator >= 997:
        new_row = {'latitude': p.x, 'longitude': p.y, 'type': "Sensor"}
    else:
        new_row = {'latitude': p.x, 'longitude': p.y, 'type': "User"}
    subset = subset.append(new_row, ignore_index=True)

from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


inRange = []
coord = []

pd.options.mode.chained_assignment = None
subset["Number of devices/sensors in range"] = np.nan
subset["Coordinates of the users/sensors in range"] = np.nan
subset["Coordinates of the users/sensors in range"] = subset["Coordinates of the users/sensors in range"].astype(
    'object')
for i in range(len(subset.index)):
    list = []
    if subset["type"][i] == 'traffic_signal':
        nr = 0
        sublist = []
        for b in range(len(subset.index)):
            if subset["type"][b] != 'traffic_signal' and subset["type"][b] != 'traffic_signal' and haversine(
                    subset["longitude"][i], subset["latitude"][i],
                    subset["longitude"][b],
                    subset["latitude"][b]) <= 500:
                sublist.append([subset["latitude"][b], subset["longitude"][b],
                                haversine(subset["longitude"][i], subset["latitude"][i], subset["longitude"][b],
                                          subset["latitude"][b]), 0])
                nr = nr + 1

        subset.at[i, "Coordinates of the users/sensors in range"] = sublist

        subset["Number of devices/sensors in range"][i] = nr

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
# subset.drop('id', inplace=True, axis=1)
# print(subset)

subset["Variable cost"] = np.nan
subset["Fixed cost"] = np.nan
subset["Total cost"] = np.nan
subset["CPU(in ghz)"] = np.nan
subset["Number of processors"] = np.nan
subset["transmission speed ( in mb/s )"] = np.nan
subset["Data that is transmitted by the users or sensors( in mb ) at a point in time"] = np.nan
subset[
    "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"] = 0

subset.drop('id', axis=1, inplace=True)
subset.drop('geometry', axis=1, inplace=True)
# print(subset)
varCosts = [100, 200, 300, 350]
for i in range(len(subset.index)):
    item = random.choice(varCosts)
    if subset["type"][i] == 'traffic_signal':
        # random.randint(100, 150)
        subset["Variable cost"][i] = item
        subset["Fixed cost"][i] = 60
        subset["Total cost"][i] = subset["Variable cost"][i] + subset["Fixed cost"][i]
        subset["CPU(in ghz)"][i] = 3
        subset["Number of processors"][i] = 18

        subset["transmission speed ( in mb/s )"][i] = 8142
    if subset["type"][i] == 'User':
        # using lte in nederlands the maximum speed of download is 65 mbps and uploading of 16.3

        subset["transmission speed ( in mb/s )"][i] = random.randint(120, 180)
        # (1,25) (40,80)
        subset["Data that is transmitted by the users or sensors( in mb ) at a point in time"][i] = random.randint(1, 2)
    if subset["type"][i] == 'Sensor':
        subset["transmission speed ( in mb/s )"][i] = random.randint(20, 49)
        # (0.0002)
        # subset["Data that is transmitted by the users or sensors( in mb ) at a point in time"][i] = 0.0002
        subset["Data that is transmitted by the users or sensors( in mb ) at a point in time"][i] = random.randint(1, 2)

subset["User/Sensor waiting time to receive a message back in milliseconds"] = np.nan


def totalData():
    for i in range(len(subset.index)):
        if subset["type"][i] == 'traffic_signal' and subset["Number of devices/sensors in range"][i]:
            subset[
                "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"][
                i] = 0
            total = 0
            for b in range(len(subset["Coordinates of the users/sensors in range"][i])):
                # subset.loc[subset.latitude == subset["Coordinates of the users/sensors in range"][i][b][
                # 0], "User/Sensor waiting time to receive a message back in milliseconds"]
                # random.uniform
                total += random.randint(
                    1,
                    subset[
                        subset[
                            'latitude'] ==
                        subset[
                            "Coordinates of the users/sensors in range"][
                            i][
                            b][
                            0]][
                        "Data that is transmitted by the users or sensors( in mb ) at a point in time"].values[
                        0])
                subset["Coordinates of the users/sensors in range"][i][b][3] = (subset[subset['latitude'] == subset[
                    "Coordinates of the users/sensors in range"][i][b][0]][
                                                                                    "Data that is transmitted by the users or sensors( in mb ) at a point in time"].values[
                                                                                    0] / subset[subset['latitude'] ==
                                                                                                subset[
                                                                                                    "Coordinates of the users/sensors in range"][
                                                                                                    i][b][0]][
                                                                                    "transmission speed ( in mb/s )"].values[
                                                                                    0] + subset[
                                                                                    "Coordinates of the users/sensors in range"][
                                                                                    i][b][2] / 299792458 + (subset[
                                                                                                                subset[
                                                                                                                    'latitude'] ==
                                                                                                                subset[
                                                                                                                    "Coordinates of the users/sensors in range"][
                                                                                                                    i][
                                                                                                                    b][
                                                                                                                    0]][
                                                                                                                "Data that is transmitted by the users or sensors( in mb ) at a point in time"].values[
                                                                                                                0] / (
                                                                                                                        subset[
                                                                                                                            "CPU(in ghz)"][
                                                                                                                            i] *
                                                                                                                        subset[
                                                                                                                            "Number of processors"][
                                                                                                                            i] * 8 * 1000000000)) + (
                                                                                            total
                                                                                            # random.randint(
                                                                                            #     1,
                                                                                            #     subset[
                                                                                            #         subset[
                                                                                            #             'latitude'] ==
                                                                                            #         subset[
                                                                                            #             "Coordinates of the users/sensors in range"][
                                                                                            #             i][
                                                                                            #             b][
                                                                                            #             0]][
                                                                                            #         "Data that is transmitted by the users or sensors( in mb ) at a point in time"].values[
                                                                                            #         0])
                                                                                            /
                                                                                            subset[
                                                                                                "transmission speed ( in mb/s )"][
                                                                                                i]) +
                                                                                subset[
                                                                                    "Coordinates of the users/sensors in range"][
                                                                                    i][b][
                                                                                    2] / 299792458) * 1000
                subset[
                    "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"][
                    i] = subset[
                             "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"][
                             i] + \
                         subset[subset['latitude'] == subset["Coordinates of the users/sensors in range"][i][b][0]][
                             "Data that is transmitted by the users or sensors( in mb ) at a point in time"].values[0]


totalData()

from topsis import Topsis

subset["Signal Radius ( in m )"] = np.nan
subset["Mean of waiting times of users and sensors assigned to this traffic signal"] = 0

import math


def mean_waititng_times():
    for i in range(len(subset.index)):
        subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i] = 0
        if subset["type"][i] == 'traffic_signal' and subset["Number of devices/sensors in range"][i] != 0:

            subset["Signal Radius ( in m )"][i] = 600
            for b in range(len(subset["Coordinates of the users/sensors in range"][i])):
                subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i] = \
                    subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i] + \
                    subset["Coordinates of the users/sensors in range"][i][b][3]
                # subset[subset['latitude'] == subset["Coordinates of the users/sensors in range"][i][b][0]][
                #     "User/Sensor waiting time to receive a message back in milliseconds"].values[0]

            subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i] = \
                subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i] / len(
                    subset["Coordinates of the users/sensors in range"][i])
            print(subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i])


mean_waititng_times()
row = []
# print(subset)
allvalues = []
import topsispy as tp

listofIndexes = []
nr = 0
condition = False
for i in range(len(subset.index)):
    row = []
    if subset["type"][i] == 'traffic_signal' and subset["Number of devices/sensors in range"][i] != 0 and pd.notna(
            subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i]):
        row.append(subset["Number of devices/sensors in range"][i])
        row.append(subset["Total cost"][i])
        row.append(subset["transmission speed ( in mb/s )"][i])
        row.append(subset[
                       "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"][
                       i])
        row.append(subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i])
        row.append(subset["Signal Radius ( in m )"][i])
        allvalues.append(row)
        listofIndexes.append(i)
        nr += 1
        # print(allvalues)

print(nr)
evaluation_matrix = np.array(allvalues)
# print(evaluation_matrix)

weights = [0.15, 0.11, 0.15, 0.15, 0.22, 0.22]

'''
if higher value is preferred - True
if lower value is preferred - False
'''

criterias = np.array([1, -1, 1, 1, -1, 1])

'''
if higher value is preferred - True
if lower value is preferred - False
'''
# criterias = np.array([True, True, True, True])

b = tp.topsis(evaluation_matrix, weights, criterias)

# print("best_distance\t", t.best_distance)
# print("worst_distance\t", t.worst_distance)
#
# # print("weighted_normalized",t.weighted_normalized)
#
# print("worst_similarity\t", t.worst_similarity)
# print("rank_to_worst_similarity\t", t.rank_to_worst_similarity())
#
# print("best_similarity\t", t.best_similarity)
# print(b)
# print(b[0])
# print(b[0],t.rank_to_best_similarity()[0])
bestcoordinates = []
latencytimes = []
nr_of_devices = []
totalcost = []
totaldata = []
# print("rank_to_best_similarity\t", t.rank_to_best_similarity())
k = 0
# print(subset)
k = listofIndexes[b[0]]
# k = random.randint(0,145)
bestcoordinates.append(k)
latencytimes.append(subset["Mean of waiting times of users and sensors assigned to this traffic signal"][k])
nr_of_devices.append(subset["Number of devices/sensors in range"][k])
totalcost.append(subset["Total cost"][k])
totaldata.append(subset[
                     "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"][
                     k])
print(bestcoordinates)
print(k)

subset["type"][k] = "edge_server"
print(subset["type"][k])
i = 0
print(subset.index)
copy = []

availableEdge = 0
sumtotal = 0
while availableEdge < 13:
    i = 0

    while i < len(subset.index):
        if subset["type"][i] == 'traffic_signal':
            b = 0
            copy = subset["Coordinates of the users/sensors in range"][i].copy()

            while b < len(subset["Coordinates of the users/sensors in range"][i]):
                ts_x = subset["Coordinates of the users/sensors in range"][i][b][0]
                ts_y = subset["Coordinates of the users/sensors in range"][i][b][1]
                j = 0
                es = subset["Coordinates of the users/sensors in range"][k]
                while j < len(es):

                    es_x = subset["Coordinates of the users/sensors in range"][k][j][0]
                    es_y = subset["Coordinates of the users/sensors in range"][k][j][1]

                    if (ts_x == es_x and ts_y == es_y):
                        # print(subset["Coordinates of the users/sensors in range"][i][b])
                        copy.remove(subset["Coordinates of the users/sensors in range"][i][b])
                        # print(copy)

                    j += 1
                b = b + 1

            subset["Coordinates of the users/sensors in range"][i] = copy
            subset["Number of devices/sensors in range"][i] = len(copy)

        i = i + 1
    totalData()
    mean_waititng_times()
    print(subset)
    # print(subset)
    availableEdge = availableEdge + 1
    listofIndexes = []
    allvalues = []
    nr = 0
    for i in range(len(subset.index)):
        row = []
        if subset["type"][i] == 'traffic_signal' and subset["Number of devices/sensors in range"][i] != 0 and pd.notna(
                subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i]):
            row.append(subset["Number of devices/sensors in range"][i])
            row.append(subset["Total cost"][i])
            row.append(subset["transmission speed ( in mb/s )"][i])
            row.append(subset[
                           "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"][
                           i])
            row.append(subset["Mean of waiting times of users and sensors assigned to this traffic signal"][i])
            row.append(subset["Signal Radius ( in m )"][i])
            allvalues.append(row)
            listofIndexes.append(i)

            nr += 1
            # print(allvalues)

    print(nr)
    if not allvalues:
        s = 0
        s1 = 0
        s2 = 0
        s2 = 0
        for i in range(len(latencytimes)):
            print(latencytimes[i])
            s = s + latencytimes[i]
            s1 = s1 + nr_of_devices[i]
            s2 = s2 + totalcost[i]
            s3 = s3 + totaldata[i]
        print(s / len(latencytimes))
        print(s1)
        print(s2)
        break;
    evaluation_matrix = np.array(allvalues)
    # print(evaluation_matrix)

    weights = [0.15, 0.11, 0.15, 0.15, 0.22, 0.22]

    '''
    if higher value is preferred - True
    if lower value is preferred - False
    '''

    criterias = np.array([1, -1, 1, 1, -1, 1])

    '''
    if higher value is preferred - True
    if lower value is preferred - False
    '''
    # criterias = np.array([True, True, True, True])

    b = tp.topsis(evaluation_matrix, weights, criterias)
    # print(b)
    print(listofIndexes)
    print(b[0])
    print(listofIndexes[b[0]])
    k = listofIndexes[b[0]]
    # cond = False
    # while cond == False:
    #     a = random.randint(0,145)
    #     if( a not in bestcoordinates):
    #         k = a
    #         cond = True
    # cond = False
    bestcoordinates.append(k)
    latencytimes.append(subset["Mean of waiting times of users and sensors assigned to this traffic signal"][k])
    nr_of_devices.append(subset["Number of devices/sensors in range"][k])
    totalcost.append(subset["Total cost"][k])
    totaldata.append(subset[
                         "Total data arrived at an edge server from all of the users/sensors that are assigned to it ( in mb ) at a point in time"][
                         k])

    print(bestcoordinates)
    print(k)
    subset["type"][k] = "edge_server"
    print(subset["type"][k])

# print(subset)
print(bestcoordinates)
s = 0
s1 = 0
s2 = 0
s3 = 0
for i in range(len(latencytimes)):
    print(latencytimes[i])
    if (latencytimes[i] != 0):
        s = s + latencytimes[i]
    s1 = s1 + nr_of_devices[i]
    s2 = s2 + totalcost[i]
    s3 = s3 + totaldata[i]
print(s / len(latencytimes))
print(s1)
print(s2)
import plotly.express as px

fig = px.scatter_mapbox(
    subset,  # Our DataFrame
    lat="latitude",
    lon="longitude",
    center={"lat": 52.220794891840754, "lon": 6.893832364785576},  # where map will be centered
    width=600,  # Width of map
    height=600,  # Height of map
    color="type",
    hover_data=["latitude", "longitude", "type"],
    zoom=12,  # what to display when hovering mouse over coordinate
)

fig.update_layout(mapbox_style="open-street-map")  # adding beautiful street layout to map

fig.show(config=dict(editable=True))
