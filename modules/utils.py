import numpy as np
import pandas as pd
import math
from geopy.distance import geodesic

paranal_lat = -24.627135
paranal_long = -70.404413
paranal_coord = np.array([paranal_lat, paranal_long])


armazones_lat = -24.589447
armazones_long = -70.191790
armazones_coord = np.array([armazones_lat, armazones_long])



def bearing2rad(angle_bearing):
    """
    this function takes a bearing angle and transforms it into an angle
    in polar coordinates
    :param angle_bearing: angle to transform
    :return: angle for polar coord
    """
    return np.pi/2 - angle_bearing * np.pi / 180


def get_bearing(lat1, lon1, lat2, lon2):
    """
    it returns the bearing anle between two coordinates
    credits to: https://stackoverflow.com/users/1019167/alexwien
    :return: bearing in degrees
    """
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
        math.cos(lat2) * math.cos(dLon)
    brng = np.rad2deg(math.atan2(y, x))
    if brng < 0:
        brng += 360
    return brng


def find_angle_of_vector(vector):
    """
    returns the angle of the vector in radians
    :param vector: vector for which to search the angle
    :return: angle in radians
    """
    x, y = vector
    tan = y / x
    return np.arctan(tan)



def find_base():
    pass

def find_max_lag_neg_or_pos():
    pass





def proyect_velocity(velocity):
    """
    proyects the velocity vector in a vertical and horizontal velocity
    :param velocity: vector velocity in rec coord
    :return: vertical velocity proyection, horixontal velocity
    """
    pass

def filter_df_by_date_key(df, date_key):
    """
    takes a df and retuns a df by the date key

    :param df: df to filter
    :param date_key: date key to filter by
    :return: filtered df
    """

    return df[df['date_key'] == date_key]


def get_diff_kinds(df):
    """
    this function takes the wind dataframe data and split it into three
    dfs for the data of the different kinds

    :param df: wind data df
    :return: 3 dfs of that day but splitted by kind of data
    """

    h_filter = df['kind_of_data'] == 'height'
    w_d_filter = df['kind_of_data'] == 'winddirection'
    h_v_filter = df['kind_of_data'] == 'horizontalvelocity'

    height = df[h_filter]
    winddirection = df[w_d_filter]
    horizontalvelocity = df[h_v_filter]

    return height, winddirection, horizontalvelocity






def vector_from_polar(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def normalize(v):
    return v / (np.sqrt(np.square(v[0, :, :]) + np.square(v[1, :, :])))





def filter_dict(dict2filter, date_key):
    """
    this functions takes a date_key and a big dictionary with all the
    wind properties and returns a new dict only with the time asked
    :param dict2filter: dict to filter
    :param date_key: date_key with which to obtain filtering
    :return: dict
    """
    new_dict = {}
    where = np.where(dict2filter['date_key'] == date_key)
    unique = np.where(np.unique(dict2filter['datetime'][where]))
    for key in dict2filter:
        new_dict[key] = dict2filter[key][where][unique]
    return new_dict






































































phase_paranal_armazones = bearing2rad(get_bearing(*paranal_coord, *armazones_coord))
d_abs = geodesic(paranal_coord, armazones_coord).km

d = vector_from_polar(d_abs, phase_paranal_armazones)
























































































































