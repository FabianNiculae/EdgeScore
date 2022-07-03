import numpy as np
import pandas as pd
from folium import Map
from folium.plugins import HeatMap
from IPython.display import HTML, display
import folium

# reading csv file
for_map = pd.read_csv('latLong.csv')

hmap = Map(location=[52.19, 6.82], zoom_start=8, )

hm_wide = HeatMap(
    list(zip(for_map.latitude.values, for_map.longitude.values)),
    min_opacity=0.2,
    radius=17,
    blur=15,
    max_zoom=1,
)

hmap.add_child(hm_wide)

hmap.save('map.html')


def auto_open(path):
    html_page = f'{path}'
    f_map.save(html_page)
    # open in browser.
    new = 2
    webbrowser.open(html_page, new=new)
