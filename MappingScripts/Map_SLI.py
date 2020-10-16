import folium
import pandas as pd
from branca.element import Template, MacroElement


# reading data
sensor1 = pd.read_csv('lan3(1).csv')
sensor2 = pd.read_csv('lan3(2).csv')
sensor3 = pd.read_csv('lan3(3).csv')
sensor4 = pd.read_csv('lan3(4).csv')
sensor5 = pd.read_csv('lan3(5).csv')
sensor6 = pd.read_csv('lan3(6).csv')

# remove NaNas
sensor1 = sensor1.dropna(subset=['longitude'])
sensor1 = sensor1.dropna(subset=['latitude'])
sensor1 = sensor1.dropna(subset=['altitude'])

sensor2 = sensor2.dropna(subset=['longitude'])
sensor2 = sensor2.dropna(subset=['latitude'])
sensor2 = sensor2.dropna(subset=['altitude'])

sensor3 = sensor3.dropna(subset=['longitude'])
sensor3 = sensor3.dropna(subset=['latitude'])
sensor3 = sensor3.dropna(subset=['altitude'])

sensor4 = sensor4.dropna(subset=['longitude'])
sensor4 = sensor4.dropna(subset=['latitude'])
sensor4 = sensor4.dropna(subset=['altitude'])

sensor5 = sensor5.dropna(subset=['longitude'])
sensor5 = sensor5.dropna(subset=['latitude'])
sensor5 = sensor5.dropna(subset=['altitude'])

sensor6 = sensor6.dropna(subset=['longitude'])
sensor6 = sensor6.dropna(subset=['latitude'])
sensor6 = sensor6.dropna(subset=['altitude'])


# function for coloring circles depending on MSI values

def color_producer(sli):
    if sli < 0.1 or sli == 0.1:
        return 'red'
    elif sli > 0.1 and sli < 0.2 or sli == 0.2:
        return 'orange'
    elif sli > 0.2 and sli < 0.3 or sli == 0.3:
        return 'yellow'
    elif sli > 0.3 and sli < 0.4 or sli == 0.4:
        return 'lime'
    elif sli > 0.4 and sli < 0.5 or sli == 0.5:
        return 'green'
    elif sli > 0.5 and sli < 0.6 or sli == 0.6:
        return 'cyan'
    elif sli > 0.6 and sli < 0.7 or sli == 0.7:
        return 'blue'
    elif sli > 0.7 and sli < 0.8 or sli == 0.8:
        return 'navy'
    else:
        return 'purple'


# function that makes lists of lat lon alt and creates circle markers depending on the sensor


def sensor_list_and_markers(sensor, layer_name):
    layer = folium.FeatureGroup(name=layer_name, show=False)
    lat = list(sensor['latitude'])
    lon = list(sensor['longitude'])
    alt = list(sensor['altitude'])
    msi = list(sensor['MSI'])
    sli = list(sensor['SLI'])

    for lt, ln, at, MSI, SLI in zip(lat, lon, alt, msi, sli):
        layer.add_child(folium.CircleMarker(location=[lt, ln],
                                            radius=2.5,
                                            fill=True,
                                            popup=folium.Popup('*Latitude:' + str(lt) + ' *Longitude:' + str(ln) +
                                                               '  *MSI:' + str(MSI) + ' *SLI:' +
                                                               str(SLI)),
                                            fill_color=color_producer(SLI),
                                            color=color_producer(SLI),
                                            fill_opacity=1))
    m.add_child(layer)


# creating the map
m = folium.Map(location=[45.4, -71.89], zoom_start=13, control_scale=True, tiles='cartodbdark_matter')

# calling the circle markers function
sensor_list_and_markers(sensor1, "sensor1")
sensor_list_and_markers(sensor2, 'sensor2')
sensor_list_and_markers(sensor3, 'sensor3')
sensor_list_and_markers(sensor4, 'sensor4')
sensor_list_and_markers(sensor5, 'sensor5')
sensor_list_and_markers(sensor6, 'sensor6')

# Adding tiles
folium.TileLayer('OpenStreetMap').add_to(m)
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('CartoDB positron').add_to(m)

# adding a legend

template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Light pollution Sherbrooke map</title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

  <script>
  $( function() {
    $( "#maplegend" ).draggable({
                    start: function (event, ui) {
                        $(this).css({
                            right: "auto",
                            top: "auto",
                            bottom: "auto"
                        });
                    }
                });
});

  </script>
</head>
<body>


<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 10px; bottom: 50px;'>

<div class='legend-title'>SLI value</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:red;opacity:0.7;'></span> 0 - 0.1 </li>
    <li><span style='background:orange;opacity:0.7;'></span> 0.1 - 0.2 </li>
    <li><span style='background:yellow;opacity:0.7;'></span> 0.2 - 0.3</li>
    <li><span style='background:lime;opacity:0.7;'></span> 0.3 - 0.4</li>
    <li><span style='background:green;opacity:0.7;'></span> 0.4 - 0.5</li>
    <li><span style='background:cyan;opacity:0.7;'></span> 0.5 - 0.6</li>
    <li><span style='background:blue;opacity:0.7;'></span> 0.6 - 0.7</li>
    <li><span style='background:navy;opacity:0.7;'></span> 0.7 - 0.8</li>
    <li><span style='background:purple;opacity:0.7;'></span> > 0.8</li>

  </ul>
</div>
</div>

</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

macro = MacroElement()
macro._template = Template(template)

m.get_root().add_child(macro)

folium.LayerControl().add_to(m)

m.save('Map_SLI.html')
