#!/bin/bash
y=`date +%Y`
mo=`date +%m`
d=`date +%d`
i=0
while [ $i -lt 1 ]
do 	grep "S1"  /var/www/html/data/$y-$mo-$d.csv | tail -1 | sed 's/,/ /g' > toto1.tmp
	grep "S2"  /var/www/html/data/$y-$mo-$d.csv | tail -1 | sed 's/,/ /g' > toto2.tmp
	grep "S3"  /var/www/html/data/$y-$mo-$d.csv | tail -1 | sed 's/,/ /g' > toto3.tmp
	grep "S4"  /var/www/html/data/$y-$mo-$d.csv | tail -1 | sed 's/,/ /g' > toto4.tmp
	grep "S5"  /var/www/html/data/$y-$mo-$d.csv | tail -1 | sed 's/,/ /g' > toto5.tmp	
	read bidon year month day hour min sec  lat lon alt nSat ga acqt temp lux r g b c tail bidon < toto1.tmp
	echo "<html>" > /var/www/html/index.html
	echo "<head>" > /var/www/html/index.html
	echo "  <meta http-equiv="refresh" content="9">" >> /var/www/html/index.html
	echo "</head>" >> /var/www/html/index.html
	echo "<body>" >> /var/www/html/index.html
	echo "DATE:" $y"-"$mo"-"$d "<br>">> /var/www/html/index.html
	echo "Time:" $hour":"$min":"$sec  "<br>">> /var/www/html/index.html
	echo "GPS Latitude:" $lat"  Longitude:"$lon"<br>">> /var/www/html/index.html
	echo "<br>">> /var/www/html/index.html
	echo "<table border=1>" >> /var/www/html/index.html
	echo "  <tr>" >> /var/www/html/index.html
	echo "    <th>Sensor</th>" >> /var/www/html/index.html
	echo "    <th>Gain</th>" >> /var/www/html/index.html
	echo "    <th>Integ. Time</th>" >> /var/www/html/index.html
	echo "    <th>CCT</th>" >> /var/www/html/index.html
	echo "    <th>Lux</th>" >> /var/www/html/index.html
	echo "    <th>R</th>" >> /var/www/html/index.html
	echo "    <th>G</th>" >> /var/www/html/index.html
	echo "    <th>B</th>" >> /var/www/html/index.html
	echo "    <th>C</th>" >> /var/www/html/index.html
	echo "    <th>Status</th>" >> /var/www/html/index.html
	echo "  </tr>" >> /var/www/html/index.html
	echo "  <tr>" >> /var/www/html/index.html
	echo "    <th>"1"</th>" >> /var/www/html/index.html
	echo "    <th>"$ga"</th>" >> /var/www/html/index.html
	echo "    <th>"$acqt"</th>" >> /var/www/html/index.html
	echo "    <th>"$temp"</th>" >> /var/www/html/index.html
	echo "    <th>"$lux"</th>" >> /var/www/html/index.html
	echo "    <th>"$r"</th>" >> /var/www/html/index.html
	echo "    <th>"$g"</th>" >> /var/www/html/index.html
	echo "    <th>"$b"</th>" >> /var/www/html/index.html
	echo "    <th>"$c"</th>" >> /var/www/html/index.html
	echo "    <th>"$tail"</th>" >> /var/www/html/index.html
	
	read bidon year month day hour min sec  lat lon alt nSat ga acqt temp lux r g b c tail bidon < toto2.tmp
	echo "  <tr>" >> /var/www/html/index.html
	echo "    <th>"2"</th>" >> /var/www/html/index.html
	echo "    <th>"$ga"</th>" >> /var/www/html/index.html
	echo "    <th>"$acqt"</th>" >> /var/www/html/index.html
	echo "    <th>"$temp"</th>" >> /var/www/html/index.html
	echo "    <th>"$lux"</th>" >> /var/www/html/index.html
	echo "    <th>"$r"</th>" >> /var/www/html/index.html
	echo "    <th>"$g"</th>" >> /var/www/html/index.html
	echo "    <th>"$b"</th>" >> /var/www/html/index.html
	echo "    <th>"$c"</th>" >> /var/www/html/index.html
	echo "    <th>"$tail"</th>" >> /var/www/html/index.html	

	read bidon year month day hour min sec  lat lon alt nSat ga acqt temp lux r g b c tail bidon < toto3.tmp
	echo "  <tr>" >> /var/www/html/index.html
	echo "    <th>"3"</th>" >> /var/www/html/index.html
	echo "    <th>"$ga"</th>" >> /var/www/html/index.html
	echo "    <th>"$acqt"</th>" >> /var/www/html/index.html
	echo "    <th>"$temp"</th>" >> /var/www/html/index.html
	echo "    <th>"$lux"</th>" >> /var/www/html/index.html
	echo "    <th>"$r"</th>" >> /var/www/html/index.html
	echo "    <th>"$g"</th>" >> /var/www/html/index.html
	echo "    <th>"$b"</th>" >> /var/www/html/index.html
	echo "    <th>"$c"</th>" >> /var/www/html/index.html
	echo "    <th>"$tail"</th>" >> /var/www/html/index.html
	
	read bidon year month day hour min sec  lat lon alt nSat ga acqt temp lux r g b c tail bidon < toto4.tmp
	echo "  <tr>" >> /var/www/html/index.html
	echo "    <th>"4"</th>" >> /var/www/html/index.html
	echo "    <th>"$ga"</th>" >> /var/www/html/index.html
	echo "    <th>"$acqt"</th>" >> /var/www/html/index.html
	echo "    <th>"$temp"</th>" >> /var/www/html/index.html
	echo "    <th>"$lux"</th>" >> /var/www/html/index.html
	echo "    <th>"$r"</th>" >> /var/www/html/index.html
	echo "    <th>"$g"</th>" >> /var/www/html/index.html
	echo "    <th>"$b"</th>" >> /var/www/html/index.html
	echo "    <th>"$c"</th>" >> /var/www/html/index.html
	echo "    <th>"$tail"</th>" >> /var/www/html/index.html
	
	read bidon year month day hour min sec  lat lon alt nSat ga acqt temp lux r g b c tail bidon < toto5.tmp
	echo "  <tr>" >> /var/www/html/index.html
	echo "    <th>"5"</th>" >> /var/www/html/index.html
	echo "    <th>"$ga"</th>" >> /var/www/html/index.html
	echo "    <th>"$acqt"</th>" >> /var/www/html/index.html
	echo "    <th>"$temp"</th>" >> /var/www/html/index.html
	echo "    <th>"$lux"</th>" >> /var/www/html/index.html
	echo "    <th>"$r"</th>" >> /var/www/html/index.html
	echo "    <th>"$g"</th>" >> /var/www/html/index.html
	echo "    <th>"$b"</th>" >> /var/www/html/index.html
	echo "    <th>"$c"</th>" >> /var/www/html/index.html
	echo "    <th>"$tail"</th>" >> /var/www/html/index.html	
	
	echo "  </tr>" >> /var/www/html/index.html
	echo "</table>" >> /var/www/html/index.html
	echo "</body>" >> /var/www/html/index.html
	echo "</html>" >> /var/www/html/index.html
	sleep 5
done
