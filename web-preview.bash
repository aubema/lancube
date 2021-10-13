#!/bin/bash
y=`date +%Y`
mo=`date +%m`
d=`date +%d`
i=0
while [ $i -lt 12 ]
do let i=i+1
	grep "S1"  /var/www/html/data/$y-$mo-$d.csv | tail -1 | sed 's/,/ /g' > toto.tmp
	read bidon year month day hour min sec  lat lon alt nSat ga acqt temp lux r g b c tail bidon < toto.tmp
	echo "<html>" > /var/www/html/index.html
	echo "<head>" > /var/www/html/index.html
	echo "  <meta http-equiv="refresh" content="5">" >> /var/www/html/index.html
	echo "</head>" >> /var/www/html/index.html
	echo "<body>" >> /var/www/html/index.html
	echo "DATE:" $y"-"$mo"-"$d "<br>">> /var/www/html/index.html
	echo "Time:" $hour":"$min":"$sec  "<br>">> /var/www/html/index.html
	echo "<br>">> /var/www/html/index.html
	echo "SENSOR #1<br>">> /var/www/html/index.html
	echo "<table border=1>" >> /var/www/html/index.html
	echo "  <tr>" >> /var/www/html/index.html
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
