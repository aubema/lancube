#!/bin/bash
y=`date +%Y`
mo=`date +%m`
d=`date +%d`
grep "S1"  $y"-"$mo"-"$d".csv" | tail -1 > toto.tmp
read bidon year month day hour min sec  lat lon alt nSat ga acqt temp lux r g b c tail bidon < toto.tmp
echo "<html>" > index.html
echo "<body>" > index.html
echo "DATE:" $year"-"$month"-"$day >> index.html
echo "Time:" $hour":"$min":"$sec >> index.html
echo"<table>" >> index.html
echo"  <tr>" >> index.html
echo"    <th>Gain</th>" >> index.html
echo"    <th>Integ. Time</th>" >> index.html
echo"    <th>CCT</th>" >> index.html
echo"    <th>Lux</th>" >> index.html
echo"    <th>R</th>" >> index.html
echo"    <th>G</th>" >> index.html
echo"    <th>B</th>" >> index.html
echo"    <th>C</th>" >> index.html
echo"    <th>Status</th>" >> index.html
echo"  </tr>" >> index.html
echo"  <tr>" >> index.html
echo"    <th>"$ga"</th>" >> index.html
echo"    <th>"$acqt"</th>" >> index.html
echo"    <th>"$temp"</th>" >> index.html
echo"    <th>"$lux"</th>" >> index.html
echo"    <th>"$r"</th>" >> index.html
echo"    <th>"$g"</th>" >> index.html
echo"    <th>"$b"</th>" >> index.html
echo"    <th>"$c"</th>" >> index.html
echo"    <th>"$tail"</th>" >> index.html
echo"  </tr>" >> index.html
echo"</table>" >> index.html
echo "</body>" >> index.html
echo "</html>" >> index.html
