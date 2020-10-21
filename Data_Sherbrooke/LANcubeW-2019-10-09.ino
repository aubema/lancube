/* This software aim to control the LANcube system 
 *  
 *  Before compiling and uploading this software, be sure to set your computer time to UTC or GMT without saving time
 *  This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 *  http://creativecommons.org/licenses/by-sa/4.0/
 */
                                  
// Include relevant libraries
#include <Wire.h>                                            // communication
#include <Adafruit_Sensor.h>                                 // light sensors
#include <Adafruit_TCS34725.h>                               // light sensor
#include <Process.h>                                         // communication
#include "RTClib.h"                                          // RTC (real time clock)
#include <SD.h>                                              // micro SD card
#include <SPI.h>                                             // communication
#include <TinyGPS.h>                                         // GPS
#include <SoftwareSerial.h>                                  // Serial communication
#include <DHT.h>                                             // Humidity temperature sensor
#include <DHT_U.h>                                           // Humidity temperature sensor               



#define TCAADDR 0x70                                         // multiplexer address
#define DHTTYPE DHT22                                        //temperaure-humidity sensor model DTH22                                            

// Configuration section - PLEASE adapt according to your wiring
#define DHTPIN 25                                            // Pin for the DHT humidity temperature sensor
int red = 45;                                                 // Pin for RGB LED - RED
int green = 47;                                               // Pin for RGB LED - GREEN
int blue = 49;                                               // Pin for RGB LED - BLUE
int boutonA = 24;                                            // rf pad pin for entering automatic mode
int boutonB = 27;                                            // rf pad pin for starting manual mode or to exit automatic mode
int boutonC = 26;                                            // rf pad pin not yet used
int boutonD = 29;                                            // rf pad pin for taking a manual recording
int rtcchanel = 0;                                           // multiplexer chanel for the RTC
int chanels[] = { 1, 2, 3, 4, 5, 6};                         // multiplexer channel for each sensor (according to sensors[])
int sensors[] = { 1, 2, 3, 4, 5, 6};                         // numbers of the sensor
int sdpin = 53;                                              // CS pin for the microsd module
#define GPS_TX_DIGITAL_OUT_PIN 5                             // Communication du GPS
#define GPS_RX_DIGITAL_OUT_PIN 6                             // Communication du GPS

// Initializations 
// be sure to initialize DHT first
DHT dht(DHTPIN, DHTTYPE);                                    // Initialize DHT sensor
RTC_DS3231 rtc;                                              // Initialize Real time clock
Adafruit_TCS34725 tcs = Adafruit_TCS34725();                 // Initialize light sensor
TinyGPS gps;                                                 // Initialize GPS

// Misc global variables initialisation/declaration
String myFileName;                                           // File name for the SD card
String dataLine;                                             // data line to write to SD card
// Light sensors integration times names
tcs34725IntegrationTime_t TCStimes[]={TCS34725_INTEGRATIONTIME_2_4MS,TCS34725_INTEGRATIONTIME_24MS,TCS34725_INTEGRATIONTIME_50MS,TCS34725_INTEGRATIONTIME_101MS,TCS34725_INTEGRATIONTIME_154MS,TCS34725_INTEGRATIONTIME_700MS};
float tcsitime[] = {2.4, 24.0, 50.4, 100.8, 153.6, 614.4 };  // Light sensors related integration times in ms
// Light sensors gains names
tcs34725Gain_t TCSgains[]={TCS34725_GAIN_1X,TCS34725_GAIN_4X,TCS34725_GAIN_16X,TCS34725_GAIN_60X};  
float tcsgain[] = {1., 4., 16., 60.};                        // Light sensors related gain factors
int actgain[] = {3, 3, 3, 3, 3, 3};                          // actual gain number for each sensor e.g. 2=24.0 ms
int acttime[] = {0, 0, 0, 0, 0, 0};                          // actual integration time number of each sensor
int ntcs = 0 ;                                               // tcs sensor index
int nc = 0;            // sensor number in arrays
//int ng = 3 ;                                                 // index used for the gain to use with the sensor
//int ni = 5 ;                                                 // index use for the integration time to use with the sensor
String year;
String month;
String day;
String hour;
String minute;
String second;
String tail;
int mode = 2;                                                // acquisiton mode 
int prevmode;                                                // 1 = loop mode w auto gain & integration time , 
                                                             // 0 = manual w automatic gain & integration time , 
                                                             // 2 = loop mode w fixed gain and integration setted to maximal values
int buttonAState = 0; 
int buttonBState = 0; 
int buttonCState = 0; 
int buttonDState = 0; 
long startMillis;                                            // Satellites acquisition timing
long secondsToFirstLocation = 0;                             // Satellites acquisition timing 

//=============================================================================
// setup section
void setup() {
  // temperature-humidity
  dht.begin();
  // define input/output pins
  // pins des leds
  pinMode(red, OUTPUT);
  pinMode(green, OUTPUT);
  pinMode(blue, OUTPUT); 
  pinMode(sdpin, OUTPUT);
  pinMode(boutonA, INPUT);
  pinMode(boutonB, INPUT);
  pinMode(boutonC, INPUT); 
  pinMode(boutonD, INPUT);  
  pinMode(GPS_TX_DIGITAL_OUT_PIN, INPUT);
  pinMode(GPS_RX_DIGITAL_OUT_PIN, INPUT);
  Serial.begin(115200);
  Wire.begin();
  Serial1.begin(9600);                                       // communication pour le GPS
  delay(200);
  pinMode(sdpin, OUTPUT);                                    // initialisation de la carte SD. DOIT ETRE FAIT DANS LE SET UP
  Serial.print("Initializing SD card...");
  startMillis = millis();
  if (!SD.begin(sdpin)) {
    Serial.println("initialization failed!");
    setColor(0, 255, 0);
    delay(500);
    while (1);
  }
  Serial.println("SD card initialization done.");
  GetTimeRTC();  // get the time from real time clock and sync it with the computer in case of power lost
  dataLine = ""; //reset data line to nothing
  setColor(255, 0, 0);
  delay(2000);
}

//============================================================================
// main program - a loop to check the button mode and start measurements
void loop() {
  float itime;
  float gain;
  tcs34725IntegrationTime_t itimes;
  tcs34725Gain_t gains;
  // On lit les 4 boutons du RF pad
  buttonAState = digitalRead(boutonA);
  buttonBState = digitalRead(boutonB);
  buttonCState = digitalRead(boutonC);
  buttonDState = digitalRead(boutonD);
  if ( buttonAState == HIGH ){                              //  automatic mode
    Serial.println("Switch to Automatic mode");
    mode=1;
  }
  if ( buttonBState == HIGH ){                              //  automatic mode
    Serial.println("Switch to Manual mode - waiting for button D on remote");
    mode=0;
  }  
  if ( mode == 1 ) {
//    if (mode != prevmode) {
//      ni=1; // max integration time
//      ng=3; // max gain      
//    }
    setColor(128, 0, 0);
    delay(15);
    setColor(0, 0, 0);
    ReadAllSensors(); 
  }
  else  {                               //  manual mode
    setColor(0, 0, 128);                // manual idle mode permanent green
//    if (mode != prevmode) {
//      ni=1; // max integration time
//      ng=3; // max gain
//    }
    if (buttonDState == HIGH) {
       setColor(0, 0, 0);
       delay(15);
       ReadAllSensors();
     }
  }
  prevmode=mode;
}

//=============================================================================
// sequence the 6 sensors reading
void ReadAllSensors()
{
      // read sensor 1
      ntcs=1;
      nc=0;
      dataLine = "S";  // identify the sensor by begining the line with S followed by the sensor number
      dataLine += sensors[nc];
      dataLine += " ";
      GetTimeRTC();
      readLocation();
      humtemp();
      mesureTcs(nc, chanels[nc], sensors[nc]);  //max integration time = 614 ms (reported as 700 ms in the library)
      Serial.println(dataLine);
      writeToSD(myFileName, dataLine);
      // read sensor 2
      ntcs=2;
      nc=1;
      dataLine = "S";  // identify the sensor by begining the line with S followed by the sensor number
      dataLine += sensors[nc];
      dataLine += " ";
      GetTimeRTC();
      readLocation();
      humtemp();
      mesureTcs(nc, chanels[nc], sensors[nc]);
      Serial.println(dataLine);
      writeToSD(myFileName, dataLine);
      // read sensor 3
      ntcs=3;
      nc=2;
      dataLine = "S";  // identify the sensor by begining the line with S followed by the sensor number
      dataLine += sensors[nc];
      dataLine += " ";
      GetTimeRTC();
      readLocation();
      humtemp();
      mesureTcs(nc, chanels[nc], sensors[nc]);
      Serial.println(dataLine);
      writeToSD(myFileName, dataLine);
      // read sensor 4
      ntcs=4;
      nc=3;
      dataLine = "S";  // identify the sensor by begining the line with S followed by the sensor number
      dataLine += sensors[nc];
      dataLine += " ";
      GetTimeRTC();
      readLocation(); 
      humtemp();     
      mesureTcs(nc, chanels[nc], sensors[nc]);
      Serial.println(dataLine);
      writeToSD(myFileName, dataLine);
      // read sensor 5
      ntcs=5;
      nc=4;
      dataLine = "S";  // identify the sensor by begining the line with S followed by the sensor number
      dataLine += sensors[nc];
      dataLine += " ";
      GetTimeRTC();
      readLocation();
      humtemp();
      mesureTcs(nc, chanels[nc], sensors[nc]);
      Serial.println(dataLine);
      writeToSD(myFileName, dataLine);
      // read sensor 6
      ntcs=6;
      nc=5;
      dataLine = "S";  // identify the sensor by begining the line with S followed by the sensor number
      dataLine += sensors[nc];
      dataLine += " ";
      GetTimeRTC();
      readLocation();
      humtemp();
      mesureTcs(nc, chanels[nc], sensors[nc]);  
      Serial.println(dataLine); 
      writeToSD(myFileName, dataLine);
}

//=============================================================================
// call one sensor measurement
void mesureTcs(int nc, int nchanel , int ntcs )
{
  buttonBState = digitalRead(boutonB);
  if ( buttonBState == HIGH ) {
    Serial.println("Switch to Manual mode - but waiting for end of measuring sequence");
    mode=0;
  }
  tcaselect(nchanel);
  TCScheck(ntcs);
  tcs.setGain(TCSgains[actgain[nc]]);  
  tcs.setIntegrationTime(TCStimes[acttime[nc]]);
  dataLine += tcsgain[actgain[nc]] ;
  dataLine += " ";
  dataLine += tcsitime[acttime[nc]] ;
  dataLine += " ";  
  simpleReadTCS(nc, nchanel);
}

//=============================================================================
// execute sensor measurement
void simpleReadTCS(int nc, int nchanel)
{
  uint16_t r, g, b, c, colorTemp, lux;
  float flux;
  //delay(1.2*tcsitime[acttime[nc]]); // minimum delay to ensure that the sensor memory is completely flushed. Lower than that will leed to random readings
  tcs.getRawData(&r, &g, &b, &c);
  colorTemp = tcs.calculateColorTemperature(r, g, b);
  lux = tcs.calculateLux(r, g, b);
  flux = (float)lux / (float)tcsgain[actgain[nc]] / (float)tcsitime[acttime[nc]] * 481.;
  if ( nc != 6 ) {  // do not consider bottom sensor for auto gain and integration. This is because in some applications that sensor will be blocked by a table surface
    if (( r >= 40000 || g >= 40000 || b >= 40000 || c >= 40000  ) || ( r == g && r == b && r > 100 )){
      Serial.print("ERROR - SENSOR SATURATION : Too much light for mode ");
      Serial.println(mode);
      tail = "OE";  // over exposed flag
        if (acttime[nc] != 0 ) {
          setColor(128, 0, 0); //long red indicate a change in sensitivity
          Serial.println("Reducing integration time for next reading");
          acttime[nc] = acttime[nc] - 1;
        }
        else if (actgain[nchanel] != 0 ) {
          setColor(0, 128, 0); //long red indicate a change in sensitivity
          Serial.println("Reducing gain for next reading");
          actgain[nc] = actgain[nc] - 1;
        }
    }
    else if ( r <= 99 || g <= 99 || b <= 99 || c <= 99  ) {
      Serial.println("WARNING - less than 1% of accuracy : low light ");
      tail = "UE";  // under exposed flag
        if (actgain[nc] != 3 ) {
          setColor(128, 0, 128); //long red indicate a change in sensitivity
          Serial.println("Increasing gain for next reading");
          actgain[nc] = actgain[nc] + 1;
        }
        else if (acttime[nc] != 5 ) {
          setColor(128, 0, 128); //long purple indicate a change in sensitivity
          Serial.println("Increasing integration time for next reading");
          acttime[nc] = acttime[nc] + 1;
        }
    }
    else {
      tail = "OK";  // correct flag
      setColor(0 , 0, 0);
    }

  }
  else {
      tail = "OK";  // correct flag
  }
  dataLine += colorTemp ;
  dataLine += " ";
  dataLine += String(flux, 6);
  dataLine += " ";
  dataLine += r;
  dataLine += " ";
  dataLine += g;
  dataLine += " ";
  dataLine += b;
  dataLine += " ";
  dataLine += c;
  dataLine += " ";
  dataLine += tail;
}


//=============================================================================
// fonction du multiplexeur (channel de 0 a 7)
void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();  
}

//=============================================================================
// Fonction pour le real time clock, donne la date et l'heure
void GetTimeRTC(){
  tcaselect(rtcchanel);  // chanel of the RTC
  if (! rtc.begin()) {
    Serial.println("Couldn't find RTC");
  }
  if (rtc.begin()) {
    //Serial.println("Found RTC");
  }
  if (rtc.lostPower()) {
    Serial.println("RTC lost power, lets set the time with computer time!");
    Serial.println("Be sure that your computer time is set to GMT without time savings.");
    // following line sets the RTC to the date & time this sketch was compiled
    // be sure that your computer is set to UTC of GMT time without time savings
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }
  DateTime now = rtc.now();
  year = now.year();
  myFileName = year;
  if (now.month() <= 9 ) {
    month= "0";
    month += now.month();
  }
  else
  {
    month = now.month();
  }
  if (now.day() <= 9 ) {
    day= "0";
    day += now.day();
  }
  else
  {
    day = now.day();
  }

  if (now.hour() <= 9 ) {
    hour= "0";
    hour += now.hour();
  }
  else
  {
    hour = now.hour();
  }
    if (now.minute() <= 9 ) {
    minute= "0";
    minute += now.minute();
  }
  else
  {
    minute = now.minute();
  }
    if (now.second() <= 9 ) {
    second= "0";
    second += now.second();
  }
  else
  {
    second = now.second();
  }
  /* Serial.print("Time set to: ");
  Serial.print(hour);
  Serial.print(":");
  Serial.print(minute);
  Serial.print(":");  
  Serial.println(second); */
  // define the file name (must not exceed 8 characters plus "." plus 3 characters for the extension  
  myFileName += month;
  myFileName += day;
  myFileName += ".csv";
  //
  dataLine += year;
  dataLine += " ";
  dataLine += month;
  dataLine += " ";
  dataLine += day;  
  dataLine += " ";
  dataLine += hour;
  dataLine += " ";
  dataLine += minute;
  dataLine += " ";
  dataLine += second;
  dataLine += " ";  
}

//=============================================================================
// check if a sensor is connected 
void TCScheck(int ntcs){
  if (tcs.begin()) 
  {
    //Serial.print("Found a TCS sensor # ");
    //Serial.println( ntcs );
  } 
  else 
  {
    Serial.print("TCS sensor ");
    Serial.print(ntcs);
    Serial.println(" found ... check your connections");
    while (1);
  }
}


//=============================================================================
// write string dataLine (text) to the microSD card
void writeToSD(String fileName, String text)
{
  // open the file. note that only one file can be open at a time,
  // so you have to close this one before opening another.
  File myFile = SD.open(fileName, FILE_WRITE);
  // if the file opened okay, write to it:
  if (myFile) {
    //Serial.print("Writing to " + fileName + "...");
    myFile.println(text);
    // close the file:
    myFile.close();
    delay(5);
    //Serial.println("Writing done.");
  } else {
    // if the file didn't open, print an error:
    Serial.println("Error opening " + fileName);
    setColor(0, 255, 0);
  }
}

// ============================================================================
// get gps data
void readLocation(){
  float altitu = 0.0;
  float latitude = 0.0;
  float longitude = 0.0;
  int nsat = 0;
  bool newData = false;
  unsigned long chars = 0;
  unsigned short sentences, failed;
  // For one second we parse GPS data and report some key values
  for (unsigned long start = millis(); millis() - start < 1000;)
  {
    while (Serial1.available())
    {
      int c = Serial1.read();
      //Serial.print((char)c); // if you uncomment this you will see the raw data from the GPS
      ++chars;
      if (gps.encode(c)) // Did a new valid sentence come in?
        newData = true;
    }
  }
  if (newData)
  {
    // we have a location fix so output the lat / long and time to acquire
    if(secondsToFirstLocation == 0){
      secondsToFirstLocation = (millis() - startMillis) / 1000;
      Serial.print("Acquired in:");
      Serial.print(secondsToFirstLocation);
      Serial.println("s");
    }
    unsigned long age;
    gps.f_get_position(&latitude, &longitude, &age);
    altitu = gps.f_altitude();
    nsat = gps.satellites();
    latitude == TinyGPS::GPS_INVALID_F_ANGLE ? 0.0 : latitude;
    longitude == TinyGPS::GPS_INVALID_F_ANGLE ? 0.0 : longitude;
  }
  else {
    setColor(128, 0, 0); // blink green if gps not working
    delay(5);
    setColor(0, 0, 0);
  }
  if (chars == 0){
    // if you haven't got any chars then likely a wiring issue
    Serial.println("Check gps wiring!");
  }
  else if(secondsToFirstLocation == 0){
    // still working
  }
  dataLine += String(latitude, 6);
  dataLine +=  " ";
  dataLine += String(longitude, 6);
  dataLine += " ";
  dataLine += String(altitu, 6);
  dataLine += " ";
  dataLine += String(nsat, 6);
  dataLine += " ";  
}

//=============================================================================
// read relative humidity and temperature from the DHT22 sensor
void humtemp()
{
  float humi = dht.readHumidity();
  float temper = dht.readTemperature();
  dataLine += humi;
  dataLine += " ";
  dataLine += temper;
  dataLine += " ";  
}

//=============================================================================
// turn on led in color
void setColor(int redValue, int greenValue, int blueValue) {
  analogWrite(red, redValue);
  analogWrite(green, greenValue);
  analogWrite(blue, blueValue);
}
