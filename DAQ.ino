#include <Servo.h>
Servo myservo;
// Define pin connections & motor's steps per revolution
const int dirPinUP = 2;
const int stepPinUP = 3;
const int dirPinR = 4;
const int stepPinR = 5;
const int dirPinF = 6;
const int stepPinF = 7;


const int trigPin = 11;
const int echoPin = 10;
long duration;
int distance;

int point=600;
int f_b=400;
int u_d=400;


void setup()
{
  // Declare pins as Outputs
  pinMode(stepPinUP, OUTPUT);
  pinMode(dirPinUP, OUTPUT);
  pinMode(stepPinR, OUTPUT);
  pinMode(dirPinR, OUTPUT);
  pinMode(stepPinF, OUTPUT);
  pinMode(dirPinF, OUTPUT);
  myservo.attach(9);
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input


 // Serial.begin(9600);


}


void up(int v){
  
  digitalWrite(dirPinUP, HIGH);
  for(int x = 0; x < v; x++)
  {
    digitalWrite(stepPinUP, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPinUP, LOW);
    delayMicroseconds(1000);
  }
  
  }

  void down(int v){
  
  digitalWrite(dirPinUP, LOW);
  
  for(int x = 0; x < v; x++)
  {
    digitalWrite(stepPinUP, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPinUP, LOW);
    delayMicroseconds(1000);
  }
  
  }
  void forward(int v){
  
  digitalWrite(dirPinF, HIGH);
  for(int x = 0; x < v; x++)
  {
    digitalWrite(stepPinF, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPinF, LOW);
    delayMicroseconds(1000);

    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);
    distance = duration * 0.034 / 2;

    if(distance <5)
      {break;}
  }
  
  }
  void backward(int v){
  
  digitalWrite(dirPinF, LOW);
  for(int x = 0; x < v; x++)
  {
    digitalWrite(stepPinF, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPinF, LOW);
    delayMicroseconds(1000);
  }
  
  }

  void right(int v){
  
  digitalWrite(dirPinR, HIGH);
  for(int x = 0; x < v; x++)
  {
    digitalWrite(stepPinR, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPinR, LOW);
    delayMicroseconds(1000);
  }
  
  }

  void left(int v){
  
  digitalWrite(dirPinR, LOW);
  for(int x = 0; x < v; x++)
  {
    digitalWrite(stepPinR, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPinR, LOW);
    delayMicroseconds(1000);
  }
  
  }

  void grip(){

    myservo.write(60);
    
    }
  void releas(){

    myservo.write(0);
    }


    
void loop()
{
//  if (Serial.available() > 0) {
//
//  point = Serial.read();
//  if(point>=0 and point <=510){
//    point = 600;
//    }
//  }
  left(point);
  forward(f_b);
  grip();
  backward(f_b);
  up(u_d);
  forward(f_b);
  releas();
  backward(f_b);
  down(u_d);
  right(point);
  
}
