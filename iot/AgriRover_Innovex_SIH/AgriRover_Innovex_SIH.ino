/*
 * AgriRover is the embedded control firmware developed by Team Innovex for
 * the Smart Crop Intelligence and Crop Vision System proposed for SIH 2026.
 *
 * Team: Innovex
 * SIH Challenge ID: SH26180
 * Organization: Qualcomm Inc
 * Category: Hardware
 * Theme: Disaster Management
 *
 * The challenge focuses on a field-deployable AI-powered Smart Farming
 * Assistant that can help farmers identify crop diseases, pests, nutrient
 * deficiencies, and irrigation requirements at an early stage. It also
 * considers agricultural risks such as droughts, floods, and heat waves,
 * where timely information can help farmers respond more effectively.
 *
 * Our proposed solution combines a mobile rover, crop vision, field sensors,
 * and AI/ML analysis. The rover moves through crop rows and brings the
 * sensing system closer to the plants being inspected. The captured visual
 * and sensor information can then be passed to the intelligence layer for
 * detection, analysis, and decision support.
 *
 * This firmware is the real-time control layer underneath that system. It
 * runs on an ESP32 and handles rover movement, the field headlight, wireless
 * communication, browser-based teleoperation, device status, and the local
 * motor safety mechanism. Keeping these responsibilities separate from the
 * AI/ML inference layer gives the project a clear boundary between sensing,
 * intelligence, and physical actuation.
 *
 * Hardware used in this build:
 * ⤷ ESP32 DevKit
 * ⤷ L293D dual H-bridge motor driver
 * ⤷ Four TT DC gear motors
 * ⤷ External LED used as the rover headlight
 * ⤷ PCA9685 16-channel 12-bit I2C PWM servo driver
 * ⤷ 4-DOF high-torque robotic manipulator (Base, Shoulder, Elbow, Gripper)
 *
 * The four motors are arranged as two drive groups. The front and rear motors
 * on the left are connected in parallel to L293D channel M1. The front and
 * rear motors on the right are connected in parallel to channel M2.
 *
 * ESP32 to L293D connections:
 *   GPIO 5  → M1 IN1
 *   GPIO 18 → M1 IN2
 *   GPIO 19 → M2 IN3
 *   GPIO 21 → M2 IN4
 *   GPIO 2  → Rover headlight
 *
 * ESP32 to PCA9685 connections:
 *   GPIO 23 → PCA9685 I2C SDA
 *   GPIO 22 → PCA9685 I2C SCL
 *   3.3V    → PCA9685 VCC (logic)
 *   GND     → PCA9685 GND
 *   OE pin  → NOT connected to ESP32 (handled in software via I2C)
 *   V+      → External 5V/6V battery (NOT from ESP32 board)
 *
 * EN1 and EN2 are enabled with the driver board jumper caps.
 *
 * Network configuration:
 *   Station network : Fayas
 *   Fallback AP     : AgriRover
 *   AP password     : 12345678
 *   AP address      : 192.168.9.1
 *   mDNS address    : http://agrirover.local
 *
 * While a direction is being held, the browser sends a small heartbeat to
 * keep the movement command alive. The ESP32 stores the time of the latest
 * accepted command. If those heartbeats stop for longer than the configured
 * safety interval, the ESP32 stops both motor sides locally. The rover is
 * therefore not dependent on the browser remaining responsive for safe
 * operation.
 *
 * Because two TT motors are connected in parallel on each L293D channel,
 * startup and stall current can be much higher than normal running current.
 * The driver, battery, wiring, and connectors should be selected using the
 * measured motor requirements before the rover is used in the field.
 *
 * Target platform: Arduino-ESP32 3.3.x.
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <DNSServer.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ESP32 hardware register access to suppress brownout resets during Wi-Fi burst
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// Firmware version identifies the software release installed on the rover.
constexpr const char *FIRMWARE_VERSION = "2.0.0";

// 1. Project and build configuration
//
// Keeping configuration in named constants makes the firmware easier to audit.
// It also prevents pin numbers and timing values from being scattered through
// motor, network, and HTTP code.
//

namespace Config {

  // Wi-Fi credentials used when the ESP32 tries to join the configured LAN.
  constexpr const char *STA_SSID = "Fayas";

  // Password for the configured station network.
  constexpr const char *STA_PASSWORD = "777888666";

  // SSID used when the ESP32 has to create its own control network.
  constexpr const char *AP_SSID = "AgriRover";

  // Password for the fallback access point.
  constexpr const char *AP_PASSWORD = "12345678";

  // Wi-Fi channel used by the fallback access point.
  constexpr uint8_t WIFI_CHANNEL = 6;

  // Maximum number of clients allowed on the fallback access point.
  constexpr uint8_t MAX_CLIENTS = 4;

  // Maximum time the rover may continue moving without a fresh command.
  constexpr uint32_t COMMAND_TIMEOUT_MS = 600;

  // Period between browser movement heartbeat requests.
  constexpr uint32_t HEARTBEAT_INTERVAL_MS = 250;

  // Period between browser status requests.
  constexpr uint32_t STATUS_POLL_INTERVAL_MS = 1000;

  // HTTP server port used by the local rover controller.
  constexpr uint16_t HTTP_PORT = 80;

  // Hostname advertised through mDNS.
  constexpr const char *MDNS_HOSTNAME = "agrirover";

  // Number of Wi-Fi connection attempts before the firmware falls back to AP.
  constexpr uint8_t WIFI_CONNECT_ATTEMPTS = 30;

  // Delay between Wi-Fi connection attempts.
  constexpr uint32_t WIFI_RETRY_DELAY_MS = 500;
}

// 2. Hardware pin configuration
//
// Each side of the rover uses one L293D channel. Both motors on a side receive
// the same two input signals. The PCA9685 I2C lines provide dedicated PWM
// control for the 4-DOF robotic arm without pin conflicts.
//

namespace Pins {

  // L293D M1 input 1 for the two left motors.
  constexpr uint8_t LEFT_IN1 = 5;

  // L293D M1 input 2 for the two left motors.
  constexpr uint8_t LEFT_IN2 = 18;

  // L293D M2 input 1 for the two right motors.
  constexpr uint8_t RIGHT_IN1 = 19;

  // L293D M2 input 2 for the two right motors.
  constexpr uint8_t RIGHT_IN2 = 21;

  // ESP32 GPIO connected to the rover headlight or status LED.
  constexpr uint8_t STATUS_LED = 2;

  // ESP32 GPIO connected to PCA9685 I2C SDA.
  constexpr uint8_t I2C_SDA = 23;

  // ESP32 GPIO connected to PCA9685 I2C SCL.
  constexpr uint8_t I2C_SCL = 22;

  // PCA9685 OE (Output Enable) pin.
  // Set to -1 because OE is NOT connected to ESP32.
  // The PCA9685 module has an onboard 10k pull-down resistor that keeps outputs
  // active, while servo sleep/wake is controlled 100% via I2C software commands.
  constexpr int8_t PCA9685_OE = -1;
}

// 3. Motor polarity and steering configuration
//
// The software direction depends on how each motor is mounted on
// the chassis. Keeping motor polarity here means the rest of the program can
// talk in terms of forward, reverse, left, and right without repeating those
// wiring details in every movement function.
//
// The current rover needs the pivot mapping inverted. With
// TURN_DIRECTION_INVERTED enabled, the LEFT command drives the left side
// forward and the right side in reverse. RIGHT does the opposite.

namespace MotorConfig {

  // Set true when the left motor pair is physically reversed relative to the
  // logical forward direction used by the rover software.
  constexpr bool LEFT_SIDE_INVERTED = true;

  // Set true when the right motor pair is physically reversed relative to the
  // logical forward direction used by the rover software.
  constexpr bool RIGHT_SIDE_INVERTED = true;

  // Select the pivot polarity required by the current rover chassis.
  constexpr bool TURN_DIRECTION_INVERTED = true;
}

// 4. Robotic arm configuration and constraints
//
// The 4-DOF manipulator uses a PCA9685 16-channel 12-bit PWM servo driver.
// Angular limits, calibration constants, home positions, and trajectory speed
// profiles are centralized here for safe physical operation.
//

namespace ArmConfig {

  // I2C address of the PCA9685 PWM driver board.
  constexpr uint8_t PCA9685_ADDR = 0x40;

  // Servo channels on PCA9685.
  constexpr uint8_t BASE_CH     = 0;
  constexpr uint8_t SHOULDER_CH = 1;
  constexpr uint8_t ELBOW_CH    = 2;
  constexpr uint8_t GRIPPER_CH  = 3;

  // PWM pulse width bounds (50 Hz analog/digital servos).
  constexpr uint16_t SERVO_MIN       = 120;
  constexpr uint16_t SERVO_MAX       = 520;
  constexpr uint8_t  SERVO_FREQUENCY = 50;

  // Mechanical direction reversals if a servo is mounted inversely.
  constexpr bool BASE_REVERSED     = false;
  constexpr bool SHOULDER_REVERSED = false;
  constexpr bool ELBOW_REVERSED    = false;
  constexpr bool GRIPPER_REVERSED  = false;

  // Joint physical angle limits in degrees.
  constexpr float BASE_MIN      = 5.0f;
  constexpr float BASE_MAX      = 175.0f;
  constexpr float SHOULDER_MIN  = 0.0f;
  constexpr float SHOULDER_MAX  = 180.0f;
  constexpr float ELBOW_MIN     = 0.0f;
  constexpr float ELBOW_MAX     = 180.0f;
  constexpr float GRIPPER_CLOSE = 70.0f;
  constexpr float GRIPPER_OPEN  = 180.0f;

  // Home and rest stance angles.
  constexpr float BASE_HOME     = 90.0f;
  constexpr float SHOULDER_HOME = 90.0f;
  constexpr float ELBOW_HOME    = 90.0f;
  constexpr float GRIPPER_HOME  = GRIPPER_OPEN;

  // Maximum angular speeds in degrees per second.
  constexpr float BASE_MAX_SPEED     = 360.0f;
  constexpr float SHOULDER_MAX_SPEED = 300.0f;
  constexpr float ELBOW_MAX_SPEED    = 340.0f;
  constexpr float GRIPPER_MAX_SPEED  = 500.0f;

  // Motion engine delta tick in milliseconds.
  constexpr uint32_t MOTION_INTERVAL = 8;
  constexpr float POSITION_TOLERANCE = 0.7f;

  // Autonomous waypoint sequence engine constraints.
  constexpr uint8_t MAX_POSITIONS = 20;
  constexpr uint32_t SEQUENCE_SETTLE_TIME = 250;
}

// 5. Server and runtime objects

// HTTP server instance that handles the rover control and status endpoints.
WebServer server(Config::HTTP_PORT);

// Captive portal and local DNS server instance for Access Point mode.
DNSServer dnsServer;

// I2C PWM servo driver instance for the 4-DOF robotic arm.
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(ArmConfig::PCA9685_ADDR);

// Flag indicating whether PCA9685 PWM hardware initialized successfully.
bool g_armInitialized = false;

// The embedded page is defined later in this file. This declaration lets the
// HTTP handler refer to it before the large flash-resident string is defined.
extern const char PAGE_INDEX[] PROGMEM;

// 6. Rover command model
//
// The enum gives the firmware one authoritative representation of movement.
// Using an enum instead of comparing strings prevents command spelling from
// leaking into motor-control logic.
//

enum class RoverCommand : uint8_t {

  // Both motor sides are stopped.
  STOPPED,

  // Both motor sides drive forward.
  FORWARD,

  // Both motor sides drive in reverse.
  BACKWARD,

  // Pivot command assigned to the left control action.
  LEFT,

  // Pivot command assigned to the right control action.
  RIGHT
};

// 7. Rover and robotic arm runtime state
//
// The state objects contain only values that need to survive between HTTP
// requests, serial commands, or loop iterations.
//

struct RoverState {

  // Last movement command accepted by the firmware.
  RoverCommand currentCommand = RoverCommand::STOPPED;

  // Current state of the rover headlight.
  bool ledOn = false;

  // Time at which the most recent movement command or heartbeat was accepted.
  uint32_t lastCommandAt = 0;

  // Number of movement commands accepted since boot.
  uint32_t commandCount = 0;
};

// Single runtime state object used by the chassis firmware.
RoverState state;

// Robotic arm waypoint representation for autonomous playback.
struct RobotWaypoint {
  int base;
  int shoulder;
  int elbow;
  int gripper;
};

// Robotic arm runtime telemetry and sequence state.
struct RoboticArmState {

  // Current dynamic positions interpolated smoothly by the motion engine.
  float baseCurrent     = ArmConfig::BASE_HOME;
  float shoulderCurrent = ArmConfig::SHOULDER_HOME;
  float elbowCurrent    = ArmConfig::ELBOW_HOME;
  float gripperCurrent  = ArmConfig::GRIPPER_HOME;

  // Target positions commanded by the user or autonomous sequence.
  float baseTarget      = ArmConfig::BASE_HOME;
  float shoulderTarget  = ArmConfig::SHOULDER_HOME;
  float elbowTarget     = ArmConfig::ELBOW_HOME;
  float gripperTarget   = ArmConfig::GRIPPER_HOME;

  // Trajectory speed scaling percentage (5% to 100%).
  int speed = 70;

  // Timestamp of the last servo motion update tick.
  unsigned long lastMotionUpdate = 0;

  // Track if arm servos are currently in software-detached sleep state.
  bool detached = false;

  // Autonomous sequence state.
  RobotWaypoint savedPositions[ArmConfig::MAX_POSITIONS];
  int savedCount = 0;
  bool runningSequence = false;
  bool pauseSequence = false;
  int sequenceIndex = 0;
  unsigned long sequenceStartTime = 0;
};

// Single runtime state object for the 4-DOF manipulator.
RoboticArmState armState;

// Forward declaration for the watchdog helper used by movement handlers.
void touchWatchdog();

// 8. Utility functions

/**
 * The command enum is converted into readable text for the status API
 * and serial logs.
 */
const char *commandToString(RoverCommand command) {

  switch (command) {

    case RoverCommand::FORWARD:
      return "FORWARD";

    case RoverCommand::BACKWARD:
      return "BACKWARD";

    case RoverCommand::LEFT:
      return "LEFT";

    case RoverCommand::RIGHT:
      return "RIGHT";

    case RoverCommand::STOPPED:
    default:
      return "STOPPED";
  }
}

/**
 * Clamps a floating-point angle to within specified safe physical bounds.
 */
float clampAngle(float value, float minimum, float maximum) {
  if (value < minimum) return minimum;
  if (value > maximum) return maximum;
  return value;
}

/**
 * Returns the absolute value of a floating-point number.
 */
float absFloat(float value) {
  return value < 0.0f ? -value : value;
}

/**
 * Maps a geometric angle in degrees to a PCA9685 12-bit PWM pulse tick count.
 */
int angleToPulse(float angle) {
  angle = clampAngle(angle, 0.0f, 180.0f);
  float pulse = ArmConfig::SERVO_MIN +
                ((ArmConfig::SERVO_MAX - ArmConfig::SERVO_MIN) * (angle / 180.0f));
  return (int)pulse;
}

// 9. Low-level motor control
//
// The functions in this section are the only place where direction-to-pin
// polarity is translated into GPIO states. Keeping this boundary small makes
// the high-level movement code easier to inspect.
//

/**
 * The requested logical direction is translated into the GPIO state required
 * by one motor side. The side inversion is applied at this boundary, which
 * keeps the rest of the rover code expressed in physical movement terms.
 */
void writeMotorSide(
  uint8_t in1,
  uint8_t in2,
  bool inverted,
  bool forward
) {

  // A logical reverse request becomes a logical forward request when that
  // physical side has been wired with reversed motor polarity.
  if (inverted) {
    forward = !forward;
  }

  // L293D input pair for the selected side.
  if (forward) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  }
}

/**
 * The left L293D channel uses LOW/LOW as its inactive state, so both
 * inputs are driven LOW whenever the left drive group is stopped.
 */
void leftMotorStop() {

  digitalWrite(Pins::LEFT_IN1, LOW);
  digitalWrite(Pins::LEFT_IN2, LOW);
}

/**
 * The controller drives both left motors in the configured forward direction.
 */
void leftMotorForward() {

  writeMotorSide(
    Pins::LEFT_IN1,
    Pins::LEFT_IN2,
    MotorConfig::LEFT_SIDE_INVERTED,
    true
  );
}

/**
 * The controller drives both left motors in the configured reverse direction.
 */
void leftMotorReverse() {

  writeMotorSide(
    Pins::LEFT_IN1,
    Pins::LEFT_IN2,
    MotorConfig::LEFT_SIDE_INVERTED,
    false
  );
}

/**
 * The right L293D channel enters its inactive LOW/LOW input state.
 */
void rightMotorStop() {

  digitalWrite(Pins::RIGHT_IN1, LOW);
  digitalWrite(Pins::RIGHT_IN2, LOW);
}

/**
 * The controller drives both right motors in the configured forward direction.
 */
void rightMotorForward() {

  writeMotorSide(
    Pins::RIGHT_IN1,
    Pins::RIGHT_IN2,
    MotorConfig::RIGHT_SIDE_INVERTED,
    true
  );
}

/**
 * The controller drives both right motors in the configured reverse direction.
 */
void rightMotorReverse() {

  writeMotorSide(
    Pins::RIGHT_IN1,
    Pins::RIGHT_IN2,
    MotorConfig::RIGHT_SIDE_INVERTED,
    false
  );
}

// 10. Rover movement logic
//
// The rover uses skid-steer pivot commands. A turn drives the two sides in
// opposite directions instead of stopping one side.
//

/**
 * The controller forces every drive output into the stopped state and updates the software state
 * updated accordingly. The same function is shared by normal STOP requests and
 * the movement safety watchdog.
 */
void stopRover() {

  // Remove drive power from both sides before changing the reported state.
  leftMotorStop();
  rightMotorStop();

  // The firmware now considers the rover stationary.
  state.currentCommand = RoverCommand::STOPPED;
}

/**
 * The controller drives both sides forward together.
 */
void moveForward() {

  // Both drive sides move in the physical forward direction.
  rightMotorReverse();
  leftMotorReverse();

  state.currentCommand = RoverCommand::FORWARD;
}

/**
 * Both drive sides move in reverse together, matching the proven Rover.ino
 * four-wheel backward behavior.
 */
void moveBackward() {

  // Both drive sides move in the physical backward direction.
  rightMotorForward();
  leftMotorForward();

  state.currentCommand = RoverCommand::BACKWARD;
}

/**
 * The LEFT command uses the same skid-steer mapping as Rover.ino.
 * The left side reverses while the right side moves forward.
 */
void turnLeft() {

  // Physical left pivot: left side reverse, right side forward.
  leftMotorReverse();
  rightMotorForward();

  state.currentCommand = RoverCommand::LEFT;
}

/**
 * The RIGHT command uses the same skid-steer mapping as Rover.ino.
 * The left side moves forward while the right side reverses.
 */
void turnRight() {

  // Physical right pivot: left side forward, right side reverse.
  leftMotorForward();
  rightMotorReverse();

  state.currentCommand = RoverCommand::RIGHT;
}

/**
 * The controller writes the requested headlight state to the GPIO output.
 */
void setLed(bool on) {

  // Store the requested state so /status and the web UI remain consistent.
  state.ledOn = on;

  // Apply the state to the physical LED pin.
  digitalWrite(
    Pins::STATUS_LED,
    on ? HIGH : LOW
  );
}

/**
 * The controller inverts the current headlight state through the normal LED
 * state handler.
 */
void toggleLed() {

  // Reuse setLed so the GPIO and state variable cannot drift apart.
  setLed(!state.ledOn);
}

// 11. Robotic arm servo and motion engine
//
// Trajectory planner, servo angle mapping, step interpolation, and sequence playback
// for the 4-DOF arm.
//

/**
 * Writes an angle directly to a designated PCA9685 PWM channel with reversal support.
 */
void writeServo(int channel, float logicalAngle, bool reversed) {
  if (!g_armInitialized) return;
  logicalAngle = clampAngle(logicalAngle, 0.0f, 180.0f);
  float physicalAngle = reversed ? (180.0f - logicalAngle) : logicalAngle;
  int pulse = angleToPulse(physicalAngle);
  pwm.setPWM(channel, 0, pulse);
}

/**
 * Transmits the current interpolated positions of all four servos to the PCA9685.
 */
void writeAllServos() {
  writeServo(ArmConfig::BASE_CH, armState.baseCurrent, ArmConfig::BASE_REVERSED);
  writeServo(ArmConfig::SHOULDER_CH, armState.shoulderCurrent, ArmConfig::SHOULDER_REVERSED);
  writeServo(ArmConfig::ELBOW_CH, armState.elbowCurrent, ArmConfig::ELBOW_REVERSED);
  writeServo(ArmConfig::GRIPPER_CH, armState.gripperCurrent, ArmConfig::GRIPPER_REVERSED);
}

/**
 * De-energizes all four servos (sets PWM pulse duty cycle to 0).
 * Emulates the PCA9685 OE (Output Enable) pin purely in software over I2C,
 * allowing servos to remain unpowered during startup and sleep modes without
 * requiring a physical GPIO wire from ESP32 to the PCA9685 OE pin.
 */
void detachArmServos() {
  if (!g_armInitialized) return;
  pwm.setPWM(ArmConfig::BASE_CH, 0, 0);
  pwm.setPWM(ArmConfig::SHOULDER_CH, 0, 0);
  pwm.setPWM(ArmConfig::ELBOW_CH, 0, 0);
  pwm.setPWM(ArmConfig::GRIPPER_CH, 0, 0);
  armState.detached = true;
  stopSequence();
}

/**
 * Computes the scaled angular velocity based on the user's configured speed percentage.
 */
float getArmSpeed(float maximumSpeed) {
  if (armState.speed <= 0) return 0.0f;
  float ratio = (float)armState.speed / 100.0f;
  float speed = maximumSpeed * ratio;
  if (speed < 15.0f) speed = 15.0f;
  return speed;
}

/**
 * Smoothly interpolates an individual servo's angle toward its commanded target.
 */
void moveServo(int channel, float &current, float target, float maximumSpeed, bool reversed, float dt) {
  float diff = target - current;
  if (absFloat(diff) <= ArmConfig::POSITION_TOLERANCE) {
    current = target;
    writeServo(channel, current, reversed);
    return;
  }

  float speed = getArmSpeed(maximumSpeed);
  if (speed <= 0.0f) {
    writeServo(channel, current, reversed);
    return;
  }

  float movement = speed * dt;
  if (movement > absFloat(diff)) movement = absFloat(diff);

  if (diff > 0.0f) current += movement;
  else current -= movement;

  if (absFloat(target - current) <= ArmConfig::POSITION_TOLERANCE) {
    current = target;
  }

  writeServo(channel, current, reversed);
}

/**
 * Periodic motion engine tick called from loop() to advance all four servo trajectories.
 */
void updateServos() {
  if (armState.detached) return;
  unsigned long now = millis();
  if (armState.lastMotionUpdate == 0) {
    armState.lastMotionUpdate = now;
  }
  unsigned long elapsed = now - armState.lastMotionUpdate;
  if (elapsed < ArmConfig::MOTION_INTERVAL) return;
  armState.lastMotionUpdate = now;

  // Real delta time clamped to 100ms max for rock-solid stability
  float dt = (float)elapsed / 1000.0f;
  if (dt > 0.1f) dt = 0.1f;

  // If speed is set to maximum (>= 95%), track target angles directly with zero lag
  if (armState.speed >= 95) {
    armState.baseCurrent = armState.baseTarget;
    armState.shoulderCurrent = armState.shoulderTarget;
    armState.elbowCurrent = armState.elbowTarget;
    armState.gripperCurrent = armState.gripperTarget;
    writeAllServos();
    return;
  }

  moveServo(ArmConfig::BASE_CH, armState.baseCurrent, armState.baseTarget, ArmConfig::BASE_MAX_SPEED, ArmConfig::BASE_REVERSED, dt);
  moveServo(ArmConfig::SHOULDER_CH, armState.shoulderCurrent, armState.shoulderTarget, ArmConfig::SHOULDER_MAX_SPEED, ArmConfig::SHOULDER_REVERSED, dt);
  moveServo(ArmConfig::ELBOW_CH, armState.elbowCurrent, armState.elbowTarget, ArmConfig::ELBOW_MAX_SPEED, ArmConfig::ELBOW_REVERSED, dt);
  moveServo(ArmConfig::GRIPPER_CH, armState.gripperCurrent, armState.gripperTarget, ArmConfig::GRIPPER_MAX_SPEED, ArmConfig::GRIPPER_REVERSED, dt);
}

/**
 * Returns true when all four servos have reached their target angles within tolerance.
 */
bool robotAtTarget() {
  return (absFloat(armState.baseCurrent - armState.baseTarget) <= ArmConfig::POSITION_TOLERANCE &&
          absFloat(armState.shoulderCurrent - armState.shoulderTarget) <= ArmConfig::POSITION_TOLERANCE &&
          absFloat(armState.elbowCurrent - armState.elbowTarget) <= ArmConfig::POSITION_TOLERANCE &&
          absFloat(armState.gripperCurrent - armState.gripperTarget) <= ArmConfig::POSITION_TOLERANCE);
}

/**
 * Commands a single joint target by joint index.
 */
void setServoTarget(int servo, float value) {
  switch (servo) {
    case 1: armState.baseTarget = clampAngle(value, ArmConfig::BASE_MIN, ArmConfig::BASE_MAX); break;
    case 2: armState.shoulderTarget = clampAngle(value, ArmConfig::SHOULDER_MIN, ArmConfig::SHOULDER_MAX); break;
    case 3: armState.elbowTarget = clampAngle(value, ArmConfig::ELBOW_MIN, ArmConfig::ELBOW_MAX); break;
    case 6: armState.gripperTarget = clampAngle(value, ArmConfig::GRIPPER_CLOSE, ArmConfig::GRIPPER_OPEN); break;
  }
}

/**
 * Commands target angles for all four joints simultaneously.
 */
void setAllServoTargets(float base, float shoulder, float elbow, float gripper) {
  armState.baseTarget     = clampAngle(base, ArmConfig::BASE_MIN, ArmConfig::BASE_MAX);
  armState.shoulderTarget = clampAngle(shoulder, ArmConfig::SHOULDER_MIN, ArmConfig::SHOULDER_MAX);
  armState.elbowTarget    = clampAngle(elbow, ArmConfig::ELBOW_MIN, ArmConfig::ELBOW_MAX);
  armState.gripperTarget  = clampAngle(gripper, ArmConfig::GRIPPER_CLOSE, ArmConfig::GRIPPER_OPEN);
}

// 12. Teach-and-repeat autonomous sequence engine
//
// Records waypoints, executes autonomous playback loops, and manages sequence states.
//

/**
 * Halts autonomous waypoint sequence playback.
 */
void stopSequence() {
  armState.runningSequence = false;
  armState.pauseSequence = false;
  armState.sequenceIndex = 0;
}

/**
 * Applies a stored waypoint index as the current active target posture.
 */
void applySavedPosition(int index) {
  if (index < 0 || index >= armState.savedCount) return;
  armState.baseTarget     = armState.savedPositions[index].base;
  armState.shoulderTarget = armState.savedPositions[index].shoulder;
  armState.elbowTarget    = armState.savedPositions[index].elbow;
  armState.gripperTarget  = armState.savedPositions[index].gripper;
}

/**
 * Initiates execution of the recorded waypoint sequence.
 */
void startSequence() {
  if (armState.savedCount <= 0) {
    Serial.println(F("[ARM_SEQ] Error: No saved positions."));
    return;
  }
  armState.runningSequence = true;
  armState.pauseSequence = false;
  armState.sequenceIndex = 0;
  applySavedPosition(0);
  armState.sequenceStartTime = millis();
  Serial.printf("[ARM_SEQ] Executing sequence: step 1 of %d\n", armState.savedCount);
}

/**
 * Advances to the next waypoint once the current target has been reached and settled.
 */
void updateSequence() {
  if (!armState.runningSequence || armState.pauseSequence) return;
  if (armState.savedCount <= 0) {
    stopSequence();
    return;
  }

  if (robotAtTarget() && (millis() - armState.sequenceStartTime >= ArmConfig::SEQUENCE_SETTLE_TIME)) {
    armState.sequenceIndex++;
    if (armState.sequenceIndex >= armState.savedCount) {
      armState.runningSequence = false;
      armState.sequenceIndex = 0;
      Serial.println(F("[ARM_SEQ] Autonomous sequence trajectory complete."));
      return;
    }
    applySavedPosition(armState.sequenceIndex);
    armState.sequenceStartTime = millis();
    Serial.printf("[ARM_SEQ] Step %d of %d\n", armState.sequenceIndex + 1, armState.savedCount);
  }
}

/**
 * Saves the current arm target posture into the waypoint buffer.
 */
bool saveCurrentPosition() {
  if (armState.savedCount >= ArmConfig::MAX_POSITIONS) {
    Serial.println(F("[ARM_SEQ] Error: Waypoint buffer full (20 max)."));
    return false;
  }
  armState.savedPositions[armState.savedCount].base     = (int)round(armState.baseTarget);
  armState.savedPositions[armState.savedCount].shoulder = (int)round(armState.shoulderTarget);
  armState.savedPositions[armState.savedCount].elbow    = (int)round(armState.elbowTarget);
  armState.savedPositions[armState.savedCount].gripper  = (int)round(armState.gripperTarget);
  armState.savedCount++;
  Serial.printf("[ARM_SEQ] Saved Waypoint #%d: [%d, %d, %d, %d]\n",
                armState.savedCount,
                armState.savedPositions[armState.savedCount - 1].base,
                armState.savedPositions[armState.savedCount - 1].shoulder,
                armState.savedPositions[armState.savedCount - 1].elbow,
                armState.savedPositions[armState.savedCount - 1].gripper);
  return true;
}

/**
 * Clears all stored waypoints from the sequencer memory.
 */
void clearSavedPositions() {
  stopSequence();
  armState.savedCount = 0;
  Serial.println(F("[ARM_SEQ] Cleared all saved waypoints."));
}

/**
 * Moves the arm to the standard neutral home posture.
 */
void resetRobot() {
  stopSequence();
  setAllServoTargets(ArmConfig::BASE_HOME, ArmConfig::SHOULDER_HOME, ArmConfig::ELBOW_HOME, ArmConfig::GRIPPER_HOME);
  Serial.println(F("[ARM] Home stance commanded: [90, 90, 90, 180]."));
}

/**
 * Actuates the gripper to its fully open limit.
 */
void openGripper() {
  stopSequence();
  armState.gripperTarget = ArmConfig::GRIPPER_OPEN;
  Serial.println(F("[ARM] Gripper commanded OPEN (180°)."));
}

/**
 * Actuates the gripper to its securely closed limit.
 */
void closeGripper() {
  stopSequence();
  armState.gripperTarget = ArmConfig::GRIPPER_CLOSE;
  Serial.println(F("[ARM] Gripper commanded CLOSE (70°)."));
}

/**
 * Applies predefined agricultural inspection stances.
 */
void applyPreset(const String &name) {
  stopSequence();
  if (name.equalsIgnoreCase("home")) {
    resetRobot();
  } else if (name.equalsIgnoreCase("inspect_canopy") || name.equalsIgnoreCase("canopy")) {
    setAllServoTargets(90.0f, 135.0f, 45.0f, ArmConfig::GRIPPER_OPEN);
  } else if (name.equalsIgnoreCase("inspect_foliar") || name.equalsIgnoreCase("foliar")) {
    setAllServoTargets(90.0f, 105.0f, 75.0f, ArmConfig::GRIPPER_OPEN);
  } else if (name.equalsIgnoreCase("sample_ground") || name.equalsIgnoreCase("ground")) {
    setAllServoTargets(90.0f, 45.0f, 135.0f, ArmConfig::GRIPPER_CLOSE);
  } else if (name.equalsIgnoreCase("stow")) {
    setAllServoTargets(90.0f, 30.0f, 160.0f, ArmConfig::GRIPPER_CLOSE);
  }
}

// 13. USB serial command processor
//
// Interprets legacy serial and Bluetooth commands for backwards compatibility.
//

/**
 * Parses and executes single-line commands from USB serial or HTTP endpoints.
 */
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  String upper = cmd;
  upper.toUpperCase();

  if (upper == "SAVE") { saveCurrentPosition(); return; }
  if (upper == "RUN") { startSequence(); return; }
  if (upper == "PAUSE") { if (armState.runningSequence) armState.pauseSequence = true; return; }
  if (upper == "RESUME") { if (armState.runningSequence) { armState.pauseSequence = false; armState.sequenceStartTime = millis(); } return; }
  if (upper == "RESET" || upper == "HOME") { resetRobot(); return; }
  if (upper == "CLEAR" || upper == "CLEARALL") { clearSavedPositions(); return; }
  if (upper == "OPEN" || upper == "GRIPOPEN") { openGripper(); return; }
  if (upper == "CLOSE" || upper == "GRIPCLOSE") { closeGripper(); return; }

  // Speed command: "ss70"
  if (cmd.length() >= 3 && cmd.charAt(0) == 's' && cmd.charAt(1) == 's') {
    armState.speed = constrain(cmd.substring(2).toInt(), 5, 100);
    Serial.printf("[ARM] Speed set to %d%%\n", armState.speed);
    return;
  }

  // Servo angle command: "s190", "s2120", etc.
  if (cmd.length() >= 3 && cmd.charAt(0) == 's') {
    int servo = cmd.charAt(1) - '0';
    if (servo == 1 || servo == 2 || servo == 3 || servo == 6) {
      float val = cmd.substring(2).toFloat();
      stopSequence();
      setServoTarget(servo, val);
      return;
    }
  }

  Serial.printf("[CLI] Unknown command: %s\n", cmd.c_str());
}

/**
 * Reads pending characters from the USB serial interface.
 */
void readSerial() {
  static String serialBuffer = "";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r' || c == '\n') {
      if (serialBuffer.length() > 0) {
        processCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 48) serialBuffer = "";
    }
  }
}

// 14. Movement safety watchdog
//
// A network connection is not a safe movement signal by itself. The browser
// therefore refreshes the command timestamp while a button is held. If that
// stream stops, the ESP32 stops the motors locally.
//

/**
 * The controller records the latest accepted movement time as the local safety
 * deadline reference.
 */
void touchWatchdog() {

  // millis() is unsigned and subtraction remains safe across its rollover.
  state.lastCommandAt = millis();
}

/**
 * An expired movement heartbeat forces a local rover stop, independent of
 * the browser control page.
 */
void serviceWatchdog() {

  // There is nothing to supervise while the rover is already stopped.
  if (state.currentCommand == RoverCommand::STOPPED) {
    return;
  }

  // Compare elapsed time using unsigned subtraction so millis() rollover is
  // handled without a separate special case.
  const uint32_t elapsed =
    millis() - state.lastCommandAt;

  if (elapsed > Config::COMMAND_TIMEOUT_MS) {

    // Fail safe by stopping both motor channels before reporting the timeout.
    stopRover();

    Serial.println(
      F("[SAFETY] Movement heartbeat timeout. Rover stopped.")
    );
  }
}

// 15. Status API

/**
 * The status endpoint returns the current rover and robotic arm state as a
 * compact JSON response. The endpoint remains read-only, so it neither
 * refreshes the watchdog nor changes motor outputs.
 */
void sendJsonStatus() {

  // Use String here because WebServer accepts a String response directly and
  // the status payload is small enough for this application.
  String json;
  json.reserve(768);

  bool isMoving = !robotAtTarget();

  // Current movement command.
  json += F("{\"command\":\"");
  json += commandToString(state.currentCommand);
  json += F("\",");

  // Headlight state.
  json += F("\"led\":");
  json += state.ledOn ? F("true") : F("false");
  json += F(",");

  // RSSI is available when the ESP32 is operating as a station.
  if (WiFi.getMode() == WIFI_STA) {

    json += F("\"rssi\":");
    json += String(WiFi.RSSI());

  } else {

    // AP mode does not expose one meaningful upstream RSSI value.
    json += F("\"rssi\":null");
  }

  // Device uptime in seconds.
  json += F(",\"uptime\":");
  json += String(millis() / 1000);

  // Report the active interface address.
  json += F(",\"ip\":\"");

  if (WiFi.getMode() == WIFI_STA) {
    json += WiFi.localIP().toString();
  } else {
    json += WiFi.softAPIP().toString();
  }

  json += F("\"");

  // Number of stations connected to the ESP32 AP. In station mode this will
  // normally be zero.
  json += F(",\"clients\":");
  json += String(WiFi.softAPgetStationNum());

  // Firmware-side command counter.
  json += F(",\"commands\":");
  json += String(state.commandCount);

  // 4-DOF Robotic Arm Telemetry
  json += F(",\"arm\":{");
  json += F("\"initialized\":"); json += g_armInitialized ? F("true") : F("false");
  json += F(",\"speed\":"); json += String(armState.speed);
  json += F(",\"is_moving\":"); json += isMoving ? F("true") : F("false");
  json += F(",\"seq_running\":"); json += armState.runningSequence ? F("true") : F("false");
  json += F(",\"seq_paused\":"); json += armState.pauseSequence ? F("true") : F("false");
  json += F(",\"seq_index\":"); json += String(armState.sequenceIndex + 1);
  json += F(",\"saved_count\":"); json += String(armState.savedCount);
  json += F(",\"base\":{\"cur\":"); json += String((int)round(armState.baseCurrent));
  json += F(",\"tgt\":"); json += String((int)round(armState.baseTarget));
  json += F(",\"min\":"); json += String((int)ArmConfig::BASE_MIN);
  json += F(",\"max\":"); json += String((int)ArmConfig::BASE_MAX); json += F("}");
  json += F(",\"shoulder\":{\"cur\":"); json += String((int)round(armState.shoulderCurrent));
  json += F(",\"tgt\":"); json += String((int)round(armState.shoulderTarget));
  json += F(",\"min\":"); json += String((int)ArmConfig::SHOULDER_MIN);
  json += F(",\"max\":"); json += String((int)ArmConfig::SHOULDER_MAX); json += F("}");
  json += F(",\"elbow\":{\"cur\":"); json += String((int)round(armState.elbowCurrent));
  json += F(",\"tgt\":"); json += String((int)round(armState.elbowTarget));
  json += F(",\"min\":"); json += String((int)ArmConfig::ELBOW_MIN);
  json += F(",\"max\":"); json += String((int)ArmConfig::ELBOW_MAX); json += F("}");
  json += F(",\"gripper\":{\"cur\":"); json += String((int)round(armState.gripperCurrent));
  json += F(",\"tgt\":"); json += String((int)round(armState.gripperTarget));
  json += F(",\"is_open\":"); json += (armState.gripperCurrent > 120.0f) ? F("true") : F("false"); json += F("}");
  json += F("}}");

  // Prevent browsers from displaying a stale status response.
  server.sendHeader(
    "Cache-Control",
    "no-cache, no-store, must-revalidate"
  );

  server.send(
    200,
    "application/json",
    json
  );
}

// 16. HTTP command handlers
//
// Every movement handler updates the watchdog timestamp only after the command
// has been accepted and the motor outputs have been written.
//

/**
 * The ESP32 serves the embedded rover controller directly from program memory.
 */
void handleRoot() {

  server.sendHeader(
    "Cache-Control",
    "no-cache, no-store, must-revalidate"
  );

  server.send_P(
    200,
    "text/html",
    PAGE_INDEX
  );
}

/**
 * The handler converts a forward request into the rover's forward state.
 */
void handleForward() {

  moveForward();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler converts a backward request into the rover's reverse state.
 */
void handleBackward() {

  moveBackward();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler converts the left control request into the configured physical left pivot.
 */
void handleLeft() {

  turnLeft();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler converts the right control request into the configured physical right pivot.
 */
void handleRight() {

  turnRight();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler stops the rover immediately.
 */
void handleStop() {

  stopRover();
  touchWatchdog();

  server.send(200, "text/plain", "OK");
}

/**
 * The handler switches the headlight ON.
 */
void handleLedOn() {

  setLed(true);

  server.send(200, "text/plain", "OK");
}

/**
 * The handler switches the headlight OFF.
 */
void handleLedOff() {

  setLed(false);

  server.send(200, "text/plain", "OK");
}

/**
 * The handler toggles the current headlight state.
 */
void handleLedToggle() {

  toggleLed();

  server.send(200, "text/plain", "OK");
}

/**
 * Sets joint target angles via query parameters.
 */
void handleArmSet() {
  stopSequence();
  armState.detached = false;
  bool direct = (armState.speed >= 90);

  if (server.hasArg("base")) {
    float val = clampAngle(server.arg("base").toFloat(), ArmConfig::BASE_MIN, ArmConfig::BASE_MAX);
    armState.baseTarget = val;
    if (direct) armState.baseCurrent = val;
  }
  if (server.hasArg("shoulder")) {
    float val = clampAngle(server.arg("shoulder").toFloat(), ArmConfig::SHOULDER_MIN, ArmConfig::SHOULDER_MAX);
    armState.shoulderTarget = val;
    if (direct) armState.shoulderCurrent = val;
  }
  if (server.hasArg("elbow")) {
    float val = clampAngle(server.arg("elbow").toFloat(), ArmConfig::ELBOW_MIN, ArmConfig::ELBOW_MAX);
    armState.elbowTarget = val;
    if (direct) armState.elbowCurrent = val;
  }
  if (server.hasArg("gripper")) {
    float val = clampAngle(server.arg("gripper").toFloat(), ArmConfig::GRIPPER_CLOSE, ArmConfig::GRIPPER_OPEN);
    armState.gripperTarget = val;
    if (direct) armState.gripperCurrent = val;
  }

  if (direct) {
    writeAllServos();
  }

  server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Target updated\"}");
}

/**
 * Sets a single joint angle via query parameters.
 */
void handleArmServo() {
  stopSequence();
  armState.detached = false;
  if (server.hasArg("joint") && server.hasArg("angle")) {
    int joint = server.arg("joint").toInt();
    float angle = server.arg("angle").toFloat();
    setServoTarget(joint, angle);
    if (armState.speed >= 90) {
      writeAllServos();
    }
    server.send(200, "application/json", "{\"status\":\"success\"}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Missing joint or angle\"}");
  }
}

/**
 * Commands the robotic arm to return to its home posture.
 */
void handleArmHome()    { resetRobot();    server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Homed\"}"); }

/**
 * Opens the gripper end-effector.
 */
void handleArmOpen()    { openGripper();   server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Gripper open\"}"); }

/**
 * Closes the gripper end-effector.
 */
void handleArmClose()   { closeGripper();  server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Gripper closed\"}"); }

/**
 * Updates the speed percentage of arm servo trajectories.
 */
void handleArmSpeed() {
  int val = -1;
  if (server.hasArg("val")) val = server.arg("val").toInt();
  else if (server.hasArg("speed")) val = server.arg("speed").toInt();
  else if (server.hasArg("s")) val = server.arg("s").toInt();

  if (val >= 0) {
    armState.speed = constrain(val, 5, 100);
    server.send(200, "application/json", "{\"status\":\"success\",\"speed\":" + String(armState.speed) + "}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Missing val or speed arg\"}");
  }
}

/**
 * Saves the current arm target posture into the waypoint buffer.
 */
void handleArmSave() {
  bool ok = saveCurrentPosition();
  if (ok) {
    server.send(200, "application/json", "{\"status\":\"success\",\"count\":" + String(armState.savedCount) + "}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Memory full\"}");
  }
}

/**
 * Starts execution of the recorded waypoint sequence.
 */
void handleArmRun() {
  startSequence();
  server.send(200, "application/json", "{\"status\":\"success\",\"count\":" + String(armState.savedCount) + "}");
}

/**
 * Pauses execution of the recorded waypoint sequence.
 */
void handleArmPause() {
  if (armState.runningSequence) armState.pauseSequence = true;
  server.send(200, "application/json", "{\"status\":\"success\",\"paused\":true}");
}

/**
 * Resumes execution of a paused waypoint sequence.
 */
void handleArmResume() {
  if (armState.runningSequence) {
    armState.pauseSequence = false;
    armState.sequenceStartTime = millis();
  }
  server.send(200, "application/json", "{\"status\":\"success\",\"paused\":false}");
}

/**
 * Stops execution of the autonomous sequence and clears sequence state.
 */
void handleArmStop() {
  stopSequence();
  server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Sequence stopped\"}");
}

/**
 * Clears all stored waypoints from memory.
 */
void handleArmClear() {
  clearSavedPositions();
  server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Cleared\"}");
}

/**
 * Applies a named agricultural posture preset.
 */
void handleArmPreset() {
  if (server.hasArg("name")) {
    applyPreset(server.arg("name"));
    server.send(200, "application/json", "{\"status\":\"success\",\"preset\":\"" + server.arg("name") + "\"}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Missing preset name\"}");
  }
}

/**
 * Returns JSON array of all currently recorded waypoints.
 */
void handleArmRecords() {
  String json = "{\"count\":" + String(armState.savedCount) + ",\"records\":[";
  for (int i = 0; i < armState.savedCount; i++) {
    if (i > 0) json += ",";
    json += "{\"id\":" + String(i + 1) +
            ",\"base\":" + String(armState.savedPositions[i].base) +
            ",\"shoulder\":" + String(armState.savedPositions[i].shoulder) +
            ",\"elbow\":" + String(armState.savedPositions[i].elbow) +
            ",\"gripper\":" + String(armState.savedPositions[i].gripper) + "}";
  }
  json += "]}";
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.send(200, "application/json", json);
}

/**
 * De-energizes all 4 servos by setting PWM duty cycle to 0.
 * Emulates the PCA9685 OE pin in software over I2C, allowing servos to cool down
 * and draw zero holding current without requiring a physical OE wire to ESP32.
 */
void handleArmSleep() {
  detachArmServos();
  server.send(200, "application/json", "{\"status\":\"success\",\"action\":\"arm_sleep\",\"message\":\"Arm servos detached (software OE)\"}");
}

/**
 * Re-energizes all 4 servos to their current target positions.
 */
void handleArmWake() {
  writeAllServos();
  server.send(200, "application/json", "{\"status\":\"success\",\"action\":\"arm_wake\",\"message\":\"Arm servos energized\"}");
}

/**
 * Executes a raw CLI or App Inventor command string received over HTTP.
 */
void handleArmCmd() {
  String c = "";
  if (server.hasArg("c")) c = server.arg("c");
  else if (server.hasArg("cmd")) c = server.arg("cmd");

  if (c.length() > 0) {
    processCommand(c);
    server.send(200, "application/json", "{\"status\":\"success\",\"cmd\":\"" + c + "\"}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Missing cmd or c parameter\"}");
  }
}

/**
 * The handler returns the current device state to the browser.
 */
void handleStatus() {
  sendJsonStatus();
}

/**
 * The server returns a controlled 404 response for unknown HTTP paths.
 */
void handleNotFound() {

  server.send(
    404,
    "text/plain",
    "404: Route not found"
  );
}

// 17. HTTP route registration

/**
 * The server registers the movement, headlight, robotic arm, status, and fallback HTTP routes.
 */
void registerRoutes() {

  // Main controller page.
  server.on("/", HTTP_GET, handleRoot);

  // Movement endpoints.
  server.on("/forward", HTTP_GET, handleForward);
  server.on("/backward", HTTP_GET, handleBackward);
  server.on("/left", HTTP_GET, handleLeft);
  server.on("/right", HTTP_GET, handleRight);
  server.on("/stop", HTTP_GET, handleStop);

  // Headlight endpoints.
  server.on("/led/on", HTTP_GET, handleLedOn);
  server.on("/led/off", HTTP_GET, handleLedOff);
  server.on("/led/toggle", HTTP_GET, handleLedToggle);

  // 4-DOF Robotic Arm endpoints.
  server.on("/arm/set", HTTP_GET, handleArmSet);
  server.on("/arm/servo", HTTP_GET, handleArmServo);
  server.on("/arm/home", HTTP_GET, handleArmHome);
  server.on("/arm/reset", HTTP_GET, handleArmHome);
  server.on("/arm/open", HTTP_GET, handleArmOpen);
  server.on("/arm/close", HTTP_GET, handleArmClose);
  server.on("/arm/speed", HTTP_GET, handleArmSpeed);
  server.on("/arm/save", HTTP_GET, handleArmSave);
  server.on("/arm/run", HTTP_GET, handleArmRun);
  server.on("/arm/pause", HTTP_GET, handleArmPause);
  server.on("/arm/resume", HTTP_GET, handleArmResume);
  server.on("/arm/stop", HTTP_GET, handleArmStop);
  server.on("/arm/clear", HTTP_GET, handleArmClear);
  server.on("/arm/preset", HTTP_GET, handleArmPreset);
  server.on("/arm/records", HTTP_GET, handleArmRecords);
  server.on("/arm/sleep", HTTP_GET, handleArmSleep);
  server.on("/arm/detach", HTTP_GET, handleArmSleep);
  server.on("/arm/wake", HTTP_GET, handleArmWake);
  server.on("/arm/status", HTTP_GET, handleStatus);
  server.on("/arm/cmd", HTTP_GET, handleArmCmd);
  server.on("/cmd", HTTP_GET, handleArmCmd);

  // Read-only device status.
  server.on("/status", HTTP_GET, handleStatus);

  // All unmatched requests receive a deterministic 404 response.
  server.onNotFound(handleNotFound);
}

// 18. Embedded web application
//
// The page is stored in flash through PROGMEM so the large HTML/CSS/JavaScript
// payload does not consume the same RAM used by the rover runtime.
//

const char PAGE_INDEX[] PROGMEM = R"HTML_PAGE(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0b1710">
<title>AgriRover & 4-DOF Arm | Team Innovex</title>

<style>
:root {
  --bg: #0b1710;
  --surface: #101f15;
  --surface2: #14271a;
  --surface3: #1b3323;
  --border: #294532;
  --border-light: rgba(142, 230, 155, 0.25);
  --green: #5fcf72;
  --green-light: #8ee69b;
  --green-dark: #2f8f45;
  --leaf: #76c893;
  --soil: #b58b63;
  --sky: #38bdf8;
  --yellow: #e5c95d;
  --red: #ef6461;
  --text: #f4f8f3;
  --text2: #c8d6c9;
  --text3: #849688;
  --sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --mono: "SF Mono", Menlo, Consolas, monospace;
  --radius: 16px;
  --radius-lg: 22px;
}

html, body {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow-x: hidden;
  background: radial-gradient(circle at 15% 0%, rgba(82,145,85,.16), transparent 40%),
              radial-gradient(circle at 85% 100%, rgba(56,189,248,.08), transparent 40%),
              var(--bg);
  color: var(--text);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
  -webkit-touch-callout: none !important;
  -webkit-user-select: none !important;
  user-select: none !important;
  touch-action: pan-y !important;
}

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  -webkit-tap-highlight-color: transparent !important;
  -webkit-touch-callout: none !important;
  -webkit-user-select: none !important;
  user-select: none !important;
  -webkit-user-drag: none !important;
}

button, .btn, .dir, .stop-btn, .step-btn, .action-btn, .preset-btn,
.btn-app-action, .toggle, .led-card, .dpad, .tab-btn, input[type="range"],
.gear-btn, .wp-chip, .positions-pill, .app-status-badge, .thumb {
  touch-action: none !important;
  -webkit-touch-callout: none !important;
  -webkit-user-select: none !important;
  user-select: none !important;
  -webkit-user-drag: none !important;
}

button {
  font-family: inherit;
  cursor: pointer;
  border: none;
  outline: none;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  padding: env(safe-area-inset-top, 8px) 14px env(safe-area-inset-bottom, 12px) 14px;
  gap: 10px;
  max-width: 900px;
  margin: 0 auto;
}

/* Header */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, rgba(31,57,36,.96), rgba(13,30,18,.96));
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 10px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,.3);
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-box {
  width: 38px;
  height: 38px;
  border-radius: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(145deg, #285c35, #173a22);
  border: 1px solid rgba(118,200,147,.4);
}

.brand h1 {
  font-size: 16px;
  font-weight: 800;
  letter-spacing: -.02em;
}

.brand p {
  font-size: 8px;
  color: var(--text3);
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.header-badges {
  display: flex;
  gap: 6px;
  align-items: center;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 9px;
  border-radius: 20px;
  background: rgba(0,0,0,.25);
  border: 1px solid var(--border);
  color: var(--text2);
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--text3);
}

.dot.live {
  background: var(--green);
  box-shadow: 0 0 8px rgba(95,207,114,.8);
  animation: pulse 1.5s infinite;
}

.dot.dead { background: var(--red); }
.dot.arm  { background: var(--sky); box-shadow: 0 0 8px rgba(56,189,248,.8); }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
}

/* Tab Switcher */
.mode-tabs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  background: rgba(14,27,19,.7);
  padding: 4px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}

.tab-btn {
  padding: 8px 10px;
  border-radius: 12px;
  background: transparent;
  color: var(--text3);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .02em;
  transition: all .16s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.tab-btn.active {
  background: linear-gradient(145deg, #24492e, #163620);
  color: var(--green-light);
  border: 1px solid rgba(142,230,155,.3);
  box-shadow: 0 4px 12px rgba(0,0,0,.25);
}

/* Stats Ribbon */
.stats {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
}

.stat {
  background: linear-gradient(145deg, rgba(24,45,29,.95), rgba(13,29,18,.95));
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 6px 2px;
  text-align: center;
}

.stat .lbl {
  color: var(--text3);
  font-size: 7px;
  font-weight: 800;
  text-transform: uppercase;
}

.stat .val {
  margin-top: 1px;
  color: var(--text);
  font-size: 10px;
  font-weight: 800;
  font-family: var(--mono);
}

.stat .go { color: var(--green-light); }
.stat .warn { color: var(--yellow); }
.stat .arm { color: var(--sky); }

/* Panels */
.tab-content { display: none; }
.tab-content.active { display: flex; flex-direction: column; gap: 10px; }

/* Arm Console (MIT App Inventor Layout and SVG Diagram) */
.app-arm-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: linear-gradient(135deg, rgba(26,49,31,.98), rgba(13,29,18,.98));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 8px;
}

.app-arm-title {
  font-size: 13px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: .02em;
}

.app-conn-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.btn-app-conn {
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  color: white;
  transition: all 0.15s ease;
  cursor: pointer;
}

.btn-app-conn.green {
  background: #2f8f45;
  border: 1px solid #5fcf72;
}
.btn-app-conn.green:hover { background: #38a150; }

.btn-app-conn.red {
  background: #a83f3d;
  border: 1px solid #ef6461;
}
.btn-app-conn.red:hover { background: #c54846; }

.app-status-badge {
  padding: 4px 9px;
  border-radius: 8px;
  font-size: 10px;
  font-weight: 700;
  background: rgba(0,0,0,0.35);
  border: 1px solid var(--border);
  color: var(--text2);
  font-family: var(--mono);
}
.app-status-badge.connected { color: #8ee69b; border-color: rgba(95,207,114,0.4); }

/* Split layout: Arm image on Left, Controls on Right */
.arm-split-deck {
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 12px;
  align-items: stretch;
}

@media (max-width: 768px) {
  .arm-split-deck {
    grid-template-columns: 1fr;
  }
}

.arm-canvas-card {
  background: linear-gradient(145deg, rgba(14,30,20,.95), rgba(9,20,13,.95));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0,0,0,.25);
}

.arm-canvas-header {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(118,200,147,.12);
  font-size: 10px;
  font-weight: 800;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: .04em;
}

.arm-svg-view {
  width: 100%;
  max-width: 240px;
  height: auto;
  filter: drop-shadow(0 8px 18px rgba(0,0,0,0.55));
}

.arm-canvas-footer {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
}

.arm-badge-row {
  display: flex;
  gap: 5px;
  justify-content: center;
  flex-wrap: wrap;
}

.arm-telemetry-pill {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(0,0,0,0.35);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 3px 8px;
  font-size: 9px;
  font-family: var(--mono);
  color: var(--text3);
}
.arm-telemetry-pill strong { color: var(--green-light); font-weight: 700; }

.arm-controls-card {
  background: linear-gradient(145deg, rgba(20,38,26,.95), rgba(11,25,16,.95));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow: 0 8px 24px rgba(0,0,0,.2);
}

.card-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2px;
  padding-bottom: 5px;
  border-bottom: 1px solid rgba(118,200,147,.12);
}

.card-title h2 {
  font-size: 11.5px;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--text2);
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-tag {
  font-size: 8.5px;
  font-weight: 800;
  padding: 2.5px 7px;
  border-radius: 8px;
  background: rgba(0,0,0,.3);
  color: var(--text3);
  border: 1px solid var(--border);
  font-family: var(--mono);
}

.status-tag.moving {
  color: var(--yellow);
  border-color: rgba(229,201,93,.4);
  animation: pulse 1s infinite;
}

.status-tag.seq {
  color: var(--sky);
  border-color: rgba(56,189,248,.4);
  animation: pulse 0.8s infinite;
}

/* Joint Sliders Grid */
.joints-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.joint-card-mit {
  background: rgba(0,0,0,0.24);
  border: 1px solid rgba(41,69,50,0.6);
  border-radius: 10px;
  padding: 5px 8px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  transition: border-color .15s ease;
}

.joint-card-mit:hover {
  border-color: rgba(95,207,114,0.4);
}

.joint-header-mit {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.joint-title-mit {
  font-size: 10.5px;
  font-weight: 800;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 5px;
}

/* Value display box above slider matching MIT App Inventor */
.value-display-box {
  min-width: 58px;
  height: 20px;
  padding: 0 6px;
  background: rgba(0,0,0,0.6);
  border: 1px solid rgba(118,200,147,0.35);
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 10.5px;
  font-weight: 800;
  color: var(--green-light);
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
}

.joint-controls {
  display: flex;
  align-items: center;
  gap: 6px;
}

.step-btn {
  width: 26px;
  height: 24px;
  border-radius: 6px;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  font-size: 10px;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .1s ease;
}

.step-btn:active {
  background: var(--green-dark);
  transform: scale(0.92);
}

.slider-wrap {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
}

input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  border-radius: 5px;
  background: rgba(255,255,255,.08);
  outline: none;
  touch-action: none;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 17px;
  height: 17px;
  border-radius: 50%;
  background: linear-gradient(145deg, var(--green-light), var(--green));
  cursor: pointer;
  box-shadow: 0 0 10px rgba(95,207,114,.6);
  border: 1.5px solid rgba(255,255,255,0.8);
}

/* MIT App Inventor Action Buttons Bar */
.app-action-bar {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-top: 2px;
}

.btn-app-action {
  padding: 7px 4px;
  border-radius: 8px;
  font-size: 10.5px;
  font-weight: 800;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  transition: all 0.15s ease;
  box-shadow: 0 4px 10px rgba(0,0,0,0.25);
  border: none;
}

.btn-app-action:active { transform: scale(0.93); }

.btn-save-mit  { background: linear-gradient(145deg, #2d6a3e, #1e4a2b); border: 1px solid #5fcf72; }
.btn-save-mit:hover { background: linear-gradient(145deg, #37824c, #245a33); }

.btn-run-mit   { background: linear-gradient(145deg, #2f8f45, #1f6830); border: 1px solid #8ee69b; }
.btn-run-mit:hover { background: linear-gradient(145deg, #3bb056, #29853e); }
.btn-run-mit.paused { background: linear-gradient(145deg, #a83f3d, #7c2d2b); border-color: #ef6461; }

.btn-reset-mit { background: linear-gradient(145deg, #3b4d61, #253342); border: 1px solid #64748b; }
.btn-reset-mit:hover { background: linear-gradient(145deg, #4b627c, #2f4154); }

.btn-clear-mit { background: linear-gradient(145deg, #5c2323, #3b1616); border: 1px solid #ef6461; color: #ffcdd2; }
.btn-clear-mit:hover { background: linear-gradient(145deg, #742d2d, #4a1c1c); }

/* Presets (Symmetrical 3x2 Grid & Action Grid) */
.presets-grid, .action-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 5px;
}

.preset-btn, .action-btn {
  padding: 6px 4px;
  border-radius: 8px;
  background: linear-gradient(145deg, rgba(26,49,31,.9), rgba(14,29,18,.9));
  border: 1px solid var(--border);
  color: var(--text2);
  font-size: 9.5px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  transition: all .12s ease;
}

.preset-btn span, .action-btn span { font-size: 11px; }

.preset-btn:active, .action-btn:active {
  transform: scale(0.94);
  background: rgba(95,207,114,.2);
  border-color: var(--green);
}

.btn-open   { border-color: rgba(56,189,248,.4); color: var(--sky); }
.btn-close  { border-color: rgba(239,100,97,.4); color: var(--red); }
.btn-home   { border-color: rgba(229,201,93,.4); color: var(--yellow); }
.btn-canopy { border-color: rgba(142,230,155,.4); color: var(--green-light); }
.btn-foliar { border-color: rgba(118,200,147,.4); color: var(--leaf); }
.btn-soil   { border-color: rgba(181,139,99,.4); color: var(--soil); }

/* Stored Waypoints Section */
.waypoints-section {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 2px;
}

.waypoints-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 3px 6px;
  font-size: 10px;
  font-weight: 800;
  color: var(--text2);
}

.wp-counter-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px;
  border-radius: 12px;
  background: rgba(0,0,0,0.5);
  border: 1px solid rgba(118,200,147,0.3);
  font-family: var(--mono);
  font-size: 9px;
  color: var(--text3);
}
.wp-counter-badge strong { color: var(--green-light); }

.waypoints-strip {
  max-height: 120px;
  overflow-y: auto;
  background: rgba(0,0,0,0.35);
  border: 1px solid rgba(41,69,50,0.7);
  border-radius: 9px;
  padding: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.waypoints-strip::-webkit-scrollbar { width: 5px; }
.waypoints-strip::-webkit-scrollbar-thumb { background: rgba(95,207,114,0.3); border-radius: 4px; }

.wp-empty {
  font-size: 9.5px;
  color: var(--text3);
  text-align: center;
  padding: 10px 0;
  font-style: italic;
}

.wp-chip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 8px;
  border-radius: 6px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  font-family: var(--mono);
  font-size: 9.5px;
  color: var(--text2);
  transition: all 0.12s ease;
  cursor: pointer;
}

.wp-chip:hover {
  background: rgba(95,207,114,0.08);
  border-color: rgba(95,207,114,0.35);
}

.wp-chip.active-step {
  background: rgba(56,189,248,0.15);
  border-color: rgba(56,189,248,0.6);
  color: #bae6fd;
  box-shadow: 0 0 10px rgba(56,189,248,0.2);
}

.wp-chip-num {
  font-weight: 800;
  color: var(--green-light);
  min-width: 24px;
}

.wp-chip-angles {
  display: flex;
  gap: 5px;
  color: var(--text2);
}

.wp-chip-angles span {
  background: rgba(0,0,0,0.35);
  padding: 1px 4px;
  border-radius: 3px;
  border: 1px solid rgba(255,255,255,0.05);
}

/* CHASSIS SPLIT DECK (Full Page Mission Control) */
.chassis-split-deck {
  display: grid;
  grid-template-columns: 310px 1fr;
  gap: 12px;
  align-items: stretch;
}

@media (max-width: 768px) {
  .chassis-split-deck {
    grid-template-columns: 1fr;
  }
}

.chassis-ctrl-card {
  background: linear-gradient(145deg, rgba(14,30,20,.95), rgba(9,20,13,.95));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 8px 24px rgba(0,0,0,.25);
}

.chassis-ops-card {
  background: linear-gradient(145deg, rgba(20,38,26,.95), rgba(11,25,16,.95));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  box-shadow: 0 8px 24px rgba(0,0,0,.2);
}

.card-title {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(118,200,147,.12);
}

.card-title h2 {
  font-size: 11.5px;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--text2);
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-tag {
  font-size: 8.5px;
  font-weight: 800;
  padding: 2.5px 7px;
  border-radius: 8px;
  background: rgba(0,0,0,.3);
  color: var(--text3);
  border: 1px solid var(--border);
  font-family: var(--mono);
}

/* Tactical D-Pad */
.chassis-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 18px 0;
  position: relative;
}

.ctrl-ring {
  position: absolute;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  border: 2px dashed rgba(95,207,114,.18);
  pointer-events: none;
}

.ctrl-ring.on {
  border-color: rgba(95,207,114,.6);
  box-shadow: 0 0 25px rgba(95,207,114,.15);
  animation: spin 16s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.dpad {
  position: relative;
  width: 185px;
  height: 185px;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(3, 1fr);
  gap: 6px;
}

.dir {
  border: 1px solid var(--border);
  background: linear-gradient(145deg, rgba(26,49,31,.98), rgba(13,29,18,.98));
  color: var(--text2);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .12s ease;
  box-shadow: 0 4px 12px rgba(0,0,0,.25);
  cursor: pointer;
}

.dir-up    { grid-column: 2; grid-row: 1; border-radius: 14px 14px 6px 6px; }
.dir-left  { grid-column: 1; grid-row: 2; border-radius: 14px 6px 6px 14px; }
.dir-right { grid-column: 3; grid-row: 2; border-radius: 6px 14px 14px 6px; }
.dir-down  { grid-column: 2; grid-row: 3; border-radius: 6px 6px 14px 14px; }

.dir:active, .dir.active {
  transform: scale(.92);
  background: rgba(95,207,114,.25);
  border-color: var(--green-light);
  color: var(--green-light);
}

.dir svg {
  width: 22px;
  height: 22px;
  stroke: currentColor;
  fill: none;
  stroke-width: 2.2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.stop-btn {
  grid-column: 2;
  grid-row: 2;
  border-radius: 50%;
  border: 1px solid rgba(239,100,97,.55);
  background: linear-gradient(145deg, #a83f3d, #702c2b);
  color: white;
  font-size: 10px;
  font-weight: 900;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 16px rgba(239,100,97,.2);
  cursor: pointer;
}

.stop-btn:active { transform: scale(.9); box-shadow: 0 0 25px rgba(239,100,97,.5); }

.chassis-feedback-box {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 4px;
  background: rgba(0,0,0,0.35);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
  margin-top: 24px;
  position: relative;
  z-index: 2;
}

.chassis-hud-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text3);
}

.chassis-hud-row strong { color: var(--green-light); font-weight: 800; }

.chassis-hint {
  font-size: 8.5px;
  color: var(--text3);
  text-align: center;
  opacity: 0.8;
  margin-top: 2px;
}

/* Headlight Card */
.led-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  background: linear-gradient(145deg, rgba(25,48,30,.96), rgba(13,29,18,.96));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all 0.15s ease;
}

.led-card:hover { border-color: rgba(95,207,114,.35); }

.led-card.on {
  border-color: rgba(95,207,114,.6);
  box-shadow: inset 0 0 20px rgba(95,207,114,.12);
}

.led-title { display: flex; align-items: center; gap: 10px; }
.led-icon { font-size: 20px; }

.toggle { width: 42px; height: 24px; border-radius: 20px; background: rgba(255,255,255,.1); position: relative; transition: background .2s ease; }
.toggle.on { background: var(--green-dark); }
.thumb { width: 20px; height: 20px; border-radius: 50%; background: white; position: absolute; top: 2px; left: 2px; transition: left .18s ease; box-shadow: 0 2px 6px rgba(0,0,0,.3); }
.toggle.on .thumb { left: 20px; }

/* Section Label */
.section-label {
  font-size: 9px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--text3);
  margin-bottom: 4px;
  display: block;
}

/* Quick Row Maneuvers */
.chassis-maneuvers-box {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.action-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
}

.action-btn {
  padding: 7px 4px;
  border-radius: 8px;
  background: linear-gradient(145deg, rgba(26,49,31,.9), rgba(14,29,18,.9));
  border: 1px solid var(--border);
  color: var(--text2);
  font-size: 9.5px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  transition: all .12s ease;
}

.action-btn:active {
  transform: scale(0.94);
  background: rgba(95,207,114,.2);
  border-color: var(--green);
}

/* Throttle & Gear Selector */
.chassis-gears-box {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.gear-selector-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}

.gear-btn {
  padding: 8px 6px;
  border-radius: 8px;
  background: rgba(0,0,0,0.3);
  border: 1px solid var(--border);
  color: var(--text3);
  font-size: 9.5px;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  transition: all .15s ease;
}

.gear-btn.active {
  background: linear-gradient(145deg, #2d6a3e, #1e4a2b);
  border-color: var(--green-light);
  color: var(--text);
  box-shadow: 0 2px 10px rgba(95,207,114,0.25);
}

/* Chassis Hardware Telemetry Grid */
.chassis-telemetry-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-top: 2px;
}

.chassis-telem-item {
  background: rgba(0,0,0,0.35);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 4px;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.chassis-telem-item .lbl {
  font-size: 7.5px;
  font-weight: 800;
  text-transform: uppercase;
  color: var(--text3);
}

.chassis-telem-item .val {
  font-size: 9.5px;
  font-family: var(--mono);
  font-weight: 800;
  color: var(--text2);
}

.chassis-telem-item .val.go { color: var(--green-light); }
.chassis-telem-item .val.arm { color: var(--sky); }

/* REBUILT, CRYSTAL-CLEAR ACTIVITY TERMINAL */
.log-box {
  background: linear-gradient(145deg, rgba(10,22,14,.96), rgba(5,14,9,.98));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 14px;
  box-shadow: 0 6px 20px rgba(0,0,0,.3);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(118,200,147,.16);
  padding-bottom: 5px;
  color: var(--text2);
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .06em;
}

.log-header-left {
  display: flex;
  align-items: center;
  gap: 6px;
}

.log-pulse-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--green-light);
  box-shadow: 0 0 8px rgba(142,230,155,0.8);
  animation: pulse 1.2s infinite;
}

.log-clock {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--green-light);
  font-weight: 800;
}

.log-body {
  height: 90px;
  overflow-y: auto;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.5;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.log-body::-webkit-scrollbar { width: 5px; }
.log-body::-webkit-scrollbar-thumb { background: rgba(95,207,114,.35); border-radius: 4px; }

.log-line {
  display: flex;
  align-items: baseline;
  gap: 8px;
  white-space: nowrap;
}

.log-time {
  color: var(--text3);
  font-size: 10px;
  min-width: 44px;
  opacity: 0.85;
}

.log-msg {
  font-weight: 600;
  letter-spacing: 0.01em;
}

.c-chassis { color: #8ee69b; font-weight: 700; }
.c-arm     { color: #38bdf8; font-weight: 700; }
.c-stop    { color: #ef6461; font-weight: 700; }
.c-led     { color: #e5c95d; font-weight: 700; }

/* Dual Mode Layout */
.dual-layout {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 10px;
}

@media (max-width: 680px) {
  .dual-layout { grid-template-columns: 1fr; }
  .stats { grid-template-columns: repeat(3, 1fr); }
  .action-grid { grid-template-columns: repeat(2, 1fr); }
}

footer {
  text-align: center;
  font-size: 8.5px;
  font-weight: 700;
  color: var(--text3);
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 4px 0 2px;
}
footer span { color: var(--green-light); font-weight: 800; }
</style>
</head>

<body oncontextmenu="return false;">
<div class="app">

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="logo-box">
        <svg width="22" height="22" viewBox="0 0 48 48" fill="none">
          <path d="M37 8C24 9 14 15 12 26C11 33 14 38 20 39C29 40 37 29 37 8Z" stroke="#8ee69b" stroke-width="2.5"/>
          <path d="M12 35C19 27 25 21 34 14" stroke="#8ee69b" stroke-width="2"/>
          <rect x="8" y="31" width="32" height="8" rx="3" stroke="#8ee69b" stroke-width="2"/>
          <circle cx="15" cy="42" r="2.5" fill="#8ee69b"/>
          <circle cx="33" cy="42" r="2.5" fill="#8ee69b"/>
        </svg>
      </div>
      <div>
        <h1>AgriRover & 4-DOF Arm</h1>
        <p>Team Innovex • SIH 2026</p>
      </div>
    </div>

    <div class="header-badges">
      <div class="pill">
        <span class="dot live" id="connDot"></span>
        <span id="connText">Linked</span>
      </div>
      <div class="pill">
        <span class="dot arm"></span>
        <span id="armHeaderBadge">PCA9685</span>
      </div>
    </div>
  </header>

  <!-- Mode Tab Switcher -->
  <div class="mode-tabs">
    <button class="tab-btn active" id="tabChassisBtn" onclick="setConsoleTab('chassis')">
      🚜 Chassis
    </button>
    <button class="tab-btn" id="tabArmBtn" onclick="setConsoleTab('arm')">
      🦾 4-DOF Arm
    </button>
    <button class="tab-btn" id="tabDualBtn" onclick="setConsoleTab('dual')">
      ⚡ Dual Console
    </button>
  </div>

  <!-- Telemetry Ribbon -->
  <div class="stats">
    <div class="stat">
      <div class="lbl">Chassis</div>
      <div class="val" id="stRoverCmd">Idle</div>
    </div>
    <div class="stat">
      <div class="lbl">Arm Motion</div>
      <div class="val arm" id="stArmMotion">Idle</div>
    </div>
    <div class="stat">
      <div class="lbl">Gripper</div>
      <div class="val" id="stGripper">Open</div>
    </div>
    <div class="stat">
      <div class="lbl">Speed</div>
      <div class="val go" id="stSpeed">70%</div>
    </div>
    <div class="stat">
      <div class="lbl">Signal</div>
      <div class="val" id="stRssi">--</div>
    </div>
    <div class="stat">
      <div class="lbl">Waypoints</div>
      <div class="val arm" id="stWaypoints">0 / 20</div>
    </div>
  </div>

  <!-- TAB 1: CHASSIS DRIVE (Full Page Dashboard with Zero Vague Space) -->
  <div class="tab-content active" id="tabChassis">
    <div class="chassis-split-deck">
      <!-- LEFT: Tactical Steering & Radar Controller -->
      <div class="chassis-ctrl-card">
        <div class="card-title">
          <h2>🚜 Tactical Navigation</h2>
          <span class="status-tag" id="chassisStatusBadge">STANDBY</span>
        </div>

        <div class="chassis-box">
          <div style="position:relative; display:flex; align-items:center; justify-content:center;">
            <div class="ctrl-ring" id="dpadRing"></div>
            <div class="dpad" id="dpad">
              <button class="dir dir-up" data-cmd="forward" aria-label="Forward" oncontextmenu="return false;">
                <svg viewBox="0 0 24 24"><polyline points="6 15 12 9 18 15"/></svg>
              </button>
              <button class="dir dir-left" data-cmd="left" aria-label="Left" oncontextmenu="return false;">
                <svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"/></svg>
              </button>
              <button class="stop-btn" data-cmd="stop" aria-label="Stop" oncontextmenu="return false;">
                STOP
              </button>
              <button class="dir dir-right" data-cmd="right" aria-label="Right" oncontextmenu="return false;">
                <svg viewBox="0 0 24 24"><polyline points="9 6 15 12 9 18"/></svg>
              </button>
              <button class="dir dir-down" data-cmd="backward" aria-label="Backward" oncontextmenu="return false;">
                <svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
              </button>
            </div>
          </div>
        </div>

        <div class="chassis-feedback-box">
          <div class="chassis-hud-row">
            <span>Vector: <strong id="hudVector">BRAKE</strong></span>
            <span>Speed: <strong id="hudSpeed">70%</strong></span>
          </div>
          <p class="chassis-hint">Press &amp; hold direction to propel &bull; Release to brake</p>
        </div>
      </div>

      <!-- RIGHT: Field Operations, Lighting, Quick Maneuvers & Telemetry -->
      <div class="chassis-ops-card">
        <div class="card-title">
          <h2>🌾 Field Operations &amp; Inspection</h2>
          <span class="status-tag" style="color:var(--green-light);">DRIVE READY</span>
        </div>

        <!-- Headlight Toggle Row -->
        <div class="led-card on" id="ledRow" onclick="toggleHeadlight()" oncontextmenu="return false;">
          <div class="led-title">
            <span class="led-icon">💡</span>
            <div>
              <div style="font-size:11px; font-weight:800;">Ultra-Bright Field Headlight</div>
              <div style="font-size:8.5px; color:var(--text3); text-transform:uppercase;">Canopy &amp; foliar night inspection beam</div>
            </div>
          </div>
          <div class="toggle on" id="ledToggle" oncontextmenu="return false;">
            <div class="thumb"></div>
          </div>
        </div>

        <!-- Quick Row Maneuvers -->
        <div class="chassis-maneuvers-box">
          <span class="section-label">Field Row Maneuvers</span>
          <div class="action-grid">
            <button type="button" class="action-btn" onclick="quickManeuver('pivot_left')"><span>↺</span> Pivot L 45°</button>
            <button type="button" class="action-btn" onclick="quickManeuver('pivot_right')"><span>↻</span> Pivot R 45°</button>
            <button type="button" class="action-btn" onclick="quickManeuver('u_turn')"><span>🔄</span> 180° Turn</button>
            <button type="button" class="action-btn" style="border-color:rgba(239,100,97,0.4); color:var(--red);" onclick="quickManeuver('stop')"><span>🛑</span> Full Stop</button>
          </div>
        </div>

        <!-- Chassis Drive Gears -->
        <div class="chassis-gears-box">
          <span class="section-label">Chassis Throttle &amp; Speed Preset</span>
          <div class="gear-selector-row">
            <button type="button" class="gear-btn" id="gearSlow" onclick="setDriveGear(35)">🐢 Precision 35%</button>
            <button type="button" class="gear-btn active" id="gearNormal" onclick="setDriveGear(70)">🚜 Standard 70%</button>
            <button type="button" class="gear-btn" id="gearFast" onclick="setDriveGear(100)">⚡ Turbo 100%</button>
          </div>
        </div>

        <!-- Hardware & Motor Driver Telemetry Diagnostics -->
        <div class="chassis-telemetry-grid">
          <div class="chassis-telem-item">
            <span class="lbl">H-Bridge Motors</span>
            <span class="val go">4x L298N OK</span>
          </div>
          <div class="chassis-telem-item">
            <span class="lbl">Row Steering</span>
            <span class="val arm">Differential</span>
          </div>
          <div class="chassis-telem-item">
            <span class="lbl">Link Latency</span>
            <span class="val go">&lt; 15 ms</span>
          </div>
          <div class="chassis-telem-item">
            <span class="lbl">Power Bus</span>
            <span class="val">5V Rail OK</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: ROBOTIC ARM (MIT App Inventor Layout and Centered Industrial Arm Diagram) -->
  <div class="tab-content" id="tabArm">
    <!-- Top App Banner -->
    <div class="app-arm-banner">
      <div class="app-arm-title"><span>🦾</span> 4-DOF Robotic Arm Mission Deck</div>
      <div class="app-conn-group">
        <span class="app-status-badge connected" id="lblArmStatus"><span class="dot live" style="display:inline-block; margin-right:4px;"></span>PCA9685 I2C Bus Active</span>
      </div>
    </div>

    <div class="arm-split-deck">
      <!-- LEFT: Complete Robotic Arm Graphic Canvas (NO CALLOUT LINES, CENTERED, HIGH-PRECISION) -->
      <div class="arm-canvas-card">
        <div class="arm-canvas-header">
          <span>Kinematic Model</span>
          <span style="color:var(--sky);">4-DOF ARTICULATED</span>
        </div>

        <svg class="arm-svg-view" id="armSvgView" viewBox="0 0 240 380" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="titaniumMetal" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#334155"/>
              <stop offset="50%" stop-color="#1e293b"/>
              <stop offset="100%" stop-color="#0f172a"/>
            </linearGradient>
            <linearGradient id="armLinkGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#0284c7"/>
              <stop offset="60%" stop-color="#0369a1"/>
              <stop offset="100%" stop-color="#075985"/>
            </linearGradient>
            <linearGradient id="metalSpecular" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#94a3b8"/>
              <stop offset="40%" stop-color="#f8fafc"/>
              <stop offset="70%" stop-color="#cbd5e1"/>
              <stop offset="100%" stop-color="#64748b"/>
            </linearGradient>
            <linearGradient id="basePlateGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#1e3a29"/>
              <stop offset="50%" stop-color="#0d2417"/>
              <stop offset="100%" stop-color="#05120a"/>
            </linearGradient>
            <filter id="jointGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" result="blur"/>
              <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
            <radialGradient id="armBackdropAura" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="rgba(56,189,248,0.12)"/>
              <stop offset="100%" stop-color="rgba(0,0,0,0)"/>
            </radialGradient>
          </defs>

          <!-- Subtle Circular Backdrop Aura -->
          <circle cx="120" cy="190" r="110" fill="url(#armBackdropAura)"/>

          <!-- Ground Contact Shadow -->
          <ellipse cx="120" cy="364" rx="80" ry="12" fill="rgba(0,0,0,0.6)"/>

          <!-- Pedestal Platform Base -->
          <ellipse cx="120" cy="354" rx="72" ry="15" fill="url(#basePlateGrad)" stroke="#294532" stroke-width="1.8"/>
          <ellipse cx="120" cy="349" rx="58" ry="11" fill="url(#titaniumMetal)" stroke="#334155" stroke-width="1.2"/>

          <!-- Azimuth Degree Marks on Base -->
          <path d="M 68 350 L 74 353" stroke="rgba(142,230,155,0.4)" stroke-width="1.2"/>
          <path d="M 90 354 L 94 357" stroke="rgba(142,230,155,0.4)" stroke-width="1.2"/>
          <path d="M 120 356 L 120 360" stroke="rgba(142,230,155,0.7)" stroke-width="1.5"/>
          <path d="M 146 354 L 150 357" stroke="rgba(142,230,155,0.4)" stroke-width="1.2"/>
          <path d="M 166 350 L 172 353" stroke="rgba(142,230,155,0.4)" stroke-width="1.2"/>

          <!-- BASE TURNTABLE & SERVO 01 HOUSING -->
          <g id="svgBaseGroup">
            <path d="M 84 326 L 156 326 L 164 348 L 76 348 Z" fill="url(#armLinkGrad)" stroke="#0284c7" stroke-width="1.5"/>
            <rect x="96" y="302" width="48" height="26" rx="4" fill="url(#titaniumMetal)" stroke="#475569" stroke-width="1.5"/>
            <line x1="102" y1="308" x2="102" y2="322" stroke="#64748b" stroke-width="1.2"/>
            <line x1="108" y1="308" x2="108" y2="322" stroke="#64748b" stroke-width="1.2"/>
            <line x1="114" y1="308" x2="114" y2="322" stroke="#64748b" stroke-width="1.2"/>
            <line x1="126" y1="308" x2="126" y2="322" stroke="#64748b" stroke-width="1.2"/>
            <line x1="132" y1="308" x2="132" y2="322" stroke="#64748b" stroke-width="1.2"/>
            <line x1="138" y1="308" x2="138" y2="322" stroke="#64748b" stroke-width="1.2"/>
            <circle cx="120" cy="315" r="7" fill="url(#metalSpecular)" stroke="#1e293b" stroke-width="1"/>
            <circle cx="120" cy="315" r="3" fill="#0f172a"/>
            <circle cx="120" cy="315" r="4" fill="#5fcf72" filter="url(#jointGlow)"/>
          </g>

          <!-- SHOULDER LINK & SERVO 02 ASSEMBLY -->
          <g id="svgShoulderGroup">
            <path d="M 106 302 L 134 302 L 138 214 L 102 214 Z" fill="url(#armLinkGrad)" stroke="#38bdf8" stroke-width="1.6"/>
            <path d="M 112 284 L 128 284 L 126 232 L 114 232 Z" fill="#09140d" stroke="#163220" stroke-width="1.2"/>
            <rect x="86" y="262" width="22" height="30" rx="3" fill="url(#titaniumMetal)" stroke="#475569" stroke-width="1.2"/>
            <circle cx="97" cy="277" r="5.5" fill="url(#metalSpecular)"/>
            <circle cx="97" cy="277" r="2.5" fill="#0f172a"/>
            <circle cx="108" cy="296" r="2" fill="#94a3b8"/>
            <circle cx="132" cy="296" r="2" fill="#94a3b8"/>
            <circle cx="120" cy="214" r="16" fill="url(#titaniumMetal)" stroke="#38bdf8" stroke-width="2"/>
            <circle cx="120" cy="214" r="9" fill="url(#metalSpecular)" stroke="#334155" stroke-width="1"/>
            <circle cx="120" cy="214" r="4" fill="#0f172a"/>
            <circle cx="120" cy="214" r="5" fill="#5fcf72" filter="url(#jointGlow)"/>
          </g>

          <!-- FOREARM LINK & SERVO 03 (ELBOW ASSEMBLY) -->
          <g id="svgElbowGroup">
            <path d="M 108 214 L 132 214 L 130 124 L 110 124 Z" fill="url(#armLinkGrad)" stroke="#38bdf8" stroke-width="1.6"/>
            <line x1="120" y1="202" x2="120" y2="136" stroke="#0284c7" stroke-width="3" stroke-linecap="round"/>
            <line x1="120" y1="202" x2="120" y2="136" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="4,3"/>
            <rect x="88" y="160" width="22" height="30" rx="3" fill="url(#titaniumMetal)" stroke="#475569" stroke-width="1.2"/>
            <circle cx="99" cy="175" r="5.5" fill="url(#metalSpecular)"/>
            <circle cx="99" cy="175" r="2.5" fill="#0f172a"/>
            <circle cx="120" cy="124" r="14" fill="url(#titaniumMetal)" stroke="#38bdf8" stroke-width="2"/>
            <circle cx="120" cy="124" r="7.5" fill="url(#metalSpecular)" stroke="#334155" stroke-width="1"/>
            <circle cx="120" cy="124" r="3" fill="#0f172a"/>
            <circle cx="120" cy="124" r="4.5" fill="#5fcf72" filter="url(#jointGlow)"/>
          </g>

          <!-- WRIST ASSEMBLY & DUAL-JAW MECHANICAL GRIPPER (Servo 06) -->
          <g id="svgGripperGroup">
            <rect x="109" y="86" width="22" height="38" rx="4" fill="url(#titaniumMetal)" stroke="#0284c7" stroke-width="1.6"/>
            <circle cx="120" cy="86" r="8" fill="url(#titaniumMetal)" stroke="#475569"/>
            <circle cx="120" cy="72" r="12" fill="url(#metalSpecular)" stroke="#334155" stroke-width="1.5"/>
            <circle cx="120" cy="72" r="5" fill="#0f172a"/>

            <!-- Left Articulated Claw Finger (Pivot 112, 68) -->
            <g id="svgClawLeft" transform="rotate(0 112 68)">
              <path d="M 112 68 Q 84 48 94 14 Q 100 12 104 18 Q 98 48 118 62 Z" fill="url(#metalSpecular)" stroke="#334155" stroke-width="1.3"/>
              <path d="M 104 18 Q 99 44 116 60 L 114 63 Q 95 46 101 16 Z" fill="#0f172a"/>
              <line x1="104" y1="28" x2="108" y2="30" stroke="#38bdf8" stroke-width="1"/>
              <line x1="107" y1="36" x2="111" y2="38" stroke="#38bdf8" stroke-width="1"/>
              <line x1="110" y1="44" x2="114" y2="46" stroke="#38bdf8" stroke-width="1"/>
              <circle cx="112" cy="68" r="3.2" fill="#ef4444" stroke="#991b1b" stroke-width="0.8"/>
            </g>

            <!-- Right Articulated Claw Finger (Pivot 128, 68) -->
            <g id="svgClawRight" transform="rotate(0 128 68)">
              <path d="M 128 68 Q 156 48 146 14 Q 140 12 136 18 Q 142 48 122 62 Z" fill="url(#metalSpecular)" stroke="#334155" stroke-width="1.3"/>
              <path d="M 136 18 Q 141 44 124 60 L 126 63 Q 145 46 139 16 Z" fill="#0f172a"/>
              <line x1="136" y1="28" x2="132" y2="30" stroke="#38bdf8" stroke-width="1"/>
              <line x1="133" y1="36" x2="129" y2="38" stroke="#38bdf8" stroke-width="1"/>
              <line x1="130" y1="44" x2="126" y2="46" stroke="#38bdf8" stroke-width="1"/>
              <circle cx="128" cy="68" r="3.2" fill="#ef4444" stroke="#991b1b" stroke-width="0.8"/>
            </g>

            <!-- Wrist Central Hub Active LED -->
            <circle cx="120" cy="72" r="4.5" fill="#5fcf72" filter="url(#jointGlow)"/>
          </g>
        </svg>

        <!-- Canvas Footer Info -->
        <div class="arm-canvas-footer">
          <div class="arm-badge-row">
            <span class="status-tag" id="armMotionTag">ARM IDLE</span>
            <span class="status-tag" style="color:var(--sky);">PCA9685 0x40</span>
            <span class="status-tag" style="color:var(--green-light);">4-DOF READY</span>
          </div>
          <div class="arm-telemetry-pill">
            <span>Stance:</span>
            <span id="lblStance">B:<strong>90°</strong> S:<strong>90°</strong> E:<strong>90°</strong> G:<strong>180°</strong></span>
          </div>
        </div>
      </div>

      <!-- RIGHT: Joint Sliders and MIT App Inventor Action Blocks -->
      <div class="arm-controls-card">
        <div class="card-title">
          <h2>🦾 Joint Calibration &amp; Motion Plan</h2>
          <span class="status-tag" id="armStatusTag">IDLE</span>
        </div>

        <div class="joints-grid">
          <!-- Joint 1: Grip (Servo 06) -->
          <div class="joint-card-mit" id="jointCardGrip">
            <div class="joint-header-mit">
              <span class="joint-title-mit">🦀 Grip (Servo 06)</span>
              <div class="value-display-box" id="valBoxGrip">180°</div>
            </div>
            <div class="joint-controls">
              <button class="step-btn" onclick="stepJoint('gripper', -5)">-5°</button>
              <div class="slider-wrap">
                <input type="range" id="sliderGripper" min="70" max="180" value="180" oninput="onJointInput('gripper', this.value)" onchange="onJointCommit('gripper', this.value)">
              </div>
              <button class="step-btn" onclick="stepJoint('gripper', 5)">+5°</button>
            </div>
          </div>

          <!-- Joint 2: Elbow (Servo 03) -->
          <div class="joint-card-mit" id="jointCardElbow">
            <div class="joint-header-mit">
              <span class="joint-title-mit">⚡ Elbow (Servo 03)</span>
              <div class="value-display-box" id="valBoxElbow">90°</div>
            </div>
            <div class="joint-controls">
              <button class="step-btn" onclick="stepJoint('elbow', -5)">-5°</button>
              <div class="slider-wrap">
                <input type="range" id="sliderElbow" min="0" max="180" value="90" oninput="onJointInput('elbow', this.value)" onchange="onJointCommit('elbow', this.value)">
              </div>
              <button class="step-btn" onclick="stepJoint('elbow', 5)">+5°</button>
            </div>
          </div>

          <!-- Joint 3: Shoulder (Servo 02) -->
          <div class="joint-card-mit" id="jointCardShoulder">
            <div class="joint-header-mit">
              <span class="joint-title-mit">📐 Shoulder (Servo 02)</span>
              <div class="value-display-box" id="valBoxShoulder">90°</div>
            </div>
            <div class="joint-controls">
              <button class="step-btn" onclick="stepJoint('shoulder', -5)">-5°</button>
              <div class="slider-wrap">
                <input type="range" id="sliderShoulder" min="0" max="180" value="90" oninput="onJointInput('shoulder', this.value)" onchange="onJointCommit('shoulder', this.value)">
              </div>
              <button class="step-btn" onclick="stepJoint('shoulder', 5)">+5°</button>
            </div>
          </div>

          <!-- Joint 4: Base (Servo 01) -->
          <div class="joint-card-mit" id="jointCardBase">
            <div class="joint-header-mit">
              <span class="joint-title-mit">🔄 Base (Servo 01)</span>
              <div class="value-display-box" id="valBoxBase">90°</div>
            </div>
            <div class="joint-controls">
              <button class="step-btn" onclick="stepJoint('base', -5)">-5°</button>
              <div class="slider-wrap">
                <input type="range" id="sliderBase" min="5" max="175" value="90" oninput="onJointInput('base', this.value)" onchange="onJointCommit('base', this.value)">
              </div>
              <button class="step-btn" onclick="stepJoint('base', 5)">+5°</button>
            </div>
          </div>

          <!-- Speed Control (ss) -->
          <div class="joint-card-mit" id="jointCardSpeed">
            <div class="joint-header-mit">
              <span class="joint-title-mit">🚀 Speed &bull; Motion Slew Rate</span>
              <div class="value-display-box" id="valBoxSpeed">70%</div>
            </div>
            <div class="slider-wrap" style="margin-top:2px;">
              <input type="range" id="sliderSpeed" min="5" max="100" value="70" oninput="onSpeedInput(this.value)" onchange="onSpeedCommit(this.value)">
            </div>
          </div>
        </div>

        <!-- MIT App Inventor Primary Action Buttons -->
        <div class="app-action-bar">
          <button type="button" class="btn-app-action btn-save-mit" id="btnAppSave" onclick="onAppSaveClick()">💾 SAVE</button>
          <button type="button" class="btn-app-action btn-run-mit" id="btnAppRun" onclick="onAppRunClick()">▶️ RUN</button>
          <button type="button" class="btn-app-action btn-reset-mit" id="btnAppReset" onclick="onAppResetClick()">🔄 RESET</button>
          <button type="button" class="btn-app-action btn-clear-mit" id="btnAppClear" onclick="onAppClearClick()">🗑️ CLEAR</button>
        </div>

        <!-- Quick Agricultural Presets (Symmetrical 3x2 Grid) -->
        <div class="presets-grid">
          <button class="preset-btn btn-open" onclick="armCommand('open')"><span>✋</span> Open Grip</button>
          <button class="preset-btn btn-close" onclick="armCommand('close')"><span>✊</span> Close Grip</button>
          <button class="preset-btn btn-home" onclick="armPreset('home')"><span>🏠</span> Safe Home</button>
          <button class="preset-btn btn-canopy" onclick="armPreset('inspect_canopy')"><span>🔍</span> Canopy High</button>
          <button class="preset-btn btn-foliar" onclick="armPreset('inspect_foliar')"><span>🌿</span> Foliar Mid</button>
          <button class="preset-btn btn-soil" onclick="armPreset('sample_ground')"><span>🧪</span> Soil Sample</button>
        </div>

        <!-- Recorded Waypoints Strip with Header -->
        <div class="waypoints-section">
          <div class="waypoints-header">
            <span>📍 Stored Trajectory Waypoints</span>
            <span class="wp-counter-badge"><strong id="lblPositions">0</strong> / 20 Recorded</span>
          </div>
          <div class="waypoints-strip" id="waypointsStrip">
            <div class="wp-empty">No recorded waypoints yet. Click <strong>SAVE</strong> to store current stance.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 3: DUAL CONSOLE -->
  <div class="tab-content" id="tabDual">
    <div class="dual-layout">
      <!-- Left: Compact Chassis -->
      <div class="arm-card" style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
        <div class="card-title" style="width:100%;">
          <h2>🚜 Chassis Steering</h2>
          <button class="step-btn" style="width:auto; padding:2px 8px;" onclick="toggleHeadlight()">💡 Light</button>
        </div>
        <div style="position:relative; display:flex; align-items:center; justify-content:center; margin:10px 0;">
          <div class="dpad" id="dualDpad" style="width:150px; height:150px;">
            <button class="dir dir-up" data-cmd="forward"><svg viewBox="0 0 24 24"><polyline points="6 15 12 9 18 15"/></svg></button>
            <button class="dir dir-left" data-cmd="left"><svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"/></svg></button>
            <button class="stop-btn" data-cmd="stop">STOP</button>
            <button class="dir dir-right" data-cmd="right"><svg viewBox="0 0 24 24"><polyline points="9 6 15 12 9 18"/></svg></button>
            <button class="dir dir-down" data-cmd="backward"><svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg></button>
          </div>
        </div>
      </div>

      <!-- Right: Compact Arm Quick Controls -->
      <div class="arm-card">
        <div class="card-title">
          <h2>🦾 Quick Arm Access</h2>
          <button class="step-btn" style="width:auto; padding:2px 8px;" onclick="armPreset('home')">🏠 Home</button>
        </div>
        <div class="action-grid" style="grid-template-columns: 1fr 1fr; margin-bottom:8px;">
          <button class="action-btn btn-open" onclick="armCommand('open')"><span>✋</span> Open Grip</button>
          <button class="action-btn btn-close" onclick="armCommand('close')"><span>✊</span> Close Grip</button>
          <button class="action-btn" onclick="armPreset('inspect_foliar')"><span>🌿</span> Foliar View</button>
          <button class="action-btn" onclick="armPreset('sample_ground')"><span>🧪</span> Soil Sample</button>
        </div>
        <div class="joint-controls" style="margin-top:6px;">
          <button class="step-btn" style="flex:1;" onclick="stepJoint('base', -15)">↺ Base Left</button>
          <button class="step-btn" style="flex:1;" onclick="stepJoint('base', 15)">Base Right ↻</button>
        </div>
        <div class="joint-controls" style="margin-top:6px;">
          <button class="step-btn" style="flex:1;" onclick="stepJoint('shoulder', 10)">↑ Shoulder Up</button>
          <button class="step-btn" style="flex:1;" onclick="stepJoint('shoulder', -10)">↓ Shoulder Down</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Rebuilt, Crisp & Visible Activity Terminal -->
  <div class="log-box">
    <div class="log-header">
      <div class="log-header-left">
        <span class="log-pulse-dot"></span>
        <span>AgriRover &bull; 4-DOF Robotic Arm Activity Log</span>
      </div>
      <span class="log-clock" id="termClock">00:00:00</span>
    </div>
    <div class="log-body" id="termBody"></div>
  </div>

  <!-- Footer -->
  <footer>
    AgriRover + 4-DOF Robotic Arm Master System • Team <span>Innovex</span>
  </footer>

</div>

<script>
(() => {
  "use strict";

  // 1. Globally prevent context menu / copy-paste bubble on tap-and-hold
  window.addEventListener("contextmenu", e => { e.preventDefault(); return false; }, { passive: false });
  document.addEventListener("contextmenu", e => { e.preventDefault(); return false; }, { passive: false });

  // 2. Prevent pinch zoom gestures
  document.addEventListener("gesturestart", e => e.preventDefault(), { passive: false });
  document.addEventListener("gesturechange", e => e.preventDefault(), { passive: false });
  document.addEventListener("gestureend", e => e.preventDefault(), { passive: false });

  // 3. Prevent double-tap-to-zoom
  let lastTouchEnd = 0;
  document.addEventListener("touchend", e => {
    const now = Date.now();
    if (now - lastTouchEnd <= 300) { e.preventDefault(); }
    lastTouchEnd = now;
  }, { passive: false });

  // 4. Prevent Ctrl+Wheel zoom
  document.addEventListener("wheel", e => {
    if (e.ctrlKey) e.preventDefault();
  }, { passive: false });

  const $ = id => document.getElementById(id);

  let activeCmd = null;
  let hbTimer = null;
  let connected = true;
  let cmdCount = 0;
  let armState = {
    base: 90,
    shoulder: 90,
    elbow: 90,
    gripper: 180,
    speed: 70,
    isMoving: false,
    seqRunning: false,
    savedCount: 0
  };

  const startTime = Date.now();
  const MAX_LOG = 30;

  function log(text, cls) {
    const body = $("termBody");
    if (!body) return;
    const el = document.createElement("div");
    el.className = "log-line";
    const sec = ((Date.now() - startTime) / 1000).toFixed(1);
    el.innerHTML = `<span class="log-time">+${sec}s</span><span class="log-msg ${cls || ''}">${text}</span>`;
    body.appendChild(el);
    while (body.children.length > MAX_LOG) body.removeChild(body.firstChild);
    body.scrollTop = body.scrollHeight;
  }

  log("AgriRover & Robotic Arm Master Online (SIH 2026)", "c-chassis");

  function updateClock() {
    const d = new Date();
    const clockEl = $("termClock");
    if (clockEl) {
      clockEl.textContent = [d.getHours(), d.getMinutes(), d.getSeconds()]
        .map(n => String(n).padStart(2, '0')).join(':');
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Tab switching
  window.setConsoleTab = function(mode) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    if (mode === 'chassis') {
      $("tabChassisBtn").classList.add('active');
      $("tabChassis").classList.add('active');
    } else if (mode === 'arm') {
      $("tabArmBtn").classList.add('active');
      $("tabArm").classList.add('active');
    } else if (mode === 'dual') {
      $("tabDualBtn").classList.add('active');
      $("tabDual").classList.add('active');
    }
  };

  // HTTP Helper
  function api(path) {
    return fetch(path, { cache: "no-store" })
      .then(res => {
        if (!res.ok) throw new Error("HTTP " + res.status);
        markConn(true);
        return res;
      })
      .catch(err => {
        markConn(false);
        throw err;
      });
  }

  function markConn(ok) {
    if (ok === connected) return;
    connected = ok;
    $("connDot").className = "dot " + (ok ? "live" : "dead");
    $("connText").textContent = ok ? "Linked" : "Offline";
  }

  // Chassis Movement
  function sendChassis(cmd, heartbeat = false) {
    api("/" + cmd).catch(() => {});
    const moving = (cmd !== "stop");
    $("stRoverCmd").textContent = (cmd === "stop") ? "Idle" : cmd.toUpperCase();
    $("stRoverCmd").className = "val" + (moving ? " go" : "");
    if ($("dpadRing")) $("dpadRing").classList.toggle("on", moving);

    const hud = $("hudVector");
    const badge = $("chassisStatusBadge");
    if (hud) hud.textContent = (cmd === "stop") ? "BRAKE" : cmd.toUpperCase();
    if (badge) {
      badge.textContent = moving ? "PROPEL" : "STANDBY";
      badge.style.color = moving ? "var(--green-light)" : "var(--text3)";
    }

    if (!heartbeat) {
      log((cmd === "stop") ? "Chassis Stopped" : `Chassis ${cmd.toUpperCase()}`, (cmd === "stop") ? "c-stop" : "c-chassis");
    }
  }

  let maneuverTimer = null;
  let maneuverHb = null;

  function holdStart(cmd) {
    if (maneuverTimer) { clearTimeout(maneuverTimer); maneuverTimer = null; }
    if (maneuverHb) { clearInterval(maneuverHb); maneuverHb = null; }
    if (activeCmd === cmd) return;
    if (activeCmd) { clearInterval(hbTimer); activeCmd = null; sendChassis("stop", true); }
    activeCmd = cmd;
    sendChassis(cmd, false);
    clearInterval(hbTimer);
    hbTimer = setInterval(() => { if (activeCmd) sendChassis(activeCmd, true); }, 250);
  }

  function holdEnd() {
    if (!activeCmd) return;
    activeCmd = null;
    clearInterval(hbTimer);
    hbTimer = null;
    sendChassis("stop", false);
  }

  function setupDpadButtons(container) {
    if (!container) return;
    container.querySelectorAll("button").forEach(btn => {
      const cmd = btn.dataset.cmd;
      btn.addEventListener("pointerdown", e => {
        e.preventDefault();
        btn.classList.add("active");
        if (cmd === "stop") {
          if (maneuverTimer) { clearTimeout(maneuverTimer); maneuverTimer = null; }
          if (maneuverHb) { clearInterval(maneuverHb); maneuverHb = null; }
          activeCmd = null;
          clearInterval(hbTimer);
          hbTimer = null;
          sendChassis("stop", false);
        } else {
          holdStart(cmd);
        }
      });
      btn.addEventListener("pointerup", e => {
        e.preventDefault();
        btn.classList.remove("active");
        if (cmd !== "stop") holdEnd();
      });
      btn.addEventListener("pointercancel", e => {
        e.preventDefault();
        btn.classList.remove("active");
        if (cmd !== "stop") holdEnd();
      });
    });
  }

  setupDpadButtons($("dpad"));
  setupDpadButtons($("dualDpad"));

  // Headlight toggle
  window.toggleHeadlight = function() {
    const isCurrentlyOn = $("ledToggle") ? $("ledToggle").classList.contains("on") : false;
    const nextState = !isCurrentlyOn;
    api(`/led/${nextState ? 'on' : 'off'}`).catch(() => {});
    if ($("ledToggle")) $("ledToggle").classList.toggle("on", nextState);
    if ($("ledRow")) $("ledRow").classList.toggle("on", nextState);
    if ($("stHeadlight")) $("stHeadlight").textContent = nextState ? "Active" : "Off";
    log(`Headlight ${nextState ? 'ON' : 'OFF'}`, "c-led");
  };

  // Quick Row Maneuvers (Calibrated & heartbeat-sustained to prevent 600ms watchdog trip)
  window.quickManeuver = function(type) {
    if (activeCmd) holdEnd();
    if (maneuverTimer) { clearTimeout(maneuverTimer); maneuverTimer = null; }
    if (maneuverHb) { clearInterval(maneuverHb); maneuverHb = null; }

    const hud = $("hudVector");
    const badge = $("chassisStatusBadge");

    if (type === 'stop') {
      sendChassis("stop", false);
      if (hud) hud.textContent = "BRAKE";
      if (badge) { badge.textContent = "STANDBY"; badge.style.color = "var(--text3)"; }
      return;
    }

    let cmd = "left";
    let duration = 650;
    let label = "MANEUVER";

    if (type === 'pivot_left') {
      cmd = "left";
      duration = 650; // Calibrated 45° left pivot
      label = "PIVOT L 45°";
    } else if (type === 'pivot_right') {
      cmd = "right";
      duration = 650; // Calibrated 45° right pivot
      label = "PIVOT R 45°";
    } else if (type === 'u_turn') {
      cmd = "left";
      duration = 2100; // Calibrated full 180° row turnaround
      label = "180° U-TURN";
    }

    if (hud) hud.textContent = label;
    if (badge) { badge.textContent = "EXECUTING"; badge.style.color = "var(--green-light)"; }
    log(`Executing ${label} (${(duration/1000).toFixed(1)}s)`, "c-chassis");

    // Send initial command
    sendChassis(cmd, false);

    // Refresh movement watchdog every 200ms so 600ms safety timeout never aborts the turn
    maneuverHb = setInterval(() => {
      sendChassis(cmd, true);
    }, 200);

    maneuverTimer = setTimeout(() => {
      clearInterval(maneuverHb);
      maneuverHb = null;
      maneuverTimer = null;
      sendChassis("stop", false);
      if (hud) hud.textContent = "BRAKE";
      if (badge) { badge.textContent = "STANDBY"; badge.style.color = "var(--text3)"; }
      log(`${label} Complete [BRAKE]`, "c-stop");
    }, duration);
  };

  // Drive Gear Preset
  window.setDriveGear = function(speed) {
    document.querySelectorAll(".gear-btn").forEach(b => b.classList.remove("active"));
    if (speed === 35 && $("gearSlow")) $("gearSlow").classList.add("active");
    else if (speed === 70 && $("gearNormal")) $("gearNormal").classList.add("active");
    else if (speed === 100 && $("gearFast")) $("gearFast").classList.add("active");
    if ($("hudSpeed")) $("hudSpeed").textContent = speed + "%";
    if ($("stSpeed")) $("stSpeed").textContent = speed + "%";
    if ($("valSpeed")) $("valSpeed").textContent = speed + "%";
    if ($("valBoxSpeed")) $("valBoxSpeed").textContent = speed + "%";
    if ($("sliderSpeed")) $("sliderSpeed").value = speed;
    speedStreamer.queue(speed);
    log(`Chassis speed preset ➔ ${speed}%`, "c-chassis");
  };

  // Real-Time Non-blocking Joint and Speed Streamers
  let lastUserInteraction = 0;

  const jointStreamer = {
    inFlight: false,
    pending: {},
    timer: null,
    queue(joint, val) {
      this.pending[joint] = val;
      if (!this.inFlight) {
        this.flush();
      }
    },
    flush() {
      if (Object.keys(this.pending).length === 0) return;
      const copy = { ...this.pending };
      this.pending = {};
      this.inFlight = true;
      const params = new URLSearchParams(copy).toString();
      const t0 = Date.now();
      api("/arm/set?" + params)
        .catch(() => {})
        .finally(() => {
          this.inFlight = false;
          const elapsed = Date.now() - t0;
          const wait = Math.max(0, 30 - elapsed);
          if (Object.keys(this.pending).length > 0) {
            clearTimeout(this.timer);
            this.timer = setTimeout(() => this.flush(), wait);
          }
        });
    }
  };

  const speedStreamer = {
    inFlight: false,
    pendingVal: null,
    timer: null,
    queue(val) {
      this.pendingVal = val;
      if (!this.inFlight) {
        this.flush();
      }
    },
    flush() {
      if (this.pendingVal === null) return;
      const val = this.pendingVal;
      this.pendingVal = null;
      this.inFlight = true;
      api("/arm/speed?val=" + val)
        .catch(() => {})
        .finally(() => {
          this.inFlight = false;
          if (this.pendingVal !== null) {
            clearTimeout(this.timer);
            this.timer = setTimeout(() => this.flush(), 35);
          }
        });
    }
  };

  // 4-DOF Robotic Arm Controls
  window.onJointInput = function(joint, val) {
    lastUserInteraction = Date.now();
    const cap = joint.charAt(0).toUpperCase() + joint.slice(1);
    const labelEl = $("val" + cap);
    if (labelEl) labelEl.textContent = val + "°";

    const boxEl = (joint === 'gripper') ? $("valBoxGrip") : $("valBox" + cap);
    if (boxEl) boxEl.textContent = val + "°";

    // Dynamic SVG Claw Articulation (Live Mechanical Kinematics with centered pivots)
    if (joint === 'gripper') {
      const frac = (180 - val) / (180 - 70); // 0 at 180° (open), 1 at 70° (closed)
      const clawAngle = frac * 22; // rotate inwards up to 22 deg
      const leftClaw = $("svgClawLeft");
      const rightClaw = $("svgClawRight");
      if (leftClaw) leftClaw.setAttribute("transform", `rotate(${clawAngle} 112 68)`);
      if (rightClaw) rightClaw.setAttribute("transform", `rotate(${-clawAngle} 128 68)`);
    }

    // Update live stance telemetry readout
    const stanceEl = $("lblStance");
    if (stanceEl) {
      const b = $("sliderBase") ? $("sliderBase").value : 90;
      const s = $("sliderShoulder") ? $("sliderShoulder").value : 90;
      const e = $("sliderElbow") ? $("sliderElbow").value : 90;
      const g = $("sliderGripper") ? $("sliderGripper").value : 180;
      stanceEl.innerHTML = `B:<strong>${b}°</strong> S:<strong>${s}°</strong> E:<strong>${e}°</strong> G:<strong>${g}°</strong>`;
    }

    // Stream continuous lively updates to ESP32!
    jointStreamer.queue(joint, val);
  };

  window.onJointCommit = function(joint, val) {
    lastUserInteraction = Date.now();
    jointStreamer.queue(joint, val);
    log(`Arm ${joint.toUpperCase()} ➔ ${val}°`, "c-arm");
  };

  window.stepJoint = function(joint, delta) {
    lastUserInteraction = Date.now();
    const cap = joint.charAt(0).toUpperCase() + joint.slice(1);
    const slider = $("slider" + cap);
    if (!slider) return;
    let nextVal = parseInt(slider.value, 10) + delta;
    nextVal = Math.max(parseInt(slider.min, 10), Math.min(parseInt(slider.max, 10), nextVal));
    slider.value = nextVal;
    window.onJointInput(joint, nextVal);
    window.onJointCommit(joint, nextVal);
  };

  window.onSpeedInput = function(val) {
    lastUserInteraction = Date.now();
    $("valSpeed").textContent = val + "%";
    $("stSpeed").textContent = val + "%";
    const boxEl = $("valBoxSpeed");
    if (boxEl) boxEl.textContent = val + "%";
    speedStreamer.queue(val);
  };

  window.onSpeedCommit = function(val) {
    lastUserInteraction = Date.now();
    speedStreamer.queue(val);
    log(`Arm speed ➔ ${val}%`, "c-arm");
  };

  window.armCommand = function(cmd) {
    api(`/arm/${cmd}`).catch(() => {});
    log(`Arm command: ${cmd.toUpperCase()}`, "c-arm");
    if (cmd === 'open') {
      $("sliderGripper").value = 180;
      window.onJointInput('gripper', 180);
      $("stGripper").textContent = "Open";
    } else if (cmd === 'close') {
      $("sliderGripper").value = 70;
      window.onJointInput('gripper', 70);
      $("stGripper").textContent = "Closed";
    }
  };

  window.armPreset = function(presetName) {
    api(`/arm/preset?name=${presetName}`).catch(() => {});
    log(`Preset applied: ${presetName.toUpperCase()}`, "c-arm");
  };

  // Autonomous Waypoint Stance Recording and Sequences

  window.onAppSaveClick = function() {
    api("/arm/save")
      .then(res => res.json())
      .then(data => {
        log(`Position #${data.count} SAVED`, "c-arm");
        const lbl = $("lblPositions");
        if (lbl) lbl.textContent = data.count;
        refreshRecords();
      })
      .catch(() => {});
  };

  window.onAppRunClick = function() {
    const btn = $("btnAppRun");
    if (!btn) return;
    if (btn.textContent.includes("RUN")) {
      api("/arm/run")
        .then(() => {
          btn.textContent = "⏸️ PAUSE";
          btn.classList.add("paused");
          log("Sequence RUNNING", "c-arm");
        })
        .catch(() => {});
    } else {
      api("/arm/pause")
        .then(() => {
          btn.textContent = "▶️ RUN";
          btn.classList.remove("paused");
          log("Sequence PAUSED", "c-stop");
        })
        .catch(() => {});
    }
  };

  window.onAppResetClick = function() {
    api("/arm/reset")
      .then(() => {
        log("Arm RESET to Home [90, 90, 90, 180]", "c-arm");
        $("sliderBase").value = 90; $("valBoxBase").textContent = "90°";
        $("sliderShoulder").value = 90; $("valBoxShoulder").textContent = "90°";
        $("sliderElbow").value = 90; $("valBoxElbow").textContent = "90°";
        $("sliderGripper").value = 180; $("valBoxGrip").textContent = "180°";
        window.onJointInput('gripper', 180);
        const btnRun = $("btnAppRun");
        if (btnRun) {
          btnRun.textContent = "▶️ RUN";
          btnRun.classList.remove("paused");
        }
        const lbl = $("lblPositions");
        if (lbl) lbl.textContent = "0";
      })
      .catch(() => {});
  };

  window.onAppClearClick = function() {
    api("/arm/clear")
      .then(() => {
        log("Waypoints CLEARED", "c-arm");
        const lbl = $("lblPositions");
        if (lbl) lbl.textContent = "0";
        refreshRecords();
      })
      .catch(() => {});
  };

  // Autonomous Sequences
  window.saveWaypoint = function() { window.onAppSaveClick(); };
  window.togglePlaySequence = function() { window.onAppRunClick(); };
  window.stopSequenceCmd = function() {
    api("/arm/stop").catch(() => {});
    log("Sequence stopped", "c-stop");
    const btnRun = $("btnAppRun");
    if (btnRun) { btnRun.textContent = "▶️ RUN"; btnRun.classList.remove("paused"); }
  };
  window.clearWaypoints = function() { window.onAppClearClick(); };

  window.cachedRecords = [];
  function refreshRecords() {
    api("/arm/records")
      .then(res => res.json())
      .then(data => {
        const strip = $("waypointsStrip");
        $("stWaypoints").textContent = `${data.count} / 20`;
        const lbl = $("lblPositions");
        if (lbl) lbl.textContent = data.count;
        if (!strip) return;
        if (!data.count || data.count === 0) {
          strip.innerHTML = `<div class="wp-empty">No recorded waypoints yet. Click <strong>SAVE</strong> to store current stance.</div>`;
          return;
        }
        window.cachedRecords = data.records || [];
        strip.innerHTML = (data.records || []).map(r => {
          const num = (r.id < 10 ? '0' : '') + r.id;
          return `<div class="wp-chip" id="wp_${r.id}" onclick="recallWaypoint(${r.id})">
            <span class="wp-chip-num">#${num}</span>
            <div class="wp-chip-angles">
              <span>B: ${r.base}°</span><span>S: ${r.shoulder}°</span><span>E: ${r.elbow}°</span><span>G: ${r.gripper}°</span>
            </div>
            <span style="color:var(--text3); font-size:8px;">RECALL</span>
          </div>`;
        }).join('');
      })
      .catch(() => {});
  }

  window.recallWaypoint = function(id) {
    if (!window.cachedRecords || !window.cachedRecords.length) return;
    const r = window.cachedRecords.find(x => x.id === id);
    if (!r) return;
    lastUserInteraction = Date.now();
    ["base", "shoulder", "elbow", "gripper"].forEach(j => {
      const val = (j === 'base') ? r.base : (j === 'shoulder') ? r.shoulder : (j === 'elbow') ? r.elbow : r.gripper;
      const cap = j.charAt(0).toUpperCase() + j.slice(1);
      const slider = (j === 'gripper') ? $("sliderGripper") : $("slider" + cap);
      const box = (j === 'gripper') ? $("valBoxGrip") : $("valBox" + cap);
      if (slider) slider.value = val;
      if (box) box.textContent = val + "°";
      if (j === 'gripper') {
        const frac = (180 - val) / (180 - 70);
        const clawAngle = frac * 22;
        const leftClaw = $("svgClawLeft");
        const rightClaw = $("svgClawRight");
        if (leftClaw) leftClaw.setAttribute("transform", `rotate(${clawAngle} 112 68)`);
        if (rightClaw) rightClaw.setAttribute("transform", `rotate(${-clawAngle} 128 68)`);
      }
      jointStreamer.queue(j, val);
    });
    const stanceEl = $("lblStance");
    if (stanceEl) {
      stanceEl.innerHTML = `B:<strong>${r.base}°</strong> S:<strong>${r.shoulder}°</strong> E:<strong>${r.elbow}°</strong> G:<strong>${r.gripper}°</strong>`;
    }
    log(`Recalled Waypoint #${id}`, "c-arm");
  };

  // Unified Status Polling
  function pollStatus() {
    api("/status")
      .then(res => res.json())
      .then(data => {
        // Chassis updates
        const rCmd = data.command || "STOPPED";
        $("stRoverCmd").textContent = (rCmd === "STOPPED") ? "Idle" : rCmd;
        $("stRoverCmd").className = "val" + ((rCmd !== "STOPPED") ? " go" : "");
        $("dpadRing").classList.toggle("on", rCmd !== "STOPPED");

        if (data.led !== undefined) {
          $("ledToggle").classList.toggle("on", data.led);
          $("ledRow").classList.toggle("on", data.led);
        }

        if (data.rssi !== null) $("stRssi").textContent = data.rssi + " dBm";
        else $("stRssi").textContent = "AP";

        // Arm updates
        if (data.arm) {
          const arm = data.arm;
          armState.isMoving = arm.is_moving;
          armState.seqRunning = arm.seq_running;
          armState.savedCount = arm.saved_count;

          $("stArmMotion").textContent = arm.seq_running ? `Seq #${arm.seq_index}` : (arm.is_moving ? "Moving" : "Idle");
          $("stArmMotion").className = "val " + (arm.seq_running ? "arm" : (arm.is_moving ? "warn" : "go"));

          const armMotionTag = $("armMotionTag");
          if (armMotionTag) {
            armMotionTag.textContent = arm.seq_running ? `SEQ #${arm.seq_index}` : (arm.is_moving ? "ARM MOVING" : "ARM IDLE");
            armMotionTag.className = "status-tag" + (arm.seq_running ? " seq" : (arm.is_moving ? " moving" : ""));
          }

          $("armStatusTag").textContent = arm.seq_running ? `SEQUENCE STEP ${arm.seq_index}` : (arm.is_moving ? "MOVING" : "IDLE");
          $("armStatusTag").className = "status-tag" + (arm.seq_running ? " seq" : (arm.is_moving ? " moving" : ""));

          $("stGripper").textContent = (arm.gripper && arm.gripper.is_open) ? "Open" : "Closed";

          const btnRun = $("btnAppRun");
          if (btnRun) {
            if (arm.seq_running && !arm.seq_paused) {
              btnRun.textContent = "⏸️ PAUSE";
              btnRun.classList.add("paused");
            } else {
              btnRun.textContent = "▶️ RUN";
              btnRun.classList.remove("paused");
            }
          }

          $("stWaypoints").textContent = `${arm.saved_count} / 20`;
          const lblPos = $("lblPositions");
          if (lblPos) lblPos.textContent = arm.saved_count;

          // Update stance pill
          const stanceEl = $("lblStance");
          if (stanceEl && arm.base && arm.shoulder && arm.elbow && arm.gripper) {
            stanceEl.innerHTML = `B:<strong>${arm.base.tgt}°</strong> S:<strong>${arm.shoulder.tgt}°</strong> E:<strong>${arm.elbow.tgt}°</strong> G:<strong>${arm.gripper.tgt}°</strong>`;
          }

          // Active sequence step highlight in waypoints strip
          document.querySelectorAll(".wp-chip").forEach(c => c.classList.remove("active-step"));
          if (arm.seq_running && arm.seq_index) {
            const activeWp = $(`wp_${arm.seq_index}`);
            if (activeWp) {
              activeWp.classList.add("active-step");
              activeWp.scrollIntoView({ block: "nearest", behavior: "smooth" });
            }
          }

          // If user has touched or moved any slider in the last 2.5 seconds,
          // DO NOT overwrite the sliders! This eliminates slider sticking and snap-backs!
          const userRecentlyActive = (Date.now() - lastUserInteraction < 2500);
          if (!userRecentlyActive) {
            if (arm.base && arm.base.tgt !== undefined) {
              $("sliderBase").value = arm.base.tgt;
              $("valBoxBase").textContent = arm.base.tgt + "°";
              const l = $("valBase"); if (l) l.textContent = arm.base.tgt + "°";
            }
            if (arm.shoulder && arm.shoulder.tgt !== undefined) {
              $("sliderShoulder").value = arm.shoulder.tgt;
              $("valBoxShoulder").textContent = arm.shoulder.tgt + "°";
              const l = $("valShoulder"); if (l) l.textContent = arm.shoulder.tgt + "°";
            }
            if (arm.elbow && arm.elbow.tgt !== undefined) {
              $("sliderElbow").value = arm.elbow.tgt;
              $("valBoxElbow").textContent = arm.elbow.tgt + "°";
              const l = $("valElbow"); if (l) l.textContent = arm.elbow.tgt + "°";
            }
            if (arm.gripper && arm.gripper.tgt !== undefined) {
              $("sliderGripper").value = arm.gripper.tgt;
              $("valBoxGrip").textContent = arm.gripper.tgt + "°";
              const l = $("valGripper"); if (l) l.textContent = arm.gripper.tgt + "°";
              const frac = (180 - arm.gripper.tgt) / (180 - 70);
              const clawAngle = frac * 22;
              const leftClaw = $("svgClawLeft");
              const rightClaw = $("svgClawRight");
              if (leftClaw) leftClaw.setAttribute("transform", `rotate(${clawAngle} 112 68)`);
              if (rightClaw) rightClaw.setAttribute("transform", `rotate(${-clawAngle} 128 68)`);
            }
            if (arm.speed !== undefined && $("sliderSpeed")) {
              $("sliderSpeed").value = arm.speed;
              $("valBoxSpeed").textContent = arm.speed + "%";
              $("valSpeed").textContent = arm.speed + "%";
              $("stSpeed").textContent = arm.speed + "%";
            }
          }
        }
      })
      .catch(() => {});
  }

  // Touch/pointer protection on all sliders to immediately lock out polling overwrites
  ["sliderBase", "sliderShoulder", "sliderElbow", "sliderGripper", "sliderSpeed"].forEach(id => {
    const el = $(id);
    if (el) {
      el.addEventListener("pointerdown", () => { lastUserInteraction = Date.now(); });
      el.addEventListener("touchstart", () => { lastUserInteraction = Date.now(); }, { passive: true });
    }
  });

  setInterval(pollStatus, 1000);
  pollStatus();
  refreshRecords();

  // Desktop keyboard bindings
  window.addEventListener("keydown", e => {
    const k = e.key.toLowerCase();
    if (k === " ") { e.preventDefault(); sendChassis("stop", false); return; }
    if (k === "w" || k === "arrowup") { e.preventDefault(); holdStart("forward"); }
    else if (k === "s" || k === "arrowdown") { e.preventDefault(); holdStart("backward"); }
    else if (k === "a" || k === "arrowleft") { e.preventDefault(); holdStart("left"); }
    else if (k === "d" || k === "arrowright") { e.preventDefault(); holdStart("right"); }
    else if (k === "o") { window.armCommand("open"); }
    else if (k === "c") { window.armCommand("close"); }
    else if (k === "h") { window.armPreset("home"); }
  });

  window.addEventListener("keyup", e => {
    const k = e.key.toLowerCase();
    if (["w","s","a","d","arrowup","arrowdown","arrowleft","arrowright"].includes(k)) {
      holdEnd();
    }
  });

})();
</script>
</body>
</html>
)HTML_PAGE";

// 19. Network initialization

/**
 * The fallback Wi-Fi access point preserves local rover control when the configured
 * station network cannot be reached. This keeps the controller accessible without
 * requiring another router.
 */
void startAccessPoint() {

  // AP mode is sufficient because the ESP32 will host the controller locally.
  WiFi.mode(WIFI_AP);

  // Fixed local address keeps the fallback controller address predictable.
  IPAddress localIP(192, 168, 9, 1);

  // The gateway points back to the ESP32 because it owns the AP network.
  IPAddress gateway(192, 168, 9, 1);

  // Standard /24 subnet for the local rover network.
  IPAddress subnet(255, 255, 255, 0);

  // Apply the fixed AP network configuration before starting the AP.
  const bool configResult =
    WiFi.softAPConfig(
      localIP,
      gateway,
      subnet
    );

  // Start the fallback AP with the configured SSID, password, channel, and
  // client limit.
  const bool apStarted =
    WiFi.softAP(
      Config::AP_SSID,
      Config::AP_PASSWORD,
      Config::WIFI_CHANNEL,
      false,
      Config::MAX_CLIENTS
    );

  // Read the final AP address reported by the Wi-Fi stack.
  const IPAddress ip =
    WiFi.softAPIP();

  // Start local DNS server to resolve 'agrirover.local' and captive portal queries to 192.168.9.1
  dnsServer.start(53, "*", localIP);

  Serial.println();
  Serial.println(F("=========================================="));
  Serial.println(F(" AGRIROVER ACCESS POINT MODE"));
  Serial.println(F("=========================================="));
  Serial.print(F(" SSID:       "));
  Serial.println(Config::AP_SSID);
  Serial.print(F(" Password:   "));
  Serial.println(Config::AP_PASSWORD);
  Serial.print(F(" IP:         "));
  Serial.println(ip);
  Serial.print(F(" Direct URL: http://"));
  Serial.println(ip);
  Serial.println(F(" mDNS URL:   http://agrirover.local"));
  Serial.print(F(" AP config:  "));
  Serial.println(configResult ? F("SUCCESS") : F("FAILED"));
  Serial.print(F(" AP start:   "));
  Serial.println(apStarted ? F("SUCCESS") : F("FAILED"));
  Serial.println(F("=========================================="));
}

/**
 * The firmware first attempts the configured station network. If the connection does not
 * complete within the configured attempt count, the firmware starts AP mode.
 */
void startWiFi() {

  Serial.print(F("Connecting to WiFi: "));
  Serial.println(Config::STA_SSID);

  // Disconnect any lingering connection and initialize fresh station radio
  WiFi.disconnect(true);
  delay(100);

  // Start in station mode so the ESP32 first attempts to join the configured LAN.
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);

  // Lower RF output power to 15dBm to throttle peak current surge and eliminate brownouts
  WiFi.setTxPower(WIFI_POWER_15dBm);

  // Set the device hostname before starting the station connection.
  WiFi.setHostname(Config::MDNS_HOSTNAME);

  // Perform quick scan to detect target 2.4 GHz network
  Serial.println(F("[WIFI] Checking 2.4 GHz wireless environment..."));
  int n = WiFi.scanNetworks(false, false, false, 250);
  bool ssidFound = false;
  if (n > 0) {
    for (int i = 0; i < n; ++i) {
      if (WiFi.SSID(i) == Config::STA_SSID) {
        ssidFound = true;
        break;
      }
    }
    if (!ssidFound) {
      Serial.println(F("────────────────────────────────────────────────────────────────"));
      Serial.printf("  [WIFI] Notice: SSID '%s' not detected in 2.4 GHz band scan.\n", Config::STA_SSID);
      Serial.println(F("  If using iPhone Hotspot: Enable Settings -> Personal Hotspot -> Maximize Compatibility"));
      Serial.println(F("────────────────────────────────────────────────────────────────"));
    } else {
      Serial.printf("  [WIFI] Target SSID '%s' found in 2.4 GHz band.\n", Config::STA_SSID);
    }
  }

  // Begin the station connection.
  WiFi.begin(
    Config::STA_SSID,
    Config::STA_PASSWORD
  );

  // Track bounded connection attempts so startup cannot wait forever.
  uint8_t attempts = 0;

  while (
    WiFi.status() != WL_CONNECTED &&
    attempts < Config::WIFI_CONNECT_ATTEMPTS
  ) {

    delay(Config::WIFI_RETRY_DELAY_MS);

    Serial.print(".");
    attempts++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println();
    Serial.println(F("=========================================="));
    Serial.println(F(" AGRIROVER WIFI CONNECTED"));
    Serial.println(F("=========================================="));
    Serial.print(F(" SSID:       "));
    Serial.println(Config::STA_SSID);
    Serial.print(F(" IP:         "));
    Serial.println(WiFi.localIP());
    Serial.print(F(" RSSI:       "));
    Serial.print(WiFi.RSSI());
    Serial.println(F(" dBm"));
    Serial.print(F(" Direct URL: http://"));
    Serial.println(WiFi.localIP());
    Serial.println(F(" mDNS URL:   http://agrirover.local"));
    Serial.println(F(" (Note: On iOS Personal Hotspot, use Direct URL if mDNS is isolated)"));
    Serial.println(F("=========================================="));

  } else {

    // The station network was not available within the startup budget.
    Serial.println();
    Serial.println(F("WiFi connection failed."));
    Serial.println(F("Starting AgriRover Access Point..."));

    startAccessPoint();
  }
}

// 20. Hardware initialization

/**
 * The firmware configures every rover GPIO pin, initializes the PCA9685 PWM driver,
 * and places both the chassis and robotic arm in a known safe state.
 */
void initializeHardware() {

  // Configure the left H-bridge inputs as outputs.
  pinMode(Pins::LEFT_IN1, OUTPUT);
  pinMode(Pins::LEFT_IN2, OUTPUT);

  // Configure the right H-bridge inputs as outputs.
  pinMode(Pins::RIGHT_IN1, OUTPUT);
  pinMode(Pins::RIGHT_IN2, OUTPUT);

  // Configure the headlight pin as an output.
  pinMode(Pins::STATUS_LED, OUTPUT);

  // Stop the drive system before any network service becomes available.
  stopRover();

  // Start with the headlight off.
  setLed(false);

  // 1. Configure ESP32 I2C pins BEFORE any Wire or PCA9685 library calls.
  // Wire.setPins() is mandatory on ESP32 so internal library calls to _wire->begin()
  // do not reset SDA to default GPIO 21 (which is shared with motor RIGHT_IN2).
  Wire.setPins(Pins::I2C_SDA, Pins::I2C_SCL);
  Wire.begin(Pins::I2C_SDA, Pins::I2C_SCL);
  Wire.setClock(100000);

  Serial.println();
  Serial.println(F("[I2C] Scanning I2C bus on SDA=GPIO23, SCL=GPIO22..."));
  uint8_t devicesFound = 0;
  bool pcaFound = false;

  for (uint8_t addr = 1; addr < 127; ++addr) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      devicesFound++;
      Serial.print(F("  -> Detected I2C device at address 0x"));
      if (addr < 16) Serial.print("0");
      Serial.print(addr, HEX);
      if (addr == ArmConfig::PCA9685_ADDR) {
        pcaFound = true;
        Serial.print(F(" (PCA9685 16-channel PWM driver)"));
      } else if (addr == 0x70) {
        Serial.print(F(" (PCA9685 All-Call address)"));
      }
      Serial.println();
    }
  }

  if (!pcaFound) {
    Serial.println(F("  ⚠️  PCA9685 not detected at 0x40 on SDA=GPIO23, SCL=GPIO22!"));
    Serial.println(F("  1. Verify PCA9685 VCC is connected to ESP32 3.3V or VIN."));
    Serial.println(F("  2. Verify PCA9685 GND is connected to ESP32 GND."));
    Serial.println(F("  3. Verify SDA -> GPIO23, SCL -> GPIO22."));
  }

  // 2. Initialize PCA9685 PWM driver
  Serial.println(F("[ARM] Initializing PCA9685 PWM driver at 0x40..."));
  pwm.begin();

  // 3. Re-affirm Wire pins in case Adafruit library re-assigned pins to defaults
  Wire.setPins(Pins::I2C_SDA, Pins::I2C_SCL);
  Wire.begin(Pins::I2C_SDA, Pins::I2C_SCL);
  Wire.setClock(100000);

  // 4. Restore L293D motor pin (GPIO 21) as digital output
  pinMode(Pins::RIGHT_IN2, OUTPUT);
  digitalWrite(Pins::RIGHT_IN2, LOW);

  // 5. Configure servo PWM frequency (50 Hz)
  pwm.setPWMFreq(ArmConfig::SERVO_FREQUENCY);
  delay(50);

  g_armInitialized = true;
  armState.detached = false;

  Serial.println(F("[ARM] PCA9685 initialized at 0x40 via I2C (SDA=GPIO23, SCL=GPIO22)."));
  Serial.println(F("[ARM] OE pin: NOT connected to ESP32 (onboard pull-down keeps outputs active)."));
  Serial.println(F("[ARM] Reminder: Servos require 5V-6V connected to PCA9685 screw terminal (V+ & GND)!"));
}

// 21. mDNS initialization

/**
 * The firmware starts mDNS and advertises the HTTP service through agrirover.local.
 */
void startMDNS() {

  // ESP32 Arduino Core 3.3.x starts mDNS through MDNS.begin(). No MDNS.update()
  // call is required in loop() for this implementation.
  if (MDNS.begin(Config::MDNS_HOSTNAME)) {

    // Advertise the local web server to mDNS-aware clients.
    MDNS.addService(
      "http",
      "tcp",
      Config::HTTP_PORT
    );

    Serial.println(
      F("mDNS started: http://agrirover.local")
    );

  } else {

    Serial.println(
      F("mDNS failed to start.")
    );
  }
}

// 22. Firmware setup

/**
 * The firmware initializes serial logging, hardware, networking, mDNS, HTTP routes,
 * and the movement watchdog in a deterministic order.
 */
void setup() {

  // Suppress brownout detector during startup to prevent resets from Wi-Fi RF power spikes
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  // Start the serial console for startup and operational status messages.
  Serial.begin(115200);

  // Give the USB serial interface a short time to become available.
  delay(200);

  Serial.println();
  Serial.println(F("=========================================="));
  Serial.println(F("        AGRIROVER ESP32 STARTUP"));
  Serial.println(F("=========================================="));
  Serial.println(F(" Team:        Innovex"));
  Serial.println(F(" SIH Challenge: SH26180"));
  Serial.println(F(" Organization: Qualcomm Inc"));
  Serial.println(F(" Category:    Hardware"));
  Serial.println(F(" Theme:       Disaster Management"));
  Serial.println(F("=========================================="));

  // Configure every physical output and put the rover and arm into a known safe state.
  initializeHardware();

  // Bring up station mode or the fallback AP.
  startWiFi();

  // Start local name resolution after the final network interface is ready.
  startMDNS();

  // Register the web controller and all device endpoints.
  registerRoutes();

  // Start accepting HTTP requests.
  server.begin();

  Serial.println(F("HTTP server started."));

  // Arm the movement watchdog from a stopped state.
  touchWatchdog();

  Serial.println(F("Motor safety watchdog armed."));

  // Now that Wi-Fi and HTTP are safely established without inrush competition,
  // engage the 4-DOF robotic arm to HOME stance if PCA9685 is ready.
  if (g_armInitialized) {
    Serial.println(F("[ARM] Engaging 4-DOF Robotic Arm to HOME stance [90, 90, 90, 180]..."));
    resetRobot();
    writeAllServos();
    Serial.println(F("[ARM] 4-DOF Robotic Arm online and holding."));
  }
  Serial.println();

  Serial.println(F("=========================================="));
  Serial.println(F(" AGRIROVER READY"));
  Serial.println(F("=========================================="));
  Serial.println(F(" Team: Innovex"));
  Serial.println(F(" Controller: Dual Teleoperation Web UI"));
  Serial.println(F(" Steering polarity: configured in MotorConfig"));
  Serial.println(F(" Robotic Arm: 4-DOF PCA9685 I2C (Address 0x40)"));
  Serial.println(F("=========================================="));
}

// 23. Main control loop

/**
 * The main loop services HTTP requests, the local movement watchdog, and the
 * robotic arm trajectory engine. The loop remains deliberately small, while motor
 * safety stays independent of the browser because serviceWatchdog() executes locally.
 */
void loop() {

  // Process DNS requests in AP mode so agrirover.local resolves reliably on any connected device
  dnsServer.processNextRequest();

  // Process pending HTTP control and status requests.
  server.handleClient();

  // Enforce the local movement timeout independently of network state.
  serviceWatchdog();

  // Advance 4-DOF smooth servo motion engine ticks.
  updateServos();

  // Update autonomous waypoint trajectory playback.
  updateSequence();

  // Process USB serial and Bluetooth CLI commands.
  readSerial();

  // Do not add MDNS.update() here. ESP32 Arduino Core 3.3.x does not expose
  // that API in the form used by older examples.
}


