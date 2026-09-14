// ============================================================
// 4-DOF ROBOTIC ARM
// PRODUCTION MASTER CONTROLLER
// ARDUINO + PCA9685 + HC-05 / HC-06
// ============================================================

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <SoftwareSerial.h>

// ============================================================
// BLUETOOTH
// ============================================================

#define BT_RX 2
#define BT_TX 3

SoftwareSerial BT(BT_RX, BT_TX);

// ============================================================
// PCA9685
// ============================================================

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// ============================================================
// SERVO CHANNELS
// ============================================================

#define BASE_CH       0
#define SHOULDER_CH   1
#define ELBOW_CH      2
#define GRIPPER_CH    3

// ============================================================
// SERVO PWM CONFIGURATION
// ============================================================

#define SERVO_MIN       120
#define SERVO_MAX       520
#define SERVO_FREQUENCY 50

// ============================================================
// SERVO DIRECTIONS
// ============================================================

#define BASE_REVERSED       false
#define SHOULDER_REVERSED   false
#define ELBOW_REVERSED      false
#define GRIPPER_REVERSED    false

// ============================================================
// SERVO ANGLE LIMITS
// ============================================================

#define BASE_MIN       5
#define BASE_MAX       175

#define SHOULDER_MIN   0
#define SHOULDER_MAX   180

#define ELBOW_MIN      0
#define ELBOW_MAX      180

#define GRIPPER_CLOSE  70
#define GRIPPER_OPEN   180

// ============================================================
// HOME POSITION
// ============================================================

#define BASE_HOME       90
#define SHOULDER_HOME   90
#define ELBOW_HOME      90
#define GRIPPER_HOME    GRIPPER_OPEN

// ============================================================
// MOTION ENGINE
// ============================================================

#define MOTION_INTERVAL 8

unsigned long lastMotionUpdate = 0;

// ============================================================
// SERVO SPEEDS
// ============================================================

#define BASE_MAX_SPEED       360.0
#define SHOULDER_MAX_SPEED   300.0
#define ELBOW_MAX_SPEED      340.0
#define GRIPPER_MAX_SPEED    500.0

int appSpeed = 70;

// ============================================================
// CURRENT POSITIONS
// ============================================================

float baseCurrent       = BASE_HOME;
float shoulderCurrent   = SHOULDER_HOME;
float elbowCurrent      = ELBOW_HOME;
float gripperCurrent    = GRIPPER_HOME;

// ============================================================
// TARGET POSITIONS
// ============================================================

float baseTarget        = BASE_HOME;
float shoulderTarget    = SHOULDER_HOME;
float elbowTarget       = ELBOW_HOME;
float gripperTarget     = GRIPPER_HOME;

// ============================================================
// BLUETOOTH PARSER
// ============================================================

String commandBuffer = "";

unsigned long lastBluetoothByte = 0;

#define COMMAND_TIMEOUT 100

// ============================================================
// SAVED POSITIONS
// ============================================================

#define MAX_POSITIONS 20

struct RobotPosition
{
  int base;
  int shoulder;
  int elbow;
  int gripper;
};

RobotPosition savedPositions[MAX_POSITIONS];

int savedCount = 0;

// ============================================================
// SEQUENCE ENGINE
// ============================================================

bool runningSequence = false;
bool pauseSequence = false;

int sequenceIndex = 0;

unsigned long sequenceStartTime = 0;

#define SEQUENCE_SETTLE_TIME 250

// ============================================================
// POSITION ARRIVAL TOLERANCE
// ============================================================

#define POSITION_TOLERANCE 0.7

// ============================================================
// ABSOLUTE VALUE
// ============================================================

float absFloat(float value)
{
  return value < 0 ? -value : value;
}

// ============================================================
// CLAMP ANGLE
// ============================================================

float clampAngle(
  float value,
  float minimum,
  float maximum
)
{
  if (value < minimum)
    return minimum;

  if (value > maximum)
    return maximum;

  return value;
}

// ============================================================
// ANGLE TO PWM
// ============================================================

int angleToPulse(float angle)
{
  angle =
    clampAngle(
      angle,
      0,
      180
    );

  float pulse =
    SERVO_MIN +
    (
      (SERVO_MAX - SERVO_MIN) *
      (angle / 180.0)
    );

  return (int)pulse;
}

// ============================================================
// WRITE PHYSICAL SERVO
// ============================================================

void writeServo(
  int channel,
  float logicalAngle,
  bool reversed
)
{
  logicalAngle =
    clampAngle(
      logicalAngle,
      0,
      180
    );

  float physicalAngle =
    logicalAngle;

  if (reversed)
    physicalAngle =
      180.0 - logicalAngle;

  int pulse =
    angleToPulse(
      physicalAngle
    );

  pwm.setPWM(
    channel,
    0,
    pulse
  );
}

// ============================================================
// WRITE ALL SERVOS
// ============================================================

void writeAllServos()
{
  writeServo(
    BASE_CH,
    baseCurrent,
    BASE_REVERSED
  );

  writeServo(
    SHOULDER_CH,
    shoulderCurrent,
    SHOULDER_REVERSED
  );

  writeServo(
    ELBOW_CH,
    elbowCurrent,
    ELBOW_REVERSED
  );

  writeServo(
    GRIPPER_CH,
    gripperCurrent,
    GRIPPER_REVERSED
  );
}

// ============================================================
// GET MOVEMENT SPEED
// ============================================================

float getSpeed(
  float maximumSpeed
)
{
  if (appSpeed <= 0)
    return 0;

  float ratio =
    (float)appSpeed / 100.0;

  float speed =
    maximumSpeed * ratio;

  if (speed < 5.0)
    speed = 5.0;

  return speed;
}

// ============================================================
// MOVE SERVO TOWARD TARGET
// ============================================================

void moveServo(
  int channel,
  float &current,
  float target,
  float maximumSpeed,
  bool reversed
)
{
  float difference =
    target - current;

  if (
    absFloat(difference) <=
    POSITION_TOLERANCE
  )
  {
    current = target;

    writeServo(
      channel,
      current,
      reversed
    );

    return;
  }

  float dt =
    (float)MOTION_INTERVAL /
    1000.0;

  float speed =
    getSpeed(
      maximumSpeed
    );

  if (speed <= 0)
  {
    writeServo(
      channel,
      current,
      reversed
    );

    return;
  }

  float movement =
    speed * dt;

  if (
    movement >
    absFloat(difference)
  )
  {
    movement =
      absFloat(difference);
  }

  if (difference > 0)
    current += movement;
  else
    current -= movement;

  if (
    absFloat(
      target - current
    ) <= POSITION_TOLERANCE
  )
  {
    current = target;
  }

  writeServo(
    channel,
    current,
    reversed
  );
}

// ============================================================
// UPDATE ALL SERVO MOTION
// ============================================================

void updateServos()
{
  unsigned long now =
    millis();

  if (
    now - lastMotionUpdate <
    MOTION_INTERVAL
  )
  {
    return;
  }

  lastMotionUpdate =
    now;

  moveServo(
    BASE_CH,
    baseCurrent,
    baseTarget,
    BASE_MAX_SPEED,
    BASE_REVERSED
  );

  moveServo(
    SHOULDER_CH,
    shoulderCurrent,
    shoulderTarget,
    SHOULDER_MAX_SPEED,
    SHOULDER_REVERSED
  );

  moveServo(
    ELBOW_CH,
    elbowCurrent,
    elbowTarget,
    ELBOW_MAX_SPEED,
    ELBOW_REVERSED
  );

  moveServo(
    GRIPPER_CH,
    gripperCurrent,
    gripperTarget,
    GRIPPER_MAX_SPEED,
    GRIPPER_REVERSED
  );
}

// ============================================================
// CHECK WHETHER ROBOT REACHED TARGET
// ============================================================

bool robotAtTarget()
{
  return
    absFloat(baseCurrent - baseTarget) <= POSITION_TOLERANCE &&
    absFloat(shoulderCurrent - shoulderTarget) <= POSITION_TOLERANCE &&
    absFloat(elbowCurrent - elbowTarget) <= POSITION_TOLERANCE &&
    absFloat(gripperCurrent - gripperTarget) <= POSITION_TOLERANCE;
}

// ============================================================
// SET SERVO TARGET
// ============================================================

void setServoTarget(
  int servo,
  float value
)
{
  switch (servo)
  {
    case 1:

      baseTarget =
        clampAngle(
          value,
          BASE_MIN,
          BASE_MAX
        );

      break;

    case 2:

      shoulderTarget =
        clampAngle(
          value,
          SHOULDER_MIN,
          SHOULDER_MAX
        );

      break;

    case 3:

      elbowTarget =
        clampAngle(
          value,
          ELBOW_MIN,
          ELBOW_MAX
        );

      break;

    case 6:

      gripperTarget =
        clampAngle(
          value,
          GRIPPER_CLOSE,
          GRIPPER_OPEN
        );

      break;
  }
}

// ============================================================
// STOP SEQUENCE
// ============================================================

void stopSequence()
{
  runningSequence = false;
  pauseSequence = false;
  sequenceIndex = 0;
}

// ============================================================
// APPLY SAVED POSITION
// ============================================================

void applySavedPosition(
  int index
)
{
  if (
    index < 0 ||
    index >= savedCount
  )
  {
    return;
  }

  baseTarget =
    savedPositions[index].base;

  shoulderTarget =
    savedPositions[index].shoulder;

  elbowTarget =
    savedPositions[index].elbow;

  gripperTarget =
    savedPositions[index].gripper;
}

// ============================================================
// START SEQUENCE
// ============================================================

void startSequence()
{
  if (savedCount <= 0)
  {
    Serial.println(
      "RUN_ERROR:NO_SAVED_POSITIONS"
    );

    return;
  }

  runningSequence = true;
  pauseSequence = false;
  sequenceIndex = 0;

  applySavedPosition(
    sequenceIndex
  );

  sequenceStartTime =
    millis();

  Serial.print(
    "RUN_START:RECORD="
  );

  Serial.println(
    sequenceIndex + 1
  );
}

// ============================================================
// UPDATE SEQUENCE
// ============================================================

void updateSequence()
{
  if (!runningSequence)
    return;

  if (pauseSequence)
    return;

  if (savedCount <= 0)
  {
    stopSequence();

    Serial.println(
      "RUN_STOP:NO_RECORDS"
    );

    return;
  }

  if (
    robotAtTarget() &&
    millis() - sequenceStartTime >=
    SEQUENCE_SETTLE_TIME
  )
  {
    sequenceIndex++;

    if (
      sequenceIndex >=
      savedCount
    )
    {
      runningSequence = false;
      sequenceIndex = 0;

      Serial.println(
        "RUN_COMPLETE"
      );

      return;
    }

    applySavedPosition(
      sequenceIndex
    );

    sequenceStartTime =
      millis();

    Serial.print(
      "RUN_RECORD:"
    );

    Serial.println(
      sequenceIndex + 1
    );
  }
}

// ============================================================
// SAVE CURRENT POSITION
// ============================================================

void savePosition()
{
  if (
    savedCount >=
    MAX_POSITIONS
  )
  {
    Serial.println(
      "SAVE_ERROR:MEMORY_FULL"
    );

    return;
  }

  savedPositions[savedCount].base =
    (int)round(baseTarget);

  savedPositions[savedCount].shoulder =
    (int)round(shoulderTarget);

  savedPositions[savedCount].elbow =
    (int)round(elbowTarget);

  savedPositions[savedCount].gripper =
    (int)round(gripperTarget);

  int record =
    savedCount + 1;

  savedCount++;

  Serial.print(
    "SAVE_OK:RECORD="
  );

  Serial.println(
    record
  );

  Serial.print(
    "BASE="
  );

  Serial.println(
    savedPositions[record - 1].base
  );

  Serial.print(
    "SHOULDER="
  );

  Serial.println(
    savedPositions[record - 1].shoulder
  );

  Serial.print(
    "ELBOW="
  );

  Serial.println(
    savedPositions[record - 1].elbow
  );

  Serial.print(
    "GRIPPER="
  );

  Serial.println(
    savedPositions[record - 1].gripper
  );

  Serial.print(
    "TOTAL_RECORDS="
  );

  Serial.println(
    savedCount
  );
}

// ============================================================
// CLEAR SAVED POSITIONS
// ============================================================

void clearPositions()
{
  stopSequence();

  savedCount = 0;

  Serial.println(
    "CLEAR_OK"
  );
}

// ============================================================
// HOME / RESET
// ============================================================

void resetRobot()
{
  stopSequence();

  baseTarget =
    BASE_HOME;

  shoulderTarget =
    SHOULDER_HOME;

  elbowTarget =
    ELBOW_HOME;

  gripperTarget =
    GRIPPER_HOME;

  Serial.println(
    "RESET_OK"
  );

  Serial.println(
    "RESET_TARGET:BASE=90,SHOULDER=90,ELBOW=90,GRIPPER=180"
  );
}

// ============================================================
// OPEN GRIPPER
// ============================================================

void openGripper()
{
  stopSequence();

  gripperTarget =
    GRIPPER_OPEN;

  Serial.println(
    "GRIPPER_OPEN"
  );
}

// ============================================================
// CLOSE GRIPPER
// ============================================================

void closeGripper()
{
  stopSequence();

  gripperTarget =
    GRIPPER_CLOSE;

  Serial.println(
    "GRIPPER_CLOSE"
  );
}

// ============================================================
// PRINT POSITION
// ============================================================

void printPosition()
{
  Serial.print(
    "POSITION:"
  );

  Serial.print(
    (int)round(baseTarget)
  );

  Serial.print(
    ","
  );

  Serial.print(
    (int)round(shoulderTarget)
  );

  Serial.print(
    ","
  );

  Serial.print(
    (int)round(elbowTarget)
  );

  Serial.print(
    ","
  );

  Serial.println(
    (int)round(gripperTarget)
  );
}

// ============================================================
// PRINT SAVED RECORDS
// ============================================================

void printRecords()
{
  Serial.print(
    "RECORD_COUNT="
  );

  Serial.println(
    savedCount
  );

  for (
    int i = 0;
    i < savedCount;
    i++
  )
  {
    Serial.print(
      "RECORD="
    );

    Serial.print(
      i + 1
    );

    Serial.print(
      ":"
    );

    Serial.print(
      savedPositions[i].base
    );

    Serial.print(
      ","
    );

    Serial.print(
      savedPositions[i].shoulder
    );

    Serial.print(
      ","
    );

    Serial.print(
      savedPositions[i].elbow
    );

    Serial.print(
      ","
    );

    Serial.println(
      savedPositions[i].gripper
    );
  }
}

// ============================================================
// PROCESS COMMAND
// ============================================================

void processCommand(
  String cmd
)
{
  cmd.trim();

  if (
    cmd.length() == 0
  )
  {
    return;
  }

  String upper =
    cmd;

  upper.toUpperCase();

  // ==========================================================
  // SAVE
  // ==========================================================

  if (
    upper == "SAVE"
  )
  {
    savePosition();
    return;
  }

  // ==========================================================
  // RUN
  // ==========================================================

  if (
    upper == "RUN"
  )
  {
    startSequence();
    return;
  }

  // ==========================================================
  // PAUSE
  // ==========================================================

  if (
    upper == "PAUSE"
  )
  {
    if (runningSequence)
    {
      pauseSequence = true;

      Serial.println(
        "PAUSE_OK"
      );
    }

    return;
  }

  // ==========================================================
  // RESUME
  // ==========================================================

  if (
    upper == "RESUME"
  )
  {
    if (
      runningSequence &&
      pauseSequence
    )
    {
      pauseSequence = false;

      sequenceStartTime =
        millis();

      Serial.println(
        "RESUME_OK"
      );
    }

    return;
  }

  // ==========================================================
  // RESET
  // ==========================================================

  if (
    upper == "RESET"
  )
  {
    resetRobot();
    return;
  }

  // ==========================================================
  // CLEAR
  // ==========================================================

  if (
    upper == "CLEAR" ||
    upper == "CLEARALL"
  )
  {
    clearPositions();
    return;
  }

  // ==========================================================
  // RECORDS
  // ==========================================================

  if (
    upper == "RECORDS" ||
    upper == "LIST"
  )
  {
    printRecords();
    return;
  }

  // ==========================================================
  // POSITION
  // ==========================================================

  if (
    upper == "POSITION" ||
    upper == "POS"
  )
  {
    printPosition();
    return;
  }

  // ==========================================================
  // OPEN
  // ==========================================================

  if (
    upper == "OPEN" ||
    upper == "GRIPOPEN"
  )
  {
    openGripper();
    return;
  }

  // ==========================================================
  // CLOSE
  // ==========================================================

  if (
    upper == "CLOSE" ||
    upper == "GRIPCLOSE"
  )
  {
    closeGripper();
    return;
  }

  // ==========================================================
  // SPEED
  // ==========================================================

  if (
    cmd.length() >= 3 &&
    cmd.charAt(0) == 's' &&
    cmd.charAt(1) == 's'
  )
  {
    int value =
      cmd.substring(2).toInt();

    value =
      constrain(
        value,
        0,
        100
      );

    appSpeed =
      value;

    Serial.print(
      "SPEED_OK:"
    );

    Serial.println(
      appSpeed
    );

    return;
  }

  // ==========================================================
  // SERVO COMMAND
  // ==========================================================

  if (
    cmd.length() >= 3 &&
    cmd.charAt(0) == 's'
  )
  {
    int servo =
      cmd.charAt(1) - '0';

    if (
      servo == 1 ||
      servo == 2 ||
      servo == 3 ||
      servo == 6
    )
    {
      float value =
        cmd.substring(2).toFloat();

      stopSequence();

      setServoTarget(
        servo,
        value
      );

      return;
    }
  }

  Serial.print(
    "UNKNOWN_COMMAND:"
  );

  Serial.println(
    cmd
  );
}

// ============================================================
// BLUETOOTH STREAM PARSER
// ============================================================

void readBluetooth()
{
  while (
    BT.available()
  )
  {
    char c =
      BT.read();

    lastBluetoothByte =
      millis();

    // ========================================================
    // LINE END
    // ========================================================

    if (
      c == '\r' ||
      c == '\n'
    )
    {
      if (
        commandBuffer.length() >
        0
      )
      {
        processCommand(
          commandBuffer
        );

        commandBuffer =
          "";
      }

      continue;
    }

    // ========================================================
    // NEW LOWERCASE SERVO COMMAND
    // ========================================================

    if (
      c == 's'
    )
    {
      if (
        commandBuffer.length() == 1 &&
        commandBuffer.charAt(0) == 's'
      )
      {
        commandBuffer += c;
      }
      else
      {
        if (
          commandBuffer.length() >
          0
        )
        {
          processCommand(
            commandBuffer
          );
        }

        commandBuffer =
          "s";
      }

      continue;
    }

    // ========================================================
    // TEXT COMMAND
    // ========================================================

    if (
      c == 'S' ||
      c == 'R' ||
      c == 'P' ||
      c == 'O' ||
      c == 'C' ||
      c == 'L'
    )
    {
      if (
        commandBuffer.length() >
        0
      )
      {
        processCommand(
          commandBuffer
        );

        commandBuffer =
          "";
      }

      commandBuffer += c;

      continue;
    }

    // ========================================================
    // NORMAL CHARACTER
    // ========================================================

    commandBuffer += c;

    // ========================================================
    // BUFFER PROTECTION
    // ========================================================

    if (
      commandBuffer.length() >
      48
    )
    {
      commandBuffer =
        "";
    }
  }

  // ==========================================================
  // TIMEOUT
  // ==========================================================

  if (
    commandBuffer.length() > 0 &&
    millis() -
    lastBluetoothByte >
    COMMAND_TIMEOUT
  )
  {
    processCommand(
      commandBuffer
    );

    commandBuffer =
      "";
  }
}

// ============================================================
// USB SERIAL
// ============================================================

void readSerial()
{
  static String serialBuffer =
    "";

  while (
    Serial.available()
  )
  {
    char c =
      Serial.read();

    if (
      c == '\r' ||
      c == '\n'
    )
    {
      if (
        serialBuffer.length() >
        0
      )
      {
        processCommand(
          serialBuffer
        );

        serialBuffer =
          "";
      }
    }
    else
    {
      serialBuffer += c;

      if (
        serialBuffer.length() >
        48
      )
      {
        serialBuffer =
          "";
      }
    }
  }
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(
    9600
  );

  BT.begin(
    9600
  );

  Wire.begin();

  pwm.begin();

  pwm.setPWMFreq(
    SERVO_FREQUENCY
  );

  delay(300);

  // ==========================================================
  // INITIAL POSITION
  // ==========================================================

  baseCurrent =
    BASE_HOME;

  shoulderCurrent =
    SHOULDER_HOME;

  elbowCurrent =
    ELBOW_HOME;

  gripperCurrent =
    GRIPPER_HOME;

  baseTarget =
    BASE_HOME;

  shoulderTarget =
    SHOULDER_HOME;

  elbowTarget =
    ELBOW_HOME;

  gripperTarget =
    GRIPPER_HOME;

  writeAllServos();

  delay(500);

  // ==========================================================
  // STARTUP MESSAGE
  // ==========================================================

  Serial.println();
  Serial.println(
    "=============================================="
  );

  Serial.println(
    "       4-DOF ROBOTIC ARM"
  );

  Serial.println(
    "       PRODUCTION MASTER CONTROLLER"
  );

  Serial.println(
    "=============================================="
  );

  Serial.println();

  Serial.println(
    "BASE     : CH0"
  );

  Serial.println(
    "SHOULDER : CH1"
  );

  Serial.println(
    "ELBOW    : CH2"
  );

  Serial.println(
    "GRIPPER  : CH3"
  );

  Serial.println();

  Serial.println(
    "BASE     : 5-175"
  );

  Serial.println(
    "SHOULDER : 0-180"
  );

  Serial.println(
    "ELBOW    : 0-180"
  );

  Serial.println(
    "GRIPPER  : 70-180"
  );

  Serial.println();

  Serial.println(
    "SHOULDER DIRECTION: NORMAL"
  );

  Serial.println(
    "ELBOW DIRECTION: NORMAL"
  );

  Serial.println();

  Serial.println(
    "HOME_OK"
  );

  Serial.println(
    "BLUETOOTH_READY"
  );

  Serial.println(
    "ROBOT_READY"
  );

  Serial.println();
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
  readBluetooth();

  readSerial();

  updateServos();

  updateSequence();
}