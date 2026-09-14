#ifndef CAMERA_PINS_H
#define CAMERA_PINS_H

/**
 * ══════════════════════════════════════════════════════════════════════════════
 * SmartCropVision ESP32-CAM Hardware Pin Definitions
 * ══════════════════════════════════════════════════════════════════════════════
 * Select your camera board model by uncommenting the appropriate macro below.
 * Default: CAMERA_MODEL_AI_THINKER (the standard ESP32-CAM with OV2640 module)
 */

#define CAMERA_MODEL_AI_THINKER // Standard ESP32-CAM with OV2640 module & baseboard
// #define CAMERA_MODEL_WROVER_KIT
// #define CAMERA_MODEL_ESP_EYE
// #define CAMERA_MODEL_M5STACK_PSRAM
// #define CAMERA_MODEL_M5STACK_V2_PSRAM
// #define CAMERA_MODEL_M5STACK_WIDE
// #define CAMERA_MODEL_M5STACK_ESP32CAM
// #define CAMERA_MODEL_TTGO_T_JOURNAL

#if defined(CAMERA_MODEL_AI_THINKER)
  #define PWDN_GPIO_NUM     32
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM      0
  #define SIOD_GPIO_NUM     26
  #define SIOC_GPIO_NUM     27

  #define Y9_GPIO_NUM       35
  #define Y8_GPIO_NUM       34
  #define Y7_GPIO_NUM       39
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       21
  #define Y4_GPIO_NUM       19
  #define Y3_GPIO_NUM       18
  #define Y2_GPIO_NUM        5
  #define VSYNC_GPIO_NUM    25
  #define HREF_GPIO_NUM     23
  #define PCLK_GPIO_NUM     22

  // Onboard Flash LED (Bright white illuminator on GPIO 4)
  #define FLASH_LED_PIN      4
  // Onboard Status LED (Red, inverted logic on GPIO 33)
  #define STATUS_LED_PIN    33

#elif defined(CAMERA_MODEL_WROVER_KIT)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     21
  #define SIOD_GPIO_NUM     26
  #define SIOC_GPIO_NUM     27

  #define Y9_GPIO_NUM       35
  #define Y8_GPIO_NUM       34
  #define Y7_GPIO_NUM       39
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       19
  #define Y4_GPIO_NUM       18
  #define Y3_GPIO_NUM        5
  #define Y2_GPIO_NUM        4
  #define VSYNC_GPIO_NUM    25
  #define HREF_GPIO_NUM     23
  #define PCLK_GPIO_NUM     22
  #define FLASH_LED_PIN     -1
  #define STATUS_LED_PIN    -1

#elif defined(CAMERA_MODEL_ESP_EYE)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM      4
  #define SIOD_GPIO_NUM     18
  #define SIOC_GPIO_NUM     23

  #define Y9_GPIO_NUM       36
  #define Y8_GPIO_NUM       37
  #define Y7_GPIO_NUM       38
  #define Y6_GPIO_NUM       39
  #define Y5_GPIO_NUM       35
  #define Y4_GPIO_NUM       14
  #define Y3_GPIO_NUM       13
  #define Y2_GPIO_NUM       34
  #define VSYNC_GPIO_NUM     5
  #define HREF_GPIO_NUM     27
  #define PCLK_GPIO_NUM     25
  #define FLASH_LED_PIN     -1
  #define STATUS_LED_PIN    -1

#else
  #error "No camera model selected! Please uncomment CAMERA_MODEL_AI_THINKER in camera_pins.h"
#endif

#endif // CAMERA_PINS_H
