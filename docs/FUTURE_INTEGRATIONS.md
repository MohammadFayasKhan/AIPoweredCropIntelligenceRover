# AI Powered Crop Intelligence Rover: Future Integrations Architecture

> **System Interfaces, Voice AI (XiaoZhi/MCP), and Third-Party Agronomic Ecosystems**  
> **Team:** Innovex | **SIH 2026 Challenge:** SIH26180  

---

## 1. XiaoZhi Voice Assistant & Model Context Protocol (MCP)

### Architecture Overview
The platform integrates with the **XiaoZhi** conversational voice AI framework via the standardized **Model Context Protocol (MCP v1)**. The voice interface enables farmers to ask natural language questions in English, Hindi, and regional dialects:

```text
[ Farmer Voice Input ] ──► [ XiaoZhi Voice Terminal ]
                                  │
                                  │ JSON-RPC / MCP v1 Tool Calls
                                  ▼
                     [ FastAPI XiaoZhi MCP Gateway ]
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
[ Live Field Status ]   [ Pesticide CIBRC Search ]  [ Spray Suitability ]
  • Temp & Humidity       • Chemical active ing.      • Wind speed & rain
  • Soil moisture %       • FRAC resistance code      • Optimal window
  • Rover GPS location    • Estimated budget / acre   • Temperature bounds
```

### Supported MCP Tools
* `get_current_rover_status`: Returns rover online state, GPS coordinates, and battery level.
* `get_sensor_status`: Fetches live temperature, relative humidity, soil moisture %, and rain accumulation.
* `get_crop_recommendation`: Executes Random Forest model on latest soil/weather telemetry.
* `search_pesticide_guidance`: Queries verified CIBRC pesticide database for approved dosages and application safety.
* `estimate_treatment_budget`: Computes estimated retail chemical expenditure in INR based on acreage.
* `search_weather_spray_suitability`: Evaluates ambient temperature, wind, and rain probability to advise if spraying is safe today.

---

## 2. Weather & Satellite Data Synchronization

1. **Open-Meteo & IMD Hyperlocal Weather API:**
   * Ingests 7-day hourly precipitation and wind forecasts to augment ground rain sensor data and predict high-risk sporulation windows.
2. **Sentinel-2 Multispectral Surface Reflectance:**
   * Syncs field GPS polygons with European Space Agency Sentinel-2 10-meter NDVI imagery to identify broad canopy vigor declines prior to ground rover dispatch.

---

## 3. Automated n8n Alerting Webhooks

* Integrates with **n8n** open-source workflow automation to send instant SMS, Telegram, and WhatsApp alert notifications to farmers when:
  * Sustained heat index exceeds critical thresholds ($>40^\circ\text{C}$).
  * High humidity ($>90\%$) persists for $>12$ hours, triggering Phytophthora fungal sporulation alerts.
  * Soil moisture drops below $15\%$, alerting for urgent irrigation scheduling.
