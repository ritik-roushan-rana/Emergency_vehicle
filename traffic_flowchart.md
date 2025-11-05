# Two-Lane AI Traffic Control System - Flowchart

## System Overview
Your AI-based emergency vehicle detection and smart traffic control system operates on a two-lane road with LEFT and RIGHT lanes, managing both normal traffic flow and emergency vehicle prioritization.

## Main Traffic Control Flow

```mermaid
flowchart TD
    A[Start Video Processing] --> B[Capture Frame]
    B --> C[Run YOLOv8 Detection]
    C --> D[Extract Detections]
    D --> E[Count Vehicles by Lane]
    
    E --> F{Emergency Vehicle Detected?}
    
    F -->|Yes| G[Identify Emergency Lane]
    G --> H[Fetch Route to Destination]
    H --> I[Build Traffic Network]
    I --> J{Road Blocked?<br/>Both lanes > 10 vehicles}
    
    J -->|Yes| K[Flash BLUE Signal<br/>for Emergency Lane]
    K --> L[Display: EMERGENCY BLOCKED!<br/>CLEAR PATH - LANE PRIORITY]
    L --> M[Show Route Preview]
    
    J -->|No| N[Set Emergency Lane GREEN<br/>Other Lane RED]
    N --> O[Display: EMERGENCY VEHICLE<br/>PRIORITY: LANE]
    O --> M
    
    F -->|No| P{Heavy Congestion?<br/>Both lanes > 10 vehicles}
    
    P -->|Yes| Q[Check Congestion State]
    Q --> R{Current State?}
    
    R -->|BLOCKED or None| S[Set Both Signals RED]
    S --> T[Display: HEAVY CONGESTION<br/>DETECTED! WAITING TO CLEAR]
    T --> U{Can Clear Lane?}
    
    U -->|Left < 15, Right >= 15| V[State: CLEARING_LEFT<br/>Left GREEN, Right RED]
    U -->|Right < 15, Left >= 15| W[State: CLEARING_RIGHT<br/>Right GREEN, Left RED]
    U -->|Both >= 15| S
    
    V --> X[Display: CLEARING LEFT LANE<br/>FROM CONGESTION]
    W --> Y[Display: CLEARING RIGHT LANE<br/>FROM CONGESTION]
    
    X --> Z{Both Lanes < 15?}
    Y --> Z
    Z -->|Yes| AA[Reset to Normal Mode]
    Z -->|No| Q
    
    R -->|CLEARING_LEFT| AB{Left < 15?}
    AB -->|Yes| V
    AB -->|No| S
    
    R -->|CLEARING_RIGHT| AC{Right < 15?}
    AC -->|Yes| W
    AC -->|No| S
    
    P -->|No| AD{Road Blocked?<br/>Both lanes > 10}
    AD -->|Yes| AE[Both Signals RED<br/>Display: ROAD BLOCKED! ALL STOP]
    AD -->|No| AF[Normal Alternating Mode]
    
    AF --> AG{Time for Switch?<br/>Every 7 seconds}
    AG -->|Yes| AH[Switch Active Lane]
    AG -->|No| AI[Keep Current State]
    
    AH --> AJ{Last Green Lane?}
    AJ -->|LEFT| AK[RIGHT GREEN, LEFT RED]
    AJ -->|RIGHT| AL[LEFT GREEN, RIGHT RED]
    
    AK --> AM[Update Display]
    AL --> AM
    AI --> AM
    AE --> AM
    AA --> AM
    M --> AM
    
    AM --> AN[Update Multi-Junction Network<br/>TL1, TL2, TL3]
    AN --> AO[Draw Visualization]
    AO --> AP[Show Route Preview if Available]
    AP --> AQ{ESC Key Pressed?}
    
    AQ -->|No| B
    AQ -->|Yes| AR[End]

    style K fill:#0066ff,color:#ffffff
    style N fill:#00cc00,color:#ffffff
    style S fill:#ff0000,color:#ffffff
    style V fill:#00cc00,color:#ffffff
    style W fill:#00cc00,color:#ffffff
```

## Emergency Vehicle Priority States

```mermaid
stateDiagram-v2
    [*] --> Normal_Traffic
    
    Normal_Traffic --> Emergency_Detected : EV Detected
    
    Emergency_Detected --> Emergency_Clear_Road : Road Not Blocked
    Emergency_Detected --> Emergency_Blocked_Road : Road Blocked (Both > 10)
    
    Emergency_Clear_Road --> EV_Lane_Green : Set EV Lane GREEN
    EV_Lane_Green --> Route_Preview : Show Route & Hospital
    
    Emergency_Blocked_Road --> Flash_Blue_Signal : Flash BLUE for EV Lane
    Flash_Blue_Signal --> Clear_Path_Warning : Display Clear Path Message
    Clear_Path_Warning --> Route_Preview
    
    Route_Preview --> Multi_Junction_Propagation : Update TL1, TL2, TL3
    Multi_Junction_Propagation --> Normal_Traffic : EV Passes/No Longer Detected
    
    Normal_Traffic --> Heavy_Congestion : Both Lanes > 10, No EV
    
    Heavy_Congestion --> Blocked_State : Both RED
    Blocked_State --> Clearing_Left : Left < 15, Right >= 15
    Blocked_State --> Clearing_Right : Right < 15, Left >= 15
    Blocked_State --> Blocked_State : Both >= 15
    
    Clearing_Left --> Normal_Traffic : Both < 15
    Clearing_Right --> Normal_Traffic : Both < 15
    Clearing_Left --> Blocked_State : Left >= 15
    Clearing_Right --> Blocked_State : Right >= 15
```

## Multi-Junction Network Propagation

```mermaid
flowchart LR
    A[EV Detection] --> B[Estimate EV Distance]
    B --> C{Distance to TL1?}
    
    C -->|< 80m| D[TL1: Emergency Mode<br/>GREEN Signal]
    C -->|80-128m| E[TL1: YELLOW Signal]
    C -->|> 128m| F[TL1: Normal Cycle]
    
    D --> G{Distance to TL2?}
    G -->|< 200m| H[TL2: Emergency Mode<br/>YELLOW then GREEN]
    G -->|> 200m| I[TL2: Normal Cycle]
    
    H --> J{Distance to TL3?}
    J -->|< 320m| K[TL3: YELLOW Signal]
    J -->|> 320m| L[TL3: RED Signal]
    
    style D fill:#00cc00,color:#ffffff
    style H fill:#ffff00
    style K fill:#ffff00
```

## Signal Color Coding

- **RED**: Stop signal, no movement allowed
- **GREEN**: Go signal, movement allowed
- **YELLOW**: Warning/transition signal
- **BLUE**: Emergency priority signal (flashing for blocked emergency)

## Key Thresholds

- **Blocked Threshold**: 10 vehicles per lane
- **Clear Threshold**: 15 vehicles per lane
- **Emergency Trigger Distance**: 80m
- **Normal Signal Interval**: 7 seconds
- **TL1 Emergency Range**: < 80m
- **TL2 Emergency Range**: < 200m (80m + 120m)
- **TL3 Emergency Range**: < 320m (80m + 240m)

## Congestion States

1. **BLOCKED**: Both lanes > 10 vehicles, both signals RED
2. **CLEARING_LEFT**: Left lane < 15, getting GREEN priority
3. **CLEARING_RIGHT**: Right lane < 15, getting GREEN priority
4. **Normal**: Alternating signals every 7 seconds

This flowchart represents your comprehensive two-lane traffic management system with emergency prioritization and multi-junction coordination!
