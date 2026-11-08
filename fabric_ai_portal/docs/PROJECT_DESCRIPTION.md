# Fabric AI Portal - project description (20 p)

**Title:** AI-based visual quality inspection of fabric on a conveyor line, with a SCADA and an OPC UA PLC.

## a. Input signals
| Signal | Source | Type |
|---|---|---|
| Fabric image (camera frame) | camera at the inspection station | image 128x128 RGB (to the AI) |
| `Start_Cmd`, `Stop_Cmd`, `Reset_Cmd` | SCADA push-buttons | Bool (pulse) |
| `EStop` | SCADA emergency stop | Bool (latched) |
| `Belt_Speed` | SCADA slider | Real, 0-100 units/s |
| `AI_Done`, `AI_Result`, `AI_Confidence` | AI (PyTorch model) via SCADA | Bool, Int, Real |
| Item position / `Camera_Trigger` | position sensor (simulated) | Real / Bool |

## b. Output signals
| Signal | Meaning |
|---|---|
| `Motor_Run` | conveyor motor contactor |
| `Reject_Pusher` | pneumatic pusher valve |
| `Lamp_Green`, `Lamp_Red` | running / fault-E-stop lamps |
| `State` (0-5) | Idle, Running, Wait AI, Rejecting, E-Stop, Fault |
| `Cnt_Total`, `Cnt_Good`, `Cnt_Rejected` | production counters |
| AI verdict + probability | shown on the SCADA, stored in the statistics |

## c. Motion elements
1. Conveyor belt driven by a motor (speed adjustable).
2. Pneumatic pusher cylinder at the reject gate (extends 1 s, removes a defective item into the reject bin).
3. The camera station is fixed; the belt stops under the camera while the AI decides.

## d. Task
Every fabric piece travels on the belt, stops under the camera, is classified by a CNN as **good** or **defect**
(holes, stains, thread errors, tears) and is either passed to the OK bin or pushed into the reject bin.
The SCADA shows the live process, statistics and AI results and controls the line.

## e. Construction (architecture)
```
 AI (PyTorch CNN)  <--->  SCADA (PyQt5 + OPC UA client)  <--- OPC UA --->  PLC (simulated, or S7-1500)
   ai/                      scada/                                            plc/
```
Cycle for one item:
1. PLC moves the item; at the camera position it stops the belt and sets `Camera_Trigger`.
2. SCADA worker sees the trigger, grabs an image, runs the model (about 5 ms).
3. SCADA writes `AI_Result`, `AI_Confidence`, `AI_Done`.
4. PLC reads the result; good items continue to the OK bin, defective ones stop at the gate and the pusher fires.
5. Counters are updated, the next item enters the belt.

## f. Safety / fail-safe behaviour
* E-STOP latches the PLC in state 4 until released **and** reset.
* If the AI does not answer within 8 s the PLC goes to FAULT (red lamp) - a missing AI never lets items pass unchecked.
* SCADA reconnects automatically if the OPC UA link drops.
