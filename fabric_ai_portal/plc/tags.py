"""Tag list shared by the simulated PLC (server) and the SCADA (client).
For a real S7-1500 keep the same names: create a DB 'FabricPLC' with these members and
expose it through the PLC's OPC UA server (see README, 'Real PLC')."""
from asyncua import ua

B, I16, I32, F = ua.VariantType.Boolean, ua.VariantType.Int16, ua.VariantType.Int32, ua.VariantType.Float

# name: (initial value, type, writable by SCADA, description)
TAGS = {
    # --- commands / data from SCADA to PLC (inputs of the PLC program) ---
    "Start_Cmd":      (False, B, True, "Start push-button (pulse)"),
    "Stop_Cmd":       (False, B, True, "Stop push-button (pulse)"),
    "EStop":          (False, B, True, "Emergency stop (latched while True)"),
    "Reset_Cmd":      (False, B, True, "Fault / E-stop reset (pulse)"),
    "Belt_Speed":     (20.0, F, True, "Conveyor speed, units/s (0-100 units = belt length)"),
    "AI_Done":        (False, B, True, "AI finished classifying the current item"),
    "AI_Result":      (0, I16, True, "AI result: 0 = good, 1 = defect"),
    "AI_Confidence":  (0.0, F, True, "AI probability of the predicted class"),
    # --- status from PLC to SCADA (outputs of the PLC program) ---
    "State":          (0, I16, False, "0 Idle, 1 Running, 2 Wait AI, 3 Reject, 4 E-Stop, 5 Fault"),
    "Motor_Run":      (False, B, False, "Conveyor motor contactor"),
    "Camera_Trigger": (False, B, False, "Fabric roll is under the camera -> AI must classify"),
    "Reject_Pusher":  (False, B, False, "Pneumatic pusher that removes defective items"),
    "Lamp_Green":     (False, B, False, "Green lamp: running"),
    "Lamp_Red":       (False, B, False, "Red lamp: fault / E-stop"),
    "Item_Present":   (False, B, False, "Item on the belt"),
    "Item_Position":  (0.0, F, False, "Item position 0-100"),
    "Item_Id":        (0, I32, False, "Running item number"),
    "Cnt_Total":      (0, I32, False, "Inspected items"),
    "Cnt_Good":       (0, I32, False, "Accepted items"),
    "Cnt_Rejected":   (0, I32, False, "Rejected items"),
}

STATES = {0: "IDLE", 1: "RUNNING", 2: "WAIT AI", 3: "REJECTING", 4: "E-STOP", 5: "FAULT"}
CAMERA_POS, GATE_POS, END_POS = 40.0, 75.0, 100.0
