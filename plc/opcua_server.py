"""Simulated PLC (stands in for the S7-1500).  Runs a 50 ms scan cycle and exposes all tags via OPC UA.

    python -m plc.opcua_server
"""
import asyncio
import logging
import time

from asyncua import Server, ua

import config
from plc.tags import CAMERA_POS, END_POS, GATE_POS, TAGS

SCAN = 0.05
AI_TIMEOUT = 8.0
PUSH_TIME = 1.0


class PLC:
    def __init__(self):
        self.nodes = {}
        self.cache = {}                 # last value written by the PLC (avoid needless writes)
        self.state, self.pos, self.t0 = 0, 0.0, 0.0
        self.item_id, self.cnt = 0, {"total": 0, "good": 0, "rej": 0}
        self.defect = False
        self.ai_done_item = False       # AI result already received for the current item

    async def setup(self):
        self.server = Server()
        await self.server.init()
        self.server.set_endpoint(config.OPCUA_ENDPOINT.replace("127.0.0.1", "0.0.0.0"))
        self.server.set_server_name("Fabric simulated PLC")
        idx = await self.server.register_namespace(config.OPCUA_NAMESPACE)
        plc = await self.server.nodes.objects.add_object(idx, "PLC")
        for name, (init, vt, writable, desc) in TAGS.items():
            node = await plc.add_variable(idx, name, ua.Variant(init, vt))
            if writable:
                await node.set_writable()
            self.nodes[name], self.cache[name] = node, init

    async def get(self, name):
        return await self.nodes[name].read_value()

    async def put(self, name, value):
        if self.cache[name] != value:
            self.cache[name] = value
            await self.nodes[name].write_value(ua.Variant(value, TAGS[name][1]))

    async def new_item(self):
        self.item_id += 1; self.pos = 0.0; self.defect = False; self.ai_done_item = False
        await self.put("Item_Id", self.item_id)
        await self.put("Camera_Trigger", False)
        await self.nodes["AI_Done"].write_value(ua.Variant(False, TAGS["AI_Done"][1]))   # clear the handshake bit

    async def pulse(self, name):
        if await self.get(name):
            await self.nodes[name].write_value(ua.Variant(False, TAGS[name][1]))
            self.cache[name] = False
            return True
        return False

    async def scan(self):
        """One PLC cycle: read inputs -> state machine -> write outputs."""
        start, stop, reset = await self.pulse("Start_Cmd"), await self.pulse("Stop_Cmd"), await self.pulse("Reset_Cmd")
        estop = await self.get("EStop")
        speed = max(0.0, min(100.0, float(await self.get("Belt_Speed"))))
        now = time.monotonic()

        if estop and self.state != 4:
            self.state = 4
        elif self.state == 4 and not estop and reset:
            self.state = 0
        elif self.state == 5 and reset:
            self.state = 0
        elif self.state == 0 and start and not estop:
            self.state = 1
            if self.item_id == 0 or self.pos >= END_POS:
                await self.new_item()
        elif self.state == 1:
            if stop:
                self.state = 0
            else:
                self.pos += speed * SCAN
                if self.pos >= CAMERA_POS and not self.ai_done_item:
                    self.pos = CAMERA_POS
                    await self.put("Camera_Trigger", True)
                    self.state, self.t0 = 2, now
                elif self.pos >= GATE_POS and self.defect:
                    self.pos = GATE_POS
                    self.state, self.t0 = 3, now
                elif self.pos >= END_POS:
                    self.cnt["total"] += 1; self.cnt["good"] += 1
                    await self.new_item()
        elif self.state == 2:                                # belt halted, waiting for the AI
            if await self.get("AI_Done"):
                self.defect = int(await self.get("AI_Result")) == 1
                self.ai_done_item = True
                await self.put("Camera_Trigger", False)
                self.state = 1
            elif now - self.t0 > AI_TIMEOUT:
                self.state = 5
            elif stop:
                self.state = 0
        elif self.state == 3:                                # pusher removes the defective item
            if now - self.t0 >= PUSH_TIME:
                self.cnt["total"] += 1; self.cnt["rej"] += 1
                await self.new_item()
                self.state = 1

        # ---- outputs ----
        running = self.state in (1,)
        await self.put("State", self.state)
        await self.put("Motor_Run", running)
        await self.put("Reject_Pusher", self.state == 3)
        await self.put("Lamp_Green", running)
        await self.put("Lamp_Red", self.state in (4, 5))
        await self.put("Item_Present", self.item_id > 0)
        await self.put("Item_Position", round(self.pos, 1))
        await self.put("Cnt_Total", self.cnt["total"]); await self.put("Cnt_Good", self.cnt["good"])
        await self.put("Cnt_Rejected", self.cnt["rej"])

    async def run(self):
        await self.setup()
        async with self.server:
            print(f"Simulated PLC running at {config.OPCUA_ENDPOINT}  (Ctrl+C to stop)")
            while True:
                await self.scan()
                await asyncio.sleep(SCAN)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(PLC().run())
    except KeyboardInterrupt:
        pass
