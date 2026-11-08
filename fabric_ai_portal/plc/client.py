"""Thin OPC UA client used by the SCADA.  Same code works with the simulated PLC and a real S7-1500."""
from asyncua import ua
from asyncua.sync import Client

import config
from plc.tags import TAGS


class PLCClient:
    def __init__(self, endpoint=config.OPCUA_ENDPOINT, namespace=config.OPCUA_NAMESPACE, real=False):
        self.endpoint, self.namespace, self.real = endpoint, namespace, real
        self.client, self.nodes = None, {}

    def connect(self):
        self.client = Client(self.endpoint, timeout=3)
        self.client.connect()
        if self.real:      # S7-1500: global DB 'FabricPLC', addressed by string node ids
            self.nodes = {n: self.client.get_node(config.REAL_NODE_FMT.format(tag=n)) for n in TAGS}
            return
        idx = self.client.get_namespace_index(self.namespace)
        plc = self.client.nodes.objects.get_child([f"{idx}:PLC"])
        self.nodes = {n: plc.get_child([f"{idx}:{n}"]) for n in TAGS}

    def disconnect(self):
        try:
            if self.client:
                self.client.disconnect()
        except Exception:
            pass
        self.client = None

    def read_all(self):
        return {n: node.read_value() for n, node in self.nodes.items()}

    def write(self, name, value):
        self.nodes[name].write_value(ua.Variant(value, TAGS[name][1]))
