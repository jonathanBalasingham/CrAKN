import amd
from crakn.utils import BaseSettings
from typing import Literal

class CrAKNConfig(BaseSettings):
    name: Literal["crakn"]

def knowledge_graph():
    pass